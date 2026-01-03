import cv2
import numpy as np
import os
import json
import glob
import re
import shutil
from PIL import Image
from paddleocr import PaddleOCR
from collections import defaultdict

class DocumentProfiler:
    def __init__(self, input_dir="input_docs", output_dir="PNG_results"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # 1. Overwrite Logic: Clean the output directory safely
        self.cleanup_output_dir()
        
        # 2. Shared Engine Initialization (Matches your working example)
        self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
        
        self.field_data_accumulator = defaultdict(list)
        
        # NER Mapping for Albanian ID fields
        self.label_map = {
            r'mbiemri|surname': 'SURNAME',
            r'emri|given': 'GIVEN_NAME',
            r'shtet.sia|nationality': 'NATIONALITY',
            r'vendlindja|place': 'PLACE_OF_BIRTH',
            r'dat.*lindja|date.*birth': 'DATE_OF_BIRTH',
            r'dat.*leshimit|date.*issue': 'DATE_OF_ISSUE',
            r'dat.*skadimit|date.*expiry': 'DATE_OF_EXPIRY',
            r'nr.*let.rnjoftim|card.*no': 'ID_CARD_NO',
            r'nr.*personal|personal.*no': 'PERSONAL_NO',
            r'gjinia|sex': 'SEX',
            r'autoriteti|authority': 'AUTHORITY'
        }

        self.known_english_labels = {'surname', 'given name', 'nationality', 'place of birth', 'date of birth', 
                             'card no', 'personal no', 'sex', 'authority', 'date of issue', 'date of expiry'}

    def cleanup_output_dir(self):
        """Safely clears existing files without deleting the folder to avoid WinError 5."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            return
        for filename in os.listdir(self.output_dir):
            file_path = os.path.join(self.output_dir, filename)
            try:
                if os.path.isfile(file_path): os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(f"[-] Locked file skipped: {filename}")

    def preprocess_for_ocr(self, img_bgr):
      """Enhanced for faint text detection."""
      target_width = 2200  # Higher res for small text
      scale = target_width / img_bgr.shape[1]
      img = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

      lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
      l, a, b = cv2.split(lab)
      clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(10, 10))  # Stronger for IDs
      l = clahe.apply(l)
      enhanced = cv2.merge((l, a, b))
      enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

      # Sharpen for thin fonts
      kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
      sharpened = cv2.filter2D(enhanced, -1, kernel)

      # Bilateral denoising (better for preserving edges on IDs)
      denoised = cv2.bilateralFilter(sharpened, d=9, sigmaColor=75, sigmaSpace=75)

      gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

      # Optional: Mild adaptive threshold for very low-contrast areas
      gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

      return gray

    def extract_forensic_features(self, roi):
        """Extracts H_Mean, Ink_Mean, etc., for the ML standard."""
        if roi.size == 0: return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ink_mean = np.mean(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        density = np.sum(binary > 0) / binary.size if binary.size > 0 else 0
        return {
            "h": roi.shape[0],
            "w": roi.shape[1],
            "ar": roi.shape[1] / roi.shape[0] if roi.shape[0] > 0 else 0,
            "ink": ink_mean,
            "density": density
        }

    def process_authentic_batch(self):
        """Filters for authentic docs, processes them, and saves results."""
        all_files = glob.glob(os.path.join(self.input_dir, "*.*"))
        authentic_files = [f for f in all_files if "fake" not in os.path.basename(f).lower()]
        
        print(f"--- Profiling {len(authentic_files)} Authentic Documents ---")

        for file_path in authentic_files:
            try:
                # --- LOAD USING PIL (As in your example) ---
                pil_image = Image.open(file_path).convert("RGB")
                target_width = 1500
                w_percent = target_width / float(pil_image.size[0])
                target_height = int(pil_image.size[1] * w_percent)
                pil_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                # --- CONVERT TO BGR FOR CV2 ---
                img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                vis_img = img_bgr.copy()

                # --- ENHANCE FOR OCR (As in your example) ---
                enhanced_gray = self.preprocess_for_ocr(img_bgr)
                
                # --- RUN OCR ---
                result = self.ocr_engine.ocr(img_bgr, cls=True)
                
                if result and result[0]:
                    self.run_ner_mapping(img_bgr, vis_img, result[0])
                    save_path = os.path.join(self.output_dir, f"profiled_{os.path.basename(file_path)}")
                    cv2.imwrite(save_path, vis_img)
                    print(f"  [Processed] {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  [Error] Failed {file_path}: {e}")

        self.generate_standards_json()

    def run_ner_mapping(self, image, vis_img, ocr_data):
      """Enhanced NER: right + below search, better inline skip."""
      boxes = []
      for line in ocr_data:
          pts, (txt, conf) = line[0], line[1]
          x1, y1 = int(min(p[0] for p in pts)), int(min(p[1] for p in pts))
          x2, y2 = int(max(p[0] for p in pts)), int(max(p[1] for p in pts))
          boxes.append({"text": txt.strip(), "lower": txt.lower().strip(), "bbox": (x1, y1, x2, y2), "conf": conf})

      for i, box in enumerate(boxes):
          matched_field = None
          match_pattern = None
          for pattern, field_name in self.label_map.items():
              m = re.search(pattern, box["lower"])
              if m:
                  matched_field = field_name
                  match_pattern = pattern
                  # Draw label box in yellow
                  cv2.rectangle(vis_img, (box["bbox"][0], box["bbox"][1]),
                                (box["bbox"][2], box["bbox"][3]), (0, 255, 255), 2)
                  break

          if not matched_field:
              continue

          # --- CASE 1: Search to the right (horizontal, vertical align) ---
          best_val = None
          min_dist = 600
          ay_mid = (box["bbox"][1] + box["bbox"][3]) / 2
          ax_mid = (box["bbox"][0] + box["bbox"][2]) / 2

          for j, target in enumerate(boxes):
              if i == j: continue
              ty_mid = (target["bbox"][1] + target["bbox"][3]) / 2
              tx_mid = (target["bbox"][0] + target["bbox"][2]) / 2
              dist_x = target["bbox"][0] - box["bbox"][2]
              dist_y = target["bbox"][1] - box["bbox"][3]

              # Right search: aligned vertically
              if abs(ty_mid - ay_mid) < 40 and 5 < dist_x < 600:
                  if dist_x < min_dist:
                      min_dist = dist_x
                      best_val = target

              # Below search: aligned horizontally (new!)
              elif abs(tx_mid - ax_mid) < 100 and 0 < dist_y < 100:  # Tight tolerances for below
                  dist = dist_y  # Can use Euclidean if needed: math.sqrt(dist_x**2 + dist_y**2)
                  if dist < min_dist:
                      min_dist = dist
                      best_val = target

          value_roi = None
          value_text = None

          if best_val:
              # Found spatial match (right or below)
              vx1, vy1, vx2, vy2 = best_val["bbox"]
              value_roi = image[vy1:vy2, vx1:vx2]
              value_text = best_val["text"]
              cv2.rectangle(vis_img, (vx1, vy1), (vx2, vy2), (0, 255, 0), 3)
              cv2.putText(vis_img, matched_field, (vx1, vy1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

          else:
              # --- CASE 2: Inline value in same box (improved to skip false positives) ---
              text = box["text"]
              lower_text = box["lower"]
              m = re.search(match_pattern, lower_text)
              if m:
                  start_idx = m.end()
                  candidate = text[start_idx:].strip().strip(":/-").strip().lower()
                  # Skip if candidate is empty or matches known English label (e.g., 'surname')
                  if candidate and candidate not in self.known_english_labels:
                      value_text = text[start_idx:].strip().strip(":/-").strip()
                      value_roi = image[box["bbox"][1]:box["bbox"][3], box["bbox"][0]:box["bbox"][2]]
                      cv2.putText(vis_img, matched_field, (box["bbox"][0], box["bbox"][1] - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

          # Extract features if valid ROI
          if value_roi is not None and value_roi.size > 0:
              feat = self.extract_forensic_features(value_roi)
              if feat:
                  self.field_data_accumulator[matched_field].append(feat)
                  print(f"  [+] Extracted {matched_field}: {value_text[:30]}...")

    def generate_standards_json(self):
        """Aggregates all data into the master .json standard."""
        master = {}
        for field, obs in self.field_data_accumulator.items():
            master[field] = {
                "H_Mean": float(np.mean([o['h'] for o in obs])),
                "H_STD": float(np.std([o['h'] for o in obs])),
                "W_Mean": float(np.mean([o['w'] for o in obs])),
                "AR_Mean": float(np.mean([o['ar'] for o in obs])),
                "AR_STD": float(np.std([o['ar'] for o in obs])),
                "Ink_Mean": float(np.mean([o['ink'] for o in obs])),
                "Sample_Count": len(obs)
            }
        with open("profile_standard.json", "w") as f:
            json.dump(master, f, indent=4)
        print("\n--- SUCCESS: profile_standard.json generated ---")

if __name__ == "__main__":
    profiler = DocumentProfiler()
    profiler.process_authentic_batch()