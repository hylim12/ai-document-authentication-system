"""
AI-POWERED DOCUMENT AUTHENTICATION FOR AML SYSTEMS
Cleaned, tightened and improved for:
 - full OCR text capture
 - stronger NER heuristics
 - stricter final verdict rules (flags forged when OCR-box anomalies / clusters present)
 - readable consolidated report
"""

import os
import re
import datetime
import warnings
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings("ignore")

# Optional imports
try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None  # gracefully degrade

# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------
def safe_int(x):
    try:
        return int(x)
    except Exception:
        return 0

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ---------------------------------------------------------------------
# Main Detector class
# ---------------------------------------------------------------------
class DocumentForgeryDetector:
    def __init__(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        self.image_path = image_path
        self.log = []
        self.ocr_boxes = []        # [{'bbox':(x1,y1,x2,y2), 'text':..., 'conf':...}, ...]
        self.ocr_full_text = ""    # every OCR line joined
        self.ner_entities = {}     # {label: bbox}
        self.characters = []       # char-level physical segments
        self.binary = None
        self.gray = None
        self.original_image = None

        # analysis results
        self.anomalies = []
        self.background_anomalies = []
        self.ocr_box_anomalies = []
        self.suspicious_regions = []
        self.baseline_stats = None
        self.background_stats = None
        self.forgery_issues = []

        self._load_image()
        self.log.append(f"Loaded: {os.path.basename(self.image_path)} ({self.width}x{self.height})")
        self.final_verdict = "VERDICT NOT RUN"

    def _load_image(self):
        # Try PIL (robust formats), then fallback to OpenCV
        try:
            pil = Image.open(self.image_path)
            arr = np.array(pil)
            if arr.ndim == 2:
                self.gray = arr
                self.original_image = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
            else:
                rgb = arr[..., :3]
                self.original_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                self.gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        except Exception:
            img = cv2.imread(self.image_path)
            if img is None:
                raise ValueError(f"Could not load image: {self.image_path}")
            self.original_image = img
            self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.height, self.width = self.gray.shape

    # ---------------------------
    # OCR
    # ---------------------------
    def perform_ocr(self, use_paddle_if_available=True):
        """
        Runs PaddleOCR (if available). Stores:
         - self.ocr_boxes: explicit bounding boxes with text
         - self.ocr_full_text: ALL lines concatenated (every word captured)
        """
        self.log.append("OCR: Starting PaddleOCR (if available).")
        if not PaddleOCR or not use_paddle_if_available:
            self.ocr_full_text = ""
            self.log.append("OCR: PaddleOCR not available, skipping OCR.")
            return

        try:
            engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
            result = engine.ocr(self.image_path, cls=True)
            # result can be [ [ [box], (text, conf) ], ... ] — engine returns nested structure
            if not result or not result[0]:
                self.ocr_full_text = ""
                self.ocr_boxes = []
                self.log.append("OCR: No text detected by PaddleOCR.")
                return

            lines = []
            boxes = []
            for line in result[0]:
                box_pts, (text, conf) = line[0], line[1]
                # coerce coords and normalize
                xs = [int(round(p[0])) for p in box_pts]
                ys = [int(round(p[1])) for p in box_pts]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                text = (text or "").strip()
                if text == "":
                    continue
                boxes.append({'bbox': (x1, y1, x2, y2), 'text': text, 'conf': float(conf)})
                # split into words to ensure every word is captured
                # but keep the line structure as well
                lines.append(text)

            # store everything
            self.ocr_boxes = boxes
            # full text uses lines joined by newline, guaranteeing all OCR output is in report
            self.ocr_full_text = "\n".join(lines)
            self.log.append(f"OCR success: extracted {len(lines)} lines.")

            # quick logical checks (domain rules)
            self._run_logical_checks()

        except Exception as e:
            self.log.append(f"OCR Error: {e}")
            self.ocr_full_text = ""
            self.ocr_boxes = []

    def _run_logical_checks(self):
        """Simple domain-specific logical rules for suspicious content."""
        text = self.ocr_full_text.lower()
        issues = []
        # Example rules found in original code + a couple of common checks
        if 'shqiptare' in text and 'm' in text and 'shqiptar' not in text:
            issues.append("Logical inconsistency: male marker 'M' with 'shqiptare' (feminine).")
        if text.count('shkoder') >= 2:
            issues.append("Logical repetition: 'Shkoder' occurs multiple times.")
        if 'signature' in text and any(word in text for word in ['printed', 'signature']) and re.search(r'\b[A-Za-z]{3,}\b', text):
            # crude check for printed signature text label + name nearby
            issues.append("Printed signature label contains name-like text.")
        self.forgery_issues = issues
        if issues:
            self.log.append(f"OCR logical checks: found {len(issues)} issue(s).")

    # ---------------------------
    # Preprocessing
    # ---------------------------
    def preprocess_image(self):
        self.log.append("Preprocess: enhancing contrast + denoising + binarization.")
        img = self.original_image.copy()
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, None, h=6)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        gray = cv2.filter2D(gray, -1, kernel)
        self.gray = gray
        self.original_image = enhanced
        # Soft binarization (invert for text white on black)
        _, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        self.binary = binary
        self.log.append("Preprocess: complete.")

    # ---------------------------
    # Text regions (using OCR boxes when available)
    # ---------------------------
    def detect_text_lines_from_ocr(self):
        """
        Convert OCR boxes to line y-intervals. Falls back to contour-based lines if no OCR.
        """
        if not hasattr(self, "binary") or self.binary is None:
            self.preprocess_image()

        if not self.ocr_boxes:
            # fallback: dilate and find large contours => lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(self.width * 0.05), 1))
            dil = cv2.dilate(self.binary, kernel, iterations=1)
            contours, _ = cv2.findContours(dil.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            lines = []
            min_w = int(self.width * 0.1)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w > min_w and h > 5:
                    lines.append((max(0, y - 5), min(self.height, y + h + 5)))
            self.text_lines = sorted(lines, key=lambda x: x[0])
            self.log.append(f"Text region detection fallback: {len(self.text_lines)} lines.")
            return self.text_lines

        # merge OCR boxes into y-ranges (group lines by vertical overlap)
        y_groups = []
        for b in self.ocr_boxes:
            x1, y1, x2, y2 = b['bbox']
            mid = (y1 + y2) // 2
            merged = False
            for i, (s, e) in enumerate(y_groups):
                if abs(mid - ((s + e) // 2)) < 20:  # small vertical tolerance
                    y_groups[i] = (min(s, y1), max(e, y2))
                    merged = True
                    break
            if not merged:
                y_groups.append((y1, y2))
        self.text_lines = sorted(y_groups, key=lambda x: x[0])
        self.log.append(f"Text region detection (OCR-based): {len(self.text_lines)} lines.")
        return self.text_lines

    # ---------------------------
    # Character segmentation (contours)
    # ---------------------------
    def segment_characters(self):
        if not hasattr(self, "binary") or self.binary is None:
            self.preprocess_image()
        contours, _ = cv2.findContours(self.binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        chars = []
        min_area = max(1, int(self.height * self.width * 1e-5))
        max_area = int(self.height * self.width * 0.02)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < min_area or area > max_area or h < 6:
                continue
            # assign to nearest OCR line
            line_idx = -1
            ycenter = y + h // 2
            for idx, (ys, ye) in enumerate(getattr(self, "text_lines", [])):
                if ys <= ycenter <= ye:
                    line_idx = idx
                    break
            if line_idx == -1:
                continue
            roi_bin = self.binary[y:y+h, x:x+w]
            density = float(np.sum(roi_bin > 0)) / (w * h) if w*h else 0.0
            chars.append({
                'bbox': (x, y, x + w, y + h),
                'x': x, 'y': y, 'width': w, 'height': h,
                'density': density, 'line_idx': line_idx
            })
        self.characters = sorted(chars, key=lambda c: (c['line_idx'], c['x']))
        self.log.append(f"Segmented {len(self.characters)} characters.")
        return self.characters

    # ---------------------------
    # Background stats
    # ---------------------------
    def calculate_background_stats(self):
        if self.binary is None:
            self.preprocess_image()
        bg_mask = cv2.bitwise_not(self.binary)
        bg_pixels = self.gray[bg_mask > 0]
        if bg_pixels.size == 0:
            mean = float(np.mean(self.gray))
            std = float(np.std(self.gray))
        else:
            filtered = bg_pixels[bg_pixels < 250]
            if filtered.size < 100:
                mean, std = float(np.mean(bg_pixels)), float(np.std(bg_pixels))
            else:
                mean, std = float(np.mean(filtered)), float(np.std(filtered))
        self.background_stats = {'mean': mean, 'std': std}
        self.log.append(f"Background stats: μ={mean:.2f}, σ={std:.2f}")
        return self.background_stats

    # ---------------------------
    # Baseline (character) stats
    # ---------------------------
    def calculate_baseline_statistics(self):
        if not self.characters:
            self.baseline_stats = None
            return None
        heights = [c['height'] for c in self.characters]
        widths = [c['width'] for c in self.characters]
        densities = [c['density'] for c in self.characters] or [0.0]
        stats = {
            'height_mean': float(np.mean(heights)), 'height_std': float(np.std(heights)),
            'width_mean': float(np.mean(widths)), 'width_std': float(np.std(widths)),
            'density_mean': float(np.mean(densities)), 'density_std': float(np.std(densities)),
            'total_chars': len(self.characters)
        }
        self.baseline_stats = stats
        self.log.append("Baseline statistics computed.")
        return stats

    # ---------------------------
    # Detect background anomalies (line-level)
    # ---------------------------
    def detect_background_anomalies(self, sensitivity=3.0):
        if not self.background_stats:
            self.calculate_background_stats()
        bg_mean = self.background_stats['mean']
        bg_std = self.background_stats['std'] + 1e-6
        self.background_anomalies = []
        for idx, (ys, ye) in enumerate(getattr(self, 'text_lines', [])):
            roi = self.gray[ys:ye, :]
            roi_bg = cv2.bitwise_not(self.binary[ys:ye, :])
            pixels = roi[roi_bg > 0]
            if pixels.size == 0:
                continue
            mean = float(np.mean(pixels))
            z = abs(mean - bg_mean) / bg_std
            if z > sensitivity:
                self.background_anomalies.append({'line_idx': idx, 'bbox': (0, ys, self.width, ye),
                                                  'mean_intensity': mean, 'z_score': z,
                                                  'severity': 'high' if z > sensitivity * 1.5 else 'medium'})
        self.log.append(f"Detected {len(self.background_anomalies)} background anomalies.")
        return self.background_anomalies

    # ---------------------------
    # OCR box anomalies
    # ---------------------------
    def detect_ocr_box_anomalies(self, sensitivity=2.5):
        """Flag OCR boxes that are unusually large/small or have vertical placement jumps.
           Returns list of anomalies (each contains box and reasoning)."""
        boxes = [b for b in self.ocr_boxes if len(b.get('text','').strip()) > 0]
        if len(boxes) < 3:
            self.ocr_box_anomalies = []
            return []

        heights = [b['bbox'][3] - b['bbox'][1] for b in boxes]
        widths = [b['bbox'][2] - b['bbox'][0] for b in boxes]
        h_mean, h_std = float(np.mean(heights)), float(np.std(heights) + 1e-6)
        w_mean, w_std = float(np.mean(widths)), float(np.std(widths) + 1e-6)

        anomalies = []
        boxes_sorted = sorted(boxes, key=lambda b: (b['bbox'][1], b['bbox'][0]))  # top->down
        prev_center = None
        y_centers = []
        for i, b in enumerate(boxes_sorted):
            x1, y1, x2, y2 = b['bbox']
            h = y2 - y1
            w = x2 - x1
            cy = (y1 + y2) / 2
            y_centers.append(cy)
            scores = []
            reasons = []

            hz = abs(h - h_mean) / h_std
            wz = abs(w - w_mean) / w_std
            if hz > sensitivity:
                scores.append(hz); reasons.append('height_z')
            if wz > sensitivity:
                scores.append(wz); reasons.append('width_z')

            # big vertical jump relative to previous box indicates misplacement/paste
            if prev_center is not None:
                jump = abs(cy - prev_center)
                # estimate typical jump
                if len(y_centers) > 1:
                    typical_jump = np.mean(np.diff(y_centers)) if len(y_centers) > 1 else jump
                else:
                    typical_jump = jump
                if typical_jump > 0:
                    jump_z = abs(jump - typical_jump) / (np.std(np.diff(y_centers)) + 1e-6) if len(y_centers) > 2 else 0
                    if jump_z > sensitivity:
                        scores.append(jump_z); reasons.append('vertical_jump')

            prev_center = cy

            if scores:
                anomalies.append({'box': b, 'scores': scores, 'types': reasons, 'max_score': max(scores),
                                  'severity': 'high' if max(scores) > sensitivity * 1.5 else 'medium'})

        self.ocr_box_anomalies = anomalies
        self.log.append(f"OCR box anomalies detected: {len(anomalies)}")
        return anomalies

    # ---------------------------
    # Character anomalies (geometric + density)
    # ---------------------------
    def detect_character_anomalies(self, sensitivity=2.0):
        if not self.baseline_stats:
            self.calculate_baseline_statistics()
        if not self.baseline_stats:
            self.anomalies = []
            return []

        stats = self.baseline_stats
        anomalies = []
        for c in self.characters:
            scores = []
            reasons = []
            hz = abs(c['height'] - stats['height_mean']) / (stats['height_std'] + 1e-6)
            dz = abs(c.get('density', 0.0) - stats['density_mean']) / (stats['density_std'] + 1e-6)
            if hz > sensitivity:
                scores.append(hz); reasons.append('height')
            if dz > sensitivity:
                scores.append(dz); reasons.append('density')
            if scores:
                anomalies.append({'char': c, 'scores': scores, 'types': reasons, 'max_score': max(scores),
                                  'severity': 'high' if max(scores) > sensitivity * 1.5 else 'medium'})

        self.anomalies = sorted(anomalies, key=lambda a: a['max_score'], reverse=True)
        self.log.append(f"Character anomalies: {len(self.anomalies)}")
        return self.anomalies

    # ---------------------------
    # Cluster anomalies into suspicious regions
    # ---------------------------
    def cluster_anomalous_regions(self, proximity_factor=0.03):
        all_bboxes = []
        # char anomalies
        all_bboxes.extend([a['char']['bbox'] for a in self.anomalies])
        # background anomalies
        all_bboxes.extend([b['bbox'] for b in self.background_anomalies])
        # ocr box anomalies
        all_bboxes.extend([o['box']['bbox'] for o in self.ocr_box_anomalies])

        if not all_bboxes:
            self.suspicious_regions = []
            self.log.append("No anomalies to cluster.")
            return []

        centroids = [((x1+x2)/2, (y1+y2)/2) for x1,y1,x2,y2 in all_bboxes]
        used = set()
        clusters = []
        threshold = int(self.height * proximity_factor)

        for i in range(len(all_bboxes)):
            if i in used: continue
            x1,y1,x2,y2 = all_bboxes[i]
            cx,cy = centroids[i]
            cluster_idxs = {i}
            used.add(i)
            merged = True
            while merged:
                merged = False
                for j in range(len(all_bboxes)):
                    if j in used: continue
                    cx2, cy2 = centroids[j]
                    dist = np.hypot(cx - cx2, cy - cy2)
                    if dist < threshold:
                        cluster_idxs.add(j)
                        used.add(j)
                        merged = True
                        # expand bounding box
                        bx1,by1,bx2,by2 = all_bboxes[j]
                        x1, y1 = min(x1, bx1), min(y1, by1)
                        x2, y2 = max(x2, bx2), max(y2, by2)
                        # update centroid approx
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        break
            pad = 8
            clusters.append({'bbox': (max(0,x1-pad), max(0,y1-pad), min(self.width, x2+pad), min(self.height, y2+pad)),
                             'anomaly_count': len(cluster_idxs),
                             'severity_proxy': len(cluster_idxs)})
        self.suspicious_regions = sorted(clusters, key=lambda r: r['severity_proxy'], reverse=True)
        self.log.append(f"Clustered into {len(self.suspicious_regions)} suspicious regions.")
        return self.suspicious_regions

    # ---------------------------
    # NER improvements: more patterns + proximity-based extraction
    # ---------------------------
    def identify_critical_entities_from_ocr(self):
        """
        Rule-based NER using OCR boxes + neighbor boxes. Tries to find common fields:
         SURNAME, GIVEN NAME, NATIONALITY, DATE OF BIRTH, DATE OF ISSUE/EXPIRY, ID/PERSONAL NO, GENDER, AUTHORITY, SIGNATURE
        """
        if not self.ocr_boxes:
            self.log.append("NER: skipped (no OCR boxes).")
            return

        boxes = sorted(self.ocr_boxes, key=lambda b: (b['bbox'][1], b['bbox'][0]))  # top-to-bottom
        texts = [b['text'] for b in boxes]
        label_patterns = {
            r'mbiemri|surname': 'SURNAME',
            r'emri|given name': 'GIVEN NAME',
            r'shtetesia|nationalit': 'NATIONALITY',
            r'data.*lindj|date.*birth|date of birth': 'DATE OF BIRTH',
            r'data.*lesh|date.*issue|date of issue': 'DATE OF ISSUE',
            r'data.*skadim|date.*expir': 'DATE OF EXPIRY',
            r'nr\.?leternjoftim|card\.?no|nr\.?leter': 'ID CARD NO',
            r'nr\.?personal|personal\.?no': 'PERSONAL NO',
            r'gjinia|sex': 'GENDER',
            r'autoriteti|authority': 'AUTHORITY',
            r'firma|signature': 'SIGNATURE',
        }

        found = {}
        # helper to find nearest following text box (by index)
        for idx, box in enumerate(boxes):
            text = box['text'].strip()
            tlower = text.lower()
            for patt, label in label_patterns.items():
                if re.search(patt, tlower):
                    # look ahead up to 4 boxes for value
                    value = None
                    for j in range(1, 5):
                        if idx + j >= len(boxes): break
                        candidate = boxes[idx + j]['text'].strip()
                        if not candidate: continue
                        # small validation heuristics
                        if 'date' in label:
                            if re.search(r'\d{1,2}[-./]\d{1,2}[-./]\d{2,4}', candidate):
                                value = candidate; val_bbox = boxes[idx + j]['bbox']; break
                        elif label in ['GENDER'] and candidate.upper() in ['M', 'F']:
                            value = candidate; val_bbox = boxes[idx + j]['bbox']; break
                        elif label in ['ID CARD NO','PERSONAL NO'] and re.search(r'[A-Z0-9]{5,}', candidate.replace(' ',''), re.I):
                            value = candidate; val_bbox = boxes[idx + j]['bbox']; break
                        else:
                            # generic fallback — accept first non-empty
                            value = candidate; val_bbox = boxes[idx + j]['bbox']; break
                    if value:
                        found[label] = val_bbox
                        break

        # Additional ad-hoc extraction: numeric lines that look like ID or passport numbers
        for b in boxes:
            t = b['text'].strip()
            if re.match(r'^[A-Z0-9]{8,}$', t.replace(' ', '')) and ('ID CARD NO' not in found and 'PERSONAL NO' not in found):
                # pick the first sufficiently long alnum string
                if len(t) >= 8:
                    # decide label by length heuristic
                    if len(re.sub(r'\D', '', t)) >= 9:
                        found['PERSONAL NO'] = b['bbox']
                    else:
                        found['ID CARD NO'] = b['bbox']

        self.ner_entities = found
        self.log.append(f"NER: Identified {len(found)} critical fields.")

    # ---------------------------
    # Orchestration
    # ---------------------------
    def process_document(self,
                         char_sensitivity=2.0,
                         bg_sensitivity=3.0,
                         ocr_sensitivity=2.5):
        # Order: OCR first (for content rules, NER), then physical analysis
        self.perform_ocr()
        self.identify_critical_entities_from_ocr()
        self.detect_ocr_box_anomalies(sensitivity=ocr_sensitivity)

        # physical analysis
        self.preprocess_image()
        self.detect_text_lines_from_ocr()
        self.segment_characters()
        self.calculate_background_stats()
        self.detect_background_anomalies(sensitivity=bg_sensitivity)
        self.calculate_baseline_statistics()
        self.detect_character_anomalies(sensitivity=char_sensitivity)
        self.cluster_anomalous_regions()

    # ---------------------------
    # Visualization
    # ---------------------------
    def visualize_results(self, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Document')
        axes[0].axis('off')

        vis = self.original_image.copy()

        # NER boxes (yellow)
        for label, bbox in self.ner_entities.items():
            x1,y1,x2,y2 = [int(v) for v in bbox]
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,255), 2)
            cv2.putText(vis, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        # character anomalies (orange/red)
        for a in self.anomalies:
            x1,y1,x2,y2 = a['char']['bbox']
            color = (0,0,255) if 'density' in a['types'] else (255,165,0)
            cv2.rectangle(vis, (x1,y1), (x2,y2), color, 1)

        # background anomalies (green)
        for b in self.background_anomalies:
            x1,y1,x2,y2 = [int(v) for v in b['bbox']]
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)

        # ocr box anomalies (magenta)
        for o in self.ocr_box_anomalies:
            x1,y1,x2,y2 = [int(v) for v in o['box']['bbox']]
            cv2.rectangle(vis, (x1,y1), (x2,y2), (255,0,255), 2)

        # clustered suspicious regions (blue heavy)
        for r in self.suspicious_regions:
            x1,y1,x2,y2 = [int(v) for v in r['bbox']]
            cv2.rectangle(vis, (x1,y1), (x2,y2), (255,0,0), 3)
            cv2.putText(vis, "SUSPICIOUS", (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        axes[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Forgery Detection: {self.final_verdict} ({len(self.suspicious_regions)} regions)')
        axes[1].axis('off')

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        return fig

    # ---------------------------
    # Report generation (exhaustive OCR + clear verdict)
    # ---------------------------
    def generate_report(self):
        lines = []
        lines.append("="*80)
        lines.append("DOCUMENT FORGERY AUTHENTICATION REPORT (AI-Enhanced)")
        lines.append("="*80)
        lines.append(f"File    : {os.path.basename(self.image_path)}")
        lines.append(f"Resolution: {self.width} x {self.height}")
        lines.append(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("-"*80)
        lines.append("Execution Log:")
        for l in self.log:
            lines.append(f" - {l}")
        lines.append("-"*80)

        # NER
        if self.ner_entities:
            lines.append("Named Entities (scoped fields):")
            # map entity bounding boxes to their OCR text (best-effort)
            for label, bbox in self.ner_entities.items():
                value = None
                for b in self.ocr_boxes:
                    if b['bbox'] == bbox:
                        value = b['text']; break
                if not value:
                    # fallback: find text whose bbox overlaps the region
                    x1,y1,x2,y2 = bbox
                    for b in self.ocr_boxes:
                        bx1,by1,bx2,by2 = b['bbox']
                        if not (bx2 < x1 or bx1 > x2 or by2 < y1 or by1 > y2):
                            value = b['text']; break
                lines.append(f"  {label:15}: {value or 'UNKNOWN'}")
            lines.append("-"*80)

        # OCR FULL (no preview truncation)
        if self.ocr_full_text and self.ocr_full_text.strip():
            ocr_lines = [l.strip() for l in self.ocr_full_text.splitlines() if l.strip()]
            lines.append("OCR Text Extraction (FULL):")
            for i, ol in enumerate(ocr_lines, 1):
                lines.append(f"  {i:03d}. {ol}")
            lines.append("-"*80)
        else:
            lines.append("OCR Text Extraction: NO OCR RESULT")
            lines.append("-"*80)

        # logical check
        if self.forgery_issues:
            lines.append("Logical Content Issues:")
            for issue in self.forgery_issues:
                lines.append(f" - {issue}")
            lines.append("-"*80)
        else:
            lines.append("Logical Content Issues: None found.")
            lines.append("-"*80)

        # physical summary
        total_chars = len(self.characters)
        char_anom = len(self.anomalies)
        ink_anom = sum(1 for a in self.anomalies if 'density' in a['types'])
        bg_anom = len(self.background_anomalies)
        ocr_box_anom = len(self.ocr_box_anomalies)
        clusters = len(self.suspicious_regions)

        lines.append("Physical Tampering Summary:")
        if self.baseline_stats:
            s = self.baseline_stats
            lines.append(f" - Characters analyzed: {total_chars}")
            lines.append(f" - Avg char height: {s['height_mean']:.1f} ± {s['height_std']:.1f}")
            lines.append(f" - Avg ink/density: {s['density_mean']:.3f} ± {s['density_std']:.3f}")
        else:
            lines.append(" - Baseline statistics not available (insufficient characters).")
        lines.append(f" - Character anomalies: {char_anom} (density-flagged: {ink_anom})")
        lines.append(f" - Background anomalies (line-level): {bg_anom}")
        lines.append(f" - OCR box anomalies (alignment/size): {ocr_box_anom}")
        lines.append(f" - Clustered suspicious regions: {clusters}")
        lines.append("-"*80)

        # Final verdict logic (made stricter and transparent)
        # RULES:
        #  - If any logical content issues → FORGED (critical)
        #  - If OCR box anomalies >= 2 OR clusters >= 2 → FORGED
        #  - If clusters >= 3 or (ink anomalies >= 5) → HIGHLY SUSPICIOUS
        #  - else LIKELY AUTHENTIC with lower bound confidence 50%
        logical_forgery = len(self.forgery_issues) > 0
        if logical_forgery:
            verdict = "**FORGED / TAMPERED**"
            confidence = 99.0
            reason = "Critical logical content inconsistencies."
        elif ocr_box_anom >= 2 or clusters >= 2:
            verdict = "**FORGED / TAMPERED**"
            confidence = 95.0
            reason = "Multiple OCR-box anomalies or clusters indicate pasted/edited regions."
        else:
            suspicion_score = char_anom + (bg_anom*2) + (ocr_box_anom*3) + (clusters*4)
            if clusters >= 3 or ink_anom >= 5 or suspicion_score > 12:
                verdict = "**HIGHLY SUSPICIOUS / FORGED**"
                confidence = clamp(60 + suspicion_score * 2.5, 60, 98)
                reason = "Multiple physical anomalies clustered in critical regions."
            elif suspicion_score > 6:
                verdict = "SUSPICIOUS"
                confidence = clamp(50 + suspicion_score*2.0, 50, 90)
                reason = "Noticeable deviations from baseline statistics and a few localized anomalies."
            else:
                verdict = "LIKELY AUTHENTIC"
                confidence = clamp(90 - suspicion_score*2.0, 50, 95)
                reason = "Few and minor inconsistencies."

        self.final_verdict = verdict.strip('* ')
        lines.append("FINAL ASSESSMENT")
        lines.append("="*35)
        lines.append(f" VERDICT        : {verdict}")
        lines.append(f" CONFIDENCE     : {confidence:.1f}%")
        lines.append(f" PRIMARY REASON  : {reason}")
        lines.append("="*35)

        return "\n".join(lines)

# ---------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------
def ensure_output_folder(folder="PNG_results"):
    os.makedirs(folder, exist_ok=True)
    return folder

def analyze_single_document(document_path, char_sensitivity=2.0, bg_sensitivity=3.0, ocr_sensitivity=2.5):
    out_folder = ensure_output_folder()
    doc_name = os.path.splitext(os.path.basename(document_path))[0]
    output_png = os.path.join(out_folder, f"{doc_name}_enhanced_analysis.PNG")

    print("\n" + "="*80)
    print("     STARTING DOCUMENT FORGERY ANALYSIS (AI-ENHANCED)")
    print("="*80)

    try:
        detector = DocumentForgeryDetector(document_path)
    except Exception as e:
        print(f"FATAL ERROR loading document: {e}")
        return None

    detector.process_document(
        char_sensitivity=char_sensitivity,
        bg_sensitivity=bg_sensitivity,
        ocr_sensitivity=ocr_sensitivity
    )

    # Print full, clean report
    print(detector.generate_report())

    print("\n[INFO] Generating visualization...")
    detector.visualize_results(save_path=output_png)
    print(f"[INFO] Saved visualization to: {output_png}")
    print("="*80)
    print("               Analysis Complete!")
    print("="*80)

    plt.show(block=False)
    return detector

# ---------------------------------------------------------------------
# If run as script
# ---------------------------------------------------------------------
if __name__ == "__main__":
    sample = "input_docs/alb_id_02_fake_6_44.jpg"
    try:
        analyze_single_document(sample, char_sensitivity=2.0, bg_sensitivity=3.0, ocr_sensitivity=2.5)
    except FileNotFoundError:
        print("CRITICAL: Input document not found. Check path.")
    except Exception as e:
        print("General error:", e)
        import traceback; traceback.print_exc()
