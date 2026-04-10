"""
Project Title: AI-POWERED DOCUMENT AUTHENTICATION FOR ANTI-MONEY LAUNDERING (AML) SYSTEMS
Created By: Eldeena Lim Huey Yinn
Student ID: 1211111904

File: feature_engineering.py
Functionality: Image preprocessing, character segmentation, and statistical anomaly detection.
"""

# Import necessary libraries and modules
import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
import os
from PIL import Image
import re 
import datetime 
import json
import unicodedata
from prompts.regex_passport_patterns import LABEL_PATTERNS
from utils.calibration import calibrate_entities, derive_nationality, compute_risk_score
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None 
warnings.filterwarnings('ignore')


# LLM NER removed — using rule-based regex only
extract_passport_fields_llm = None

COUNTRY_REQUIRED_FIELDS = {
    "ALBANIA": [
        "SURNAME", "GIVEN NAME", "DATE OF BIRTH", "DATE OF ISSUE",
        "DATE OF EXPIRY", "SEX", "ID CARD NO", "PERSONAL NO"
    ],
    "LATVIA": [
        "SURNAME", "GIVEN NAME", "DATE OF BIRTH", "DATE OF ISSUE",
        "DATE OF EXPIRY", "SEX", "PASSPORT NO"
    ],
    "SLOVAKIA": [
        "SURNAME", "GIVEN NAME", "DATE OF BIRTH", "DATE OF ISSUE",
        "DATE OF EXPIRY", "SEX", "ID CARD NO", "PERSONAL NO"
    ]
}

COUNTRY_OPTIONAL_FIELDS = {
    "ALBANIA": ["PLACE OF BIRTH", "AUTHORITY", "SIGNATURE"],
    "LATVIA": ["HEIGHT", "PLACE OF BIRTH", "AUTHORITY", "PERSONAL NO", "SIGNATURE"],
    "SLOVAKIA": ["AUTHORITY", "SIGNATURE"]
}

class DocumentForgeryDetector:

    def __init__(self, image_path, ocr_engine=None, target_width=1500):
        """Initializes forensic storage and standardizes input resolution."""
        self.image_path = image_path
        self.ocr_engine = ocr_engine
        self.log = []
        self.ground_truth_label = self.get_ground_truth_label()
        self.log.append(f"- Ground Truth Label: {self.ground_truth_label}")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        try:
            # Standardize input for consistent feature extraction
            pil_image = Image.open(image_path).convert("RGB")
            target_width = max(1200, int(target_width))
            w_percent = target_width / float(pil_image.size[0])
            target_height = int(pil_image.size[1] * w_percent)

            pil_image = pil_image.resize(
                (target_width, target_height),
                Image.Resampling.LANCZOS
            )
            image_array = np.array(pil_image)
            self.original_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        except Exception:
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                raise ValueError(f"Failed to load image: {image_path}")

        if self.original_image.dtype != np.uint8:
            self.original_image = np.clip(self.original_image, 0, 255).astype(np.uint8)
        self.display_image = self.original_image.copy()
        self.gray_original = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.gray = self.gray_original.copy()
        self.height, self.width = self.gray_original.shape
        self.log.append(
            f"- Loaded & Resized: {os.path.basename(image_path)} ({self.width}x{self.height})"
        )

        # Internal state initialization
        self.characters = []
        self.text_lines = []
        self.anomalies = []
        self.background_anomalies = []
        self.ocr_box_anomalies = []
        self.baseline_stats = None
        self.background_stats = None
        self.ocr_results = None
        self.ocr_full_text = ""
        self.ocr_boxes = []
        self.ner_entities = {}
        self.suspicious_regions = []
        self.final_verdict = "VERDICT NOT RUN"
        self.forgery_features = {}
        self.forgery_issues = []
        self.risk_score = 0.0
        self.risk_issues = []
        self.llm_ner_disabled_reason = None

    def get_ground_truth_label(self):
        """
        Determines ground truth label from filename.
        If 'fake' in filename → FORGED
        Else → GENUINE
        """
        filename = os.path.basename(self.image_path).lower()

        if "fake" in filename:
            return "FORGED"
        return "GENUINE"



    def preprocess_image(self):
        """
        Enhances image quality using CLAHE and Otsu's Binarization.
        Generates OCR-only enhanced grayscale and binary mask.
        """
        self.log.append("- Preprocess: OCR-only enhancement (safe mode).")
        temp_img = self.original_image.copy()

        lab = cv2.cvtColor(temp_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))
        enhanced_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        gray_ocr = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        gray_ocr = cv2.fastNlMeansDenoising(gray_ocr, None, h=5)
        self.gray_ocr = gray_ocr
        _, binary = cv2.threshold(
            self.gray_ocr, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return self.binary

    
    def detect_text_regions(self):
        """Detects text line regions using OCR boxes or contour fallback."""
        if not self.ocr_boxes:
            if not hasattr(self, 'binary'): self.preprocess_image()
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(self.width * 0.05), 1))
            dilated_binary = cv2.dilate(self.binary, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            lines = []
            min_line_width = int(self.width * 0.10) 
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > min_line_width and h > 5:
                    lines.append((max(0, y - 5), min(self.height, y + h + 5)))
            self.text_lines = sorted(lines, key=lambda l: l[0])
            self.log.append(f"- Text region detection (Fallback): {len(self.text_lines)} lines.")
        else:
            line_y_ranges = {}
            for box in self.ocr_boxes:
                y1, y2 = box['bbox'][1], box['bbox'][3]
                merged = False
                for y_center, (start, end) in line_y_ranges.items():
                    if abs((y1 + y2)/2 - y_center) < 15: 
                        line_y_ranges[y_center] = (min(start, y1), max(end, y2))
                        merged = True
                        break
                if not merged:
                    line_y_ranges[(y1 + y2)/2] = (y1, y2)
            self.text_lines = sorted([(y1, y2) for y_center, (y1, y2) in line_y_ranges.items()], key=lambda l: l[0])
            self.log.append(f"- Text region detection (OCR-based): {len(self.text_lines)} lines.")
        return self.text_lines

    def segment_characters(self):
        """Performs contour analysis to isolate individual text elements."""
        contours, _ = cv2.findContours(self.binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        characters = []
        min_char_area = self.height * self.width * 0.00001 
        max_char_area = self.height * self.width * 0.015
        
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            area = w * h
            if area < min_char_area or area > max_char_area or h < 5: continue
            line_idx = -1
            for idx, (y_start, y_end) in enumerate(self.text_lines):
                y_center = y + h // 2
                if y_center >= y_start and y_center <= y_end:
                    line_idx = idx
                    break
            if line_idx == -1: continue
            aspect_ratio = w / h if h else 0
            char_roi_binary = self.binary[y:y+h, x:x+w]
            density = np.sum(char_roi_binary > 0) / (w * h) if (w * h) else 0
            
            characters.append({
                'x': x, 'y': y, 'width': w, 'height': h, 
                'aspect_ratio': aspect_ratio, 'density': density,
                'line_idx': line_idx, 'bbox': (x, y, x + w, y + h)
            })
        self.characters = sorted(characters, key=lambda c: (c['line_idx'], c['x'])) 
        self.log.append(f"- Segmented {len(self.characters)} characters.")
        return self.characters

    def calculate_character_ink_analysis(self):
        """Calculates character ink intensity and edge gradient magnitude."""
        gray = self.gray_original

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

        for char in self.characters:
            x1, y1, x2, y2 = char['bbox']
            char_roi_gray = gray[y1:y2, x1:x2]
            char_roi_binary = self.binary[y1:y2, x1:x2]
            ink_pixels = char_roi_gray[char_roi_binary > 0]
            char['mean_ink_intensity'] = np.mean(ink_pixels) if ink_pixels.size > 0 else 255
            grad_roi = gradient_magnitude[y1:y2, x1:x2]
            char['mean_gradient'] = np.mean(grad_roi[char_roi_binary > 0]) if grad_roi[char_roi_binary > 0].size > 0 else 0
        
        intensities = [c['mean_ink_intensity'] for c in self.characters]
        gradients = [c['mean_gradient'] for c in self.characters]
        densities = [c['density'] for c in self.characters]
        if self.baseline_stats is None: self.baseline_stats = {}
            

    def calculate_background_stats(self):
        """Calculates global background intensity mean and standard deviation."""
        background_mask = cv2.bitwise_not(self.binary)
        background_pixels = self.gray_original[background_mask > 0]
        filtered_pixels = background_pixels[background_pixels < 250] 

        if len(filtered_pixels) < 100:
            bg_mean = np.mean(background_pixels)
            bg_std = np.std(background_pixels)
        else:
            bg_mean = np.mean(filtered_pixels)
            bg_std = np.std(filtered_pixels)

        self.background_stats = { 'mean': bg_mean, 'std': bg_std }
        self.log.append(f"- Background stats: μ={bg_mean:.2f}, σ={bg_std:.2f}")
        return self.background_stats

    def detect_background_anomalies(self, sensitivity=3.0):
        """Detects text lines whose local background intensity deviates significantly."""
        if not self.background_stats: self.calculate_background_stats()

        bg_mean = self.background_stats['mean']
        bg_std = self.background_stats['std']
        self.background_anomalies = []

        for line_idx, (y_start, y_end) in enumerate(self.text_lines):
            line_roi_gray = self.gray_original[y_start:y_end, :]
            line_roi_binary = self.binary[y_start:y_end, :]
            line_bg_mask = cv2.bitwise_not(line_roi_binary)
            line_bg_pixels = line_roi_gray[line_bg_mask > 0]

            if len(line_bg_pixels) == 0: continue

            line_mean = np.mean(line_bg_pixels)
            z_score = abs(line_mean - bg_mean) / (bg_std + 1e-6)

            if z_score > sensitivity:
                self.background_anomalies.append({
                    'line_idx': line_idx,
                    'bbox': (0, y_start, self.width, y_end),
                    'mean_intensity': line_mean,
                    'z_score': z_score,
                    'severity': 'high' if z_score > sensitivity * 1.5 else 'medium'
                })
        self.log.append(f"- Detected {len(self.background_anomalies)} background anomalies.")
        return self.background_anomalies

    def calculate_baseline_statistics(self):
        """Calculates statistics with a zero-initialization fallback to prevent crashes."""
        # Initialize with neutral defaults to prevent NoneType errors in detect_anomalies
        defaults = {
            'height_mean': 0, 'height_std': 1, 'width_mean': 0, 'width_std': 1,
            'aspect_ratio_mean': 0, 'aspect_ratio_std': 1, 'density_mean': 0, 
            'density_std': 1, 'ink_mean': 127, 'ink_std': 1, 'grad_mean': 0, 'grad_std': 1
        }
        
        if not self.characters:
            self.baseline_stats = defaults
            return defaults

        # Calculate actual values if characters exist
        self.calculate_character_ink_analysis()
        h = [c['height'] for c in self.characters]
        w = [c['width'] for c in self.characters]
        ar = [c['aspect_ratio'] for c in self.characters]
        d = [c['density'] for c in self.characters]
        i = [c.get('mean_ink_intensity', 127) for c in self.characters]

        self.baseline_stats = {
            'height_mean': np.mean(h), 'height_std': np.std(h),
            'width_mean': np.mean(w), 'width_std': np.std(w),
            'aspect_ratio_mean': np.mean(ar), 'aspect_ratio_std': np.std(ar),
            'density_mean': np.mean(d), 'density_std': np.std(d),
            'ink_mean': np.mean(i), 'ink_std': np.std(i),
            'grad_mean': np.mean([c.get('mean_gradient', 0) for c in self.characters]),
            'grad_std': np.std([c.get('mean_gradient', 0) for c in self.characters])
        }
        return self.baseline_stats

    def detect_anomalies(self, sensitivity=2.0):
        """
        FORENSIC CHECK: Compares characters against statistical baselines and (if available) NER-standard constraints.
        """
        if not self.baseline_stats:
            self.calculate_baseline_statistics()

        stats = self.baseline_stats
        anomalies = []

        line_baselines = defaultdict(list)
        for c in self.characters:
            line_baselines[c['line_idx']].append(c['y'] + c['height'])

        line_avg_y = {
            idx: np.mean(vals) for idx, vals in line_baselines.items()
        }

        for char in self.characters:
            scores = []
            types = []
            h_z = abs(char['height'] - stats['height_mean']) / (stats['height_std'] + 1e-6)
            if h_z > sensitivity:
                scores.append(h_z)
                types.append('GLOBAL_HEIGHT_OUTLIER')

            w_z = abs(char['width'] - stats['width_mean']) / (stats['width_std'] + 1e-6)
            if w_z > sensitivity:
                scores.append(w_z)
                types.append('GLOBAL_WIDTH_OUTLIER')

            ar_z = abs(char['aspect_ratio'] - stats['aspect_ratio_mean']) / (stats['aspect_ratio_std'] + 1e-6)
            if ar_z > sensitivity:
                scores.append(ar_z)
                types.append('GLOBAL_ASPECT_RATIO_OUTLIER')

            ink_z = abs(char.get('mean_ink_intensity', stats['ink_mean']) - stats['ink_mean']) / (stats['ink_std'] + 1e-6)
            if ink_z > sensitivity:
                scores.append(ink_z)
                types.append('INK_INTENSITY_ANOMALY')

            grad_z = abs(char.get('mean_gradient', stats['grad_mean']) - stats['grad_mean']) / (stats['grad_std'] + 1e-6)
            if grad_z > sensitivity:
                scores.append(grad_z)
                types.append('EDGE_GRADIENT_ANOMALY')

            char_base = char['y'] + char['height']
            if abs(char_base - line_avg_y.get(char['line_idx'], char_base)) > stats['height_std']:
                scores.append(2.0)
                types.append('BASELINE_DRIFT')

            if scores:
                anomalies.append({
                    'char': char,
                    'scores': scores,
                    'types': types,
                    'max_score': max(scores),
                    'severity': 'high' if max(scores) > sensitivity * 1.5 else 'medium'
                })

        self.anomalies = sorted(anomalies, key=lambda x: x['max_score'], reverse=True)
        return self.anomalies


    def detect_ocr_box_anomalies(self, sensitivity=2.5):
        """Analyzes PaddleOCR boxes for size and vertical placement inconsistencies."""
        if len(self.ocr_boxes) < 5:
            self.ocr_box_anomalies = []; self.log.append("- OCR box anomalies detected: 0"); return []

        valid_boxes = []; heights = []; widths = []
        for box in self.ocr_boxes:
            x1, y1, x2, y2 = box['bbox']; h = y2 - y1; w = x2 - x1
            if h > 5 and w > 5 and len(box['text'].strip()) > 1: 
                box['height'] = h; box['width'] = w; box['y_center'] = (y1 + y2) / 2
                heights.append(h); widths.append(w); valid_boxes.append(box)

        if len(valid_boxes) < 5:
            self.ocr_box_anomalies = []; self.log.append("- OCR box anomalies detected: 0"); return []

        h_mean, h_std = np.mean(heights), np.std(heights)
        w_mean, w_std = np.mean(widths), np.std(widths)
        y_centers_valid = [box['y_center'] for box in valid_boxes]
        y_diffs = np.diff(y_centers_valid) 
        y_diff_mean, y_diff_std = (np.mean(y_diffs), np.std(y_diffs)) if len(y_diffs) > 0 else (0, 1e-6)

        anomalies = []
        for i, box in enumerate(valid_boxes): 
            scores = []; types = []
            h_z = abs(box['height'] - h_mean) / (h_std + 1e-6)
            w_z = abs(box['width'] - w_mean) / (w_std + 1e-6)
            if h_z > sensitivity: scores.append(h_z); types.append('ocr_height')
            if w_z > sensitivity: scores.append(w_z); types.append('ocr_width')

            if i > 0 and y_diff_std > 1e-6:
                y_jump = box['y_center'] - valid_boxes[i-1]['y_center']
                y_z = abs(y_jump - y_diff_mean) / (y_diff_std + 1e-6)
                if y_z > sensitivity: scores.append(y_z); types.append('ocr_alignment_jump')

            if scores:
                 anomalies.append({
                     'box': box, 'scores': scores, 'types': types,
                     'max_score': max(scores),
                     'severity': 'high' if max(scores) > sensitivity * 1.5 else 'medium'
                 })
        
        self.ocr_box_anomalies = anomalies
        self.log.append(f"- OCR box anomalies detected: {len(anomalies)}")
        return self.ocr_box_anomalies

    def cluster_anomalous_regions(self, proximity_factor=0.035):
        """Clusters proximal anomalies using a normalized proximity threshold."""
        proximity_threshold = int(self.height * proximity_factor) 
        
        all_bboxes = []
        if self.anomalies: all_bboxes.extend([a['char']['bbox'] for a in self.anomalies])
        if self.background_anomalies: all_bboxes.extend([b['bbox'] for b in self.background_anomalies])
        if self.ocr_box_anomalies: all_bboxes.extend([o['box']['bbox'] for o in self.ocr_box_anomalies])

        if not all_bboxes:
            self.suspicious_regions = []; self.log.append("- Clustered into 0 suspicious regions."); return []
        
        centroids = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in all_bboxes]
        regions = []; used_indices = set()

        for i in range(len(all_bboxes)):
            if i in used_indices: continue

            current_cluster_indices = {i}; used_indices.add(i)
            x1, y1, x2, y2 = all_bboxes[i]
            seed_centroid = centroids[i]

            did_merge = True
            while did_merge:
                did_merge = False
                for j in range(len(all_bboxes)):
                    if j in used_indices: continue
                    c2 = centroids[j]
                    distance = np.sqrt((seed_centroid[0] - c2[0])**2 + (seed_centroid[1] - c2[1])**2)
                    
                    if distance < proximity_threshold:
                        current_cluster_indices.add(j); used_indices.add(j); did_merge = True
                        bx1, by1, bx2, by2 = all_bboxes[j]
                        x1, y1 = min(x1, bx1), min(y1, by1)
                        x2, y2 = max(x2, bx2), max(y2, by2)
                        break 
            
            pad = 10
            regions.append({
                'bbox': (max(0, x1 - pad), max(0, y1 - pad), min(self.width, x2 + pad), min(self.height, y2 + pad)),
                'anomaly_count': len(current_cluster_indices),
                'severity_proxy': len(current_cluster_indices)
            })

        self.suspicious_regions = sorted(regions, key=lambda x: x['severity_proxy'], reverse=True)
        self.log.append(f"- Clustered into {len(regions)} suspicious regions.")
        return self.suspicious_regions

    def _is_header_text(self, text):
        text = str(text or "").upper().replace(" ", "")

        blacklist = [
            "REPUBLIK",
            "REPUBLIC",
            "SHQIP",
            "ALBANIA",
            "LETERNJOFTIM",
            "PASSPORT",
            "LATVIJ",
            "SLOVENSK"
        ]

        # Reject long uppercase strings (very important)
        if len(text) > 15:
            return True

        return any(word in text for word in blacklist)

    def _validate_dates(self, entities):
        def _parse_date(value):
            cleaned = str(value or "").strip().replace(".", "-").replace("/", "-")
            return datetime.strptime(cleaned, "%d-%m-%Y")

        try:
            dob = _parse_date(entities["DATE OF BIRTH"]["text"])
            issue = _parse_date(entities["DATE OF ISSUE"]["text"])
            expiry = _parse_date(entities["DATE OF EXPIRY"]["text"])

            # Logical rule
            if not (dob < issue < expiry):
                # 🚫 Remove incorrect assignments
                if issue <= dob:
                    entities.pop("DATE OF ISSUE", None)

                if expiry <= issue:
                    entities.pop("DATE OF EXPIRY", None)

        except Exception:
            pass

    def _is_passport(self, boxes):
        mrz_lines = [b for b in boxes if "<" in b["text"] and len(b["text"]) > 20]
        return len(mrz_lines) >= 2

    def _parse_date_from_text(self, text):
        cleaned = str(text or "").strip().replace(".", "-").replace("/", "-")
        if not re.fullmatch(r"\d{2}-\d{2}-\d{4}", cleaned):
            return None
        try:
            return datetime.strptime(cleaned, "%d-%m-%Y")
        except Exception:
            return None

    def identify_critical_entities_from_ocr(self, print_summary=True):
        """
        Extracts and validates structured information from OCR results using geometric proximity and pattern matching.
        """

        def normalize(text):
            # Standardizes text for robust matching.
            text = text.upper()
            text = unicodedata.normalize("NFKD", text)
            text = "".join(c for c in text if not unicodedata.combining(c))
            text = re.sub(r'[^A-Z0-9 ]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

        # Sanitize and validate bounding box geometry to prevent slicing errors
        boxes = []
        for box in self.ocr_boxes:
            bbox = box.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = map(int, map(round, bbox))
            except Exception:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            text_clean = box.get("text", "").strip()

            # 🚫 REMOVE HEADER TEXT EARLY
            if self._is_header_text(text_clean):
                continue

            boxes.append({
                "text": text_clean,
                "norm": normalize(text_clean),
                "bbox": (x1, y1, x2, y2),
                "confidence": box.get("confidence", 0.0)
            })

        self.ocr_boxes = boxes
        if not boxes:
            self.ner_entities = {}
            self.ner_metrics = {}
            self.missing_ner_fields = []
            return

        filtered_boxes = []
        for box in boxes:
            if self._is_header_text(box["text"]):
                continue
            filtered_boxes.append(box)

        boxes = filtered_boxes
        self.ocr_boxes = boxes
        if not boxes:
            self.ner_entities = {}
            self.ner_metrics = {}
            self.missing_ner_fields = []
            return

        def y_mid(b): return (b[1] + b[3]) // 2
        country = self.detect_country(self.ocr_full_text)
        required_fields = COUNTRY_REQUIRED_FIELDS.get(country, [])
        optional_fields = COUNTRY_OPTIONAL_FIELDS.get(country, [])
        width, height = self.width, self.height
        def is_label_like(text):
            for patterns in LABEL_PATTERNS.values():
                for p in patterns:
                    if re.search(p, text):
                        return True
            return False

        def get_region(box):
            x_center = (box["bbox"][0] + box["bbox"][2]) / 2
            y_center = (box["bbox"][1] + box["bbox"][3]) / 2

            if y_center < height * 0.15:
                return "HEADER"
            if y_center > height * 0.8:
                return "FOOTER"
            if x_center < width * 0.5:
                return "LEFT"
            return "RIGHT"

        # Local alias prevents any undefined-name issues in nested matching loops.
        is_valid_for_field = self._is_valid_for_field

        # Direct Entity Extraction: Identify fields with distinct, globally unique patterns
        entities = {}
        used = set()
        used_boxes = set()
        label_map = {}
        for field, patterns in LABEL_PATTERNS.items():
            for p in patterns:
                label_map[p] = field

        def assign_if_valid(field, box, idx=None):
            # 🚫 DO NOT override strong values
            if field in entities:
                return

            # 🚫 prevent ID ↔ Personal swap
            if field == "PERSONAL NO" and "ID CARD NO" in entities:
                if box["text"] == entities["ID CARD NO"]["text"]:
                    return

            if field == "ID CARD NO" and "PERSONAL NO" in entities:
                if box["text"] == entities["PERSONAL NO"]["text"]:
                    return

            if self._is_header_text(box["text"]):
                return

            if is_label_like(box["norm"]):
                return

            if not self._is_valid_for_field(field, box["text"]):
                return

            entities[field] = box

        # 🚀 COUNTRY-SPECIFIC EXTRACTION
        for field in required_fields:
            candidate = self.match_field_by_label(field, boxes)
            if candidate:
                entities[field] = candidate

        # DIRECT LABEL → VALUE EXTRACTION (SCORING-BASED)
        for i, box in enumerate(boxes):
            for pattern, field in label_map.items():
                if re.search(pattern, box["norm"]):

                    for j, candidate in enumerate(boxes):
                        if j == i:
                            continue

                        dy = candidate["bbox"][1] - box["bbox"][3]
                        dx = candidate["bbox"][0] - box["bbox"][0]

                        score = 0
                        if 0 <= dx <= 400:
                            score += 2
                        if abs(dy) <= 80:
                            score += 2
                        if 0 <= dy <= 150:
                            score += 1

                        if score < 2:
                            continue

                        if is_label_like(candidate["norm"]):
                            continue

                        if not self._is_valid_for_field(field, candidate["text"]):
                            continue

                        assign_if_valid(field, candidate)
                        used.add(j)
                        break

        # 🚀 FALLBACK RULES
        for box in boxes:
            text = box["text"].strip().upper()

            if country == "LATVIA":
                if re.fullmatch(r'[A-Z]{2}\d{7}', text):
                    entities["PASSPORT NO"] = box
                if re.fullmatch(r'\d{6}-\d{5}', text):
                    entities["PERSONAL NO"] = box
                if text in ["M", "F"]:
                    entities["SEX"] = box
                if re.fullmatch(r'\d{3}', text):
                    entities["HEIGHT"] = box

            if country == "ALBANIA":
                if re.fullmatch(r'\d{6,10}', text):
                    entities["ID CARD NO"] = box
                if re.fullmatch(r'[A-Z]\d{7,9}[A-Z]', text):
                    entities["PERSONAL NO"] = box

            if country == "SLOVAKIA":
                if re.fullmatch(r'[A-Z]{2}\d{6}', text):
                    entities["ID CARD NO"] = box
                if re.fullmatch(r'\d{6}/\d{4}', text):
                    entities["PERSONAL NO"] = box

        # Strong regex extraction (country-independent)
        for i, box in enumerate(boxes):
            text = box["norm"]

            # -------------------------
            # ID CARD / PASSPORT NUMBER
            # -------------------------
            if re.fullmatch(r'\d{6,10}', text):
                assign_if_valid("ID CARD NO", box)
                used.add(i)

            # -------------------------
            # PERSONAL NUMBER
            # -------------------------
            if re.fullmatch(r'[A-Z]\d{7,9}[A-Z]', text):
                assign_if_valid("PERSONAL NO", box)
                used.add(i)

            # -------------------------
            # DATE
            # -------------------------
            if re.search(r'\d{2}[./-]\d{2}[./-]\d{4}', text):
                if "DATE OF BIRTH" not in entities:
                    assign_if_valid("DATE OF BIRTH", box)
                elif "DATE OF ISSUE" not in entities:
                    assign_if_valid("DATE OF ISSUE", box)
                elif "DATE OF EXPIRY" not in entities:
                    assign_if_valid("DATE OF EXPIRY", box)

            # -------------------------
            # SEX
            # -------------------------
            if text in ["M", "F"]:
                entities["SEX"] = box

            # HEIGHT detection (optional, country-dependent)
            if re.search(r'\b\d{3}\b', text):
                near_height_hint = any(
                    re.search(p, text) for p in [r'HEIGHT', r'TAILLE', r'AUGUMS', r'CM']
                )
                if not near_height_hint:
                    for neighbor in boxes:
                        if neighbor is box:
                            continue
                        ny = y_mid(neighbor["bbox"])
                        by = y_mid(box["bbox"])
                        if abs(ny - by) > 35:
                            continue
                        if abs(neighbor["bbox"][0] - box["bbox"][0]) > 250:
                            continue
                        if any(re.search(p, neighbor["norm"]) for p in [r'HEIGHT', r'TAILLE', r'AUGUMS', r'CM']):
                            near_height_hint = True
                            break
                if near_height_hint and "HEIGHT" not in entities:
                    entities["HEIGHT"] = box

        # STRICT label-based date extraction ONLY
        for i, box in enumerate(boxes):
            text = box["norm"]

            if "BIRTH" in text:
                for candidate in boxes:
                    dy = candidate["bbox"][1] - box["bbox"][3]
                    if 0 <= dy <= 60:
                        assign_if_valid("DATE OF BIRTH", candidate)

            elif "ISSUE" in text:
                for candidate in boxes:
                    dy = candidate["bbox"][1] - box["bbox"][3]
                    if 0 <= dy <= 60:
                        assign_if_valid("DATE OF ISSUE", candidate)

            elif "EXPIRY" in text or "SKAD" in text:
                for candidate in boxes:
                    dy = candidate["bbox"][1] - box["bbox"][3]
                    if 0 <= dy <= 60:
                        assign_if_valid("DATE OF EXPIRY", candidate)

        # MRZ detection
        mrz_lines = [b for b in boxes if "<" in b["text"] and len(b["text"]) > 20]

        if len(mrz_lines) >= 2:
            mrz_data = self.mrz_parse_(mrz_lines)

            # 🚀 MRZ IS GROUND TRUTH
            for field, value in mrz_data.items():
                if value:
                    entities[field] = {
                        "text": value,
                        "bbox": mrz_lines[0]["bbox"],  # approximate
                        "confidence": 0.99  # HIGH CONFIDENCE
                    }
            entities["MRZ LINE 1"] = mrz_lines[0]
            entities["MRZ LINE 2"] = mrz_lines[1]
            if country == "LATVIA":
                self.log.append("- Country detected: LATVIA (optional HEIGHT expected if present).")

        if self._is_passport(boxes):
            # 🚫 If passport detected, disable PERSONAL NO from OCR
            if "PERSONAL NO" in entities:
                del entities["PERSONAL NO"]

            # Force names from MRZ only
            if "SURNAME" in entities and entities["SURNAME"].get("confidence", 0) < 0.95:
                del entities["SURNAME"]

            if "GIVEN NAME" in entities and entities["GIVEN NAME"].get("confidence", 0) < 0.95:
                del entities["GIVEN NAME"]
        # GIVEN NAME (STRONG FIX)
        for i, box in enumerate(boxes):
            if "GIVEN" in box["norm"] or "EMRI" in box["norm"]:
                best_candidate = None
                best_score = -999

                for j, candidate in enumerate(boxes):
                    if j == i:
                        continue
                    dy = candidate["bbox"][1] - box["bbox"][3]
                    dx = abs(candidate["bbox"][0] - box["bbox"][0])

                    if 0 <= dy <= 120 and dx <= 200:
                        if not is_valid_for_field("GIVEN NAME", candidate["text"]):
                            continue
                        score = self._score_candidate("GIVEN NAME", candidate, box)
                        if score > best_score:
                            best_score = score
                            best_candidate = candidate

                if best_candidate:
                    entities["GIVEN NAME"] = best_candidate
                    break

        # SURNAME (STRONG FIX)
        for i, box in enumerate(boxes):
            if any(k in box["norm"] for k in ["SURNAME", "MBIEMR"]):
                for j, candidate in enumerate(boxes):
                    if j == i:
                        continue
                    dy = candidate["bbox"][1] - box["bbox"][3]
                    dx = abs(candidate["bbox"][0] - box["bbox"][0])
                    if 0 <= dy <= 70 and dx <= 120 and is_valid_for_field("SURNAME", candidate["text"]):
                        assign_if_valid("SURNAME", candidate)
                        used.add(j)
                        break
                if "SURNAME" in entities:
                    break

        # 🚀 GIVEN NAME DERIVATION FROM SAME LINE
        if "SURNAME" in entities and "GIVEN NAME" not in entities:
            surname = entities["SURNAME"]["text"]

            for box in boxes:
                text = box["text"].strip()

                # Skip same word
                if text == surname:
                    continue

                # Look for multi-word name (e.g. "John Agani")
                if surname in text and len(text.split()) >= 2:
                    parts = text.split()

                    # Assume last word = surname
                    if parts[-1] == surname:
                        given_name = " ".join(parts[:-1])

                        entities["GIVEN NAME"] = {
                            "text": given_name,
                            "bbox": box["bbox"],
                            "confidence": 0.85
                        }
                        break

        # -------------------------
        # NATIONALITY EXTRACTION (STRONG)
        # -------------------------
        for i, box in enumerate(boxes):
            text = box["text"].strip().upper()

            if self._is_header_text(text):
                continue

            # MRZ (strongest)
            if re.fullmatch(r'[A-Z]{3}', text):
                entities["NATIONALITY"] = {
                    "text": text,
                    "bbox": box["bbox"],
                    "confidence": 0.95
                }
                continue

            # Label-based detection
            if any(keyword in text for keyword in ["NATIONALITY", "NATION", "SHTET"]):
                for candidate in boxes:
                    dy = candidate["bbox"][1] - box["bbox"][3]
                    dx = abs(candidate["bbox"][0] - box["bbox"][0])

                    if 0 <= dy <= 80 and dx <= 200:
                        val = candidate["text"].strip().upper()

                        if not self._is_header_text(val) and not re.search(r'\d', val):
                            entities["NATIONALITY"] = {
                                "text": val,
                                "bbox": candidate["bbox"],
                                "confidence": 0.9
                            }
                            break

        for box in boxes:
            if "," in box["text"] and not any(char.isdigit() for char in box["text"]):
                assign_if_valid("PLACE OF BIRTH", box)

        # 🚀 DERIVE NATIONALITY FROM PLACE OF BIRTH
        if "NATIONALITY" not in entities and "PLACE OF BIRTH" in entities:
            pob_text = entities["PLACE OF BIRTH"]["text"].upper()

            # Look for country codes
            match = re.search(r'\b(ALB|LVA|SVK)\b', pob_text)
            if match:
                country_code = match.group(1)

                nationality_map = {
                    "ALB": "ALBANIAN",
                    "LVA": "LVA",
                    "SVK": "SVK"
                }

                entities["NATIONALITY"] = {
                    "text": nationality_map.get(country_code, country_code),
                    "bbox": entities["PLACE OF BIRTH"]["bbox"],
                    "confidence": 0.85
                }

        # AUTHORITY (IMPROVED)
        for i, box in enumerate(boxes):
            if any(k in box["norm"] for k in ["AUTORIT", "AUTHORITY"]):
                for j, candidate in enumerate(boxes):
                    if j == i:
                        continue

                    dy = candidate["bbox"][1] - box["bbox"][3]
                    dx = abs(candidate["bbox"][0] - box["bbox"][0])
                    if 0 <= dy <= 60 and dx <= 150:
                        text = candidate["text"].strip()
                        if re.fullmatch(r'[A-Z]{2,5}', text):
                            entities["AUTHORITY"] = candidate
                            break
                if "AUTHORITY" in entities:
                    break

        if "AUTHORITY" not in entities:
            for box in boxes:
                text = box["text"].strip()
                if re.fullmatch(r'[A-Z]{2,5}', text) and text not in ["M", "F"]:
                    entities["AUTHORITY"] = box
                    break

        # -------------------------
        # FORCE NATIONALITY FROM COUNTRY
        # -------------------------
        country_map = {
            "LATVIA": "LVA",
            "ALBANIA": "ALBANIAN",
            "SLOVAKIA": "SVK"
        }

        if "NATIONALITY" not in entities or entities["NATIONALITY"]["text"] == "UNKNOWN":
            if country in country_map:
                entities["NATIONALITY"] = {
                    "text": country_map[country],
                    "bbox": (0, 0, 0, 0),
                    "confidence": 0.99
                }

        if not all(k in entities for k in ["DATE OF BIRTH", "DATE OF ISSUE", "DATE OF EXPIRY"]):
            # STEP 1: Collect ALL date candidates
            date_candidates = [
                box for box in boxes
                if re.search(r'\d{2}[./-]\d{2}[./-]\d{4}', box["text"])
            ]

            # STEP 2: Convert to datetime
            parsed_dates = []
            for box in date_candidates:
                try:
                    d = self._parse_date_from_text(box["text"])
                    if d:
                        parsed_dates.append((d, box))
                except Exception:
                    continue

            # STEP 3: SORT BY ACTUAL DATE VALUE
            parsed_dates = sorted(parsed_dates, key=lambda x: x[0])

            if len(parsed_dates) >= 3:
                entities["DATE OF BIRTH"] = parsed_dates[0][1]
                entities["DATE OF ISSUE"] = parsed_dates[1][1]
                entities["DATE OF EXPIRY"] = parsed_dates[2][1]

        anchors = []
        for i, box in enumerate(boxes):
            for pat, label in label_map.items():
                if re.search(pat, box["norm"]):
                    anchors.append((i, label))
                    break

        # Geometric Value Association: Link anchors to adjacent values using a weighted distance cost.
        for idx, label in anchors:
            if label in entities and entities[label].get("confidence", 0) >= 0.95:
                continue

            anchor = boxes[idx]
            ay, ax = y_mid(anchor["bbox"]), anchor["bbox"][2]

            best_idx = None
            best_score = -999

            for j, box in enumerate(boxes):
                if j == idx or j in used:
                    continue
                # Reject label-like text as value
                if is_label_like(box["norm"]):
                    continue
                if not is_valid_for_field(label, box["text"]):
                    continue
                vy = y_mid(box["bbox"])
                dx = box["bbox"][0] - ax

                # Strong horizontal preference
                if dx < 0 or dx > 400:
                    continue
                if abs(vy - ay) > 40:
                    continue

                score = self._score_candidate(label, box, anchor)
                if score > best_score:
                    best_score = score
                    best_idx = j

            if best_idx is not None:
                if label not in entities:
                    entities[label] = boxes[best_idx]
                    used.add(best_idx)

        self._validate_dates(entities)

        # 🚫 PREVENT GIVEN NAME = SURNAME
        if "GIVEN NAME" in entities and "SURNAME" in entities:
            if entities["GIVEN NAME"]["text"] == entities["SURNAME"]["text"]:
                # Keep SURNAME, remove GIVEN NAME
                del entities["GIVEN NAME"]

        # STRONG FALLBACK: infer GIVEN NAME from nearby SURNAME line
        if "SURNAME" in entities and "GIVEN NAME" not in entities:
            surname_box = entities["SURNAME"]["bbox"]
            sy = y_mid(surname_box)
            surname_text = entities["SURNAME"]["text"].strip().upper()

            best_candidate = None
            best_score = -999
            for candidate in boxes:
                ctext = candidate["text"].strip()
                ctext_upper = ctext.upper()
                if not ctext or ctext_upper == surname_text:
                    continue
                if is_label_like(candidate["norm"]):
                    continue
                if not is_valid_for_field("GIVEN NAME", ctext):
                    continue

                cy = y_mid(candidate["bbox"])
                dx = candidate["bbox"][0] - surname_box[2]
                dy = cy - sy

                if abs(dy) > 60:
                    continue
                if not (-120 <= dx <= 350):
                    continue

                score = self._score_candidate("GIVEN NAME", candidate, entities["SURNAME"])
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            if best_candidate:
                entities["GIVEN NAME"] = best_candidate

        # 🚀 FALLBACK GIVEN NAME FROM NEARBY TEXT
        if "SURNAME" in entities and "GIVEN NAME" not in entities:
            surname_box = entities["SURNAME"]["bbox"]

            for box in boxes:
                text = box["text"].strip()

                if text == entities["SURNAME"]["text"]:
                    continue

                # Must look like a name
                if not re.fullmatch(r'[A-Za-z ]{2,}', text):
                    continue

                # Check vertical alignment
                dy = abs((box["bbox"][1] + box["bbox"][3]) / 2 -
                         (surname_box[1] + surname_box[3]) / 2)

                if dy < 40:
                    entities["GIVEN NAME"] = box
                    break

        # STRONG FALLBACK: relaxed PERSONAL NO detection (alphanumeric, OCR-noise tolerant)
        if "PERSONAL NO" not in entities:
            best_candidate = None
            best_score = -999
            for candidate in boxes:
                if is_label_like(candidate["norm"]):
                    continue
                cleaned = re.sub(r'[^A-Z0-9]', '', candidate["text"].upper())
                if not (7 <= len(cleaned) <= 12):
                    continue
                if not re.fullmatch(r'[A-Z0-9]{7,12}', cleaned):
                    continue
                # avoid obvious date/height-like noise
                if re.fullmatch(r'\d{3}', cleaned):
                    continue
                if re.fullmatch(r'\d{8}', cleaned):
                    continue

                score = candidate.get("confidence", 0)
                if any(ch.isalpha() for ch in cleaned):
                    score += 0.5
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            if best_candidate:
                entities["PERSONAL NO"] = best_candidate

        # 🚫 FINAL CLEANUP PASS
        for field, data in list(entities.items()):
            if self._is_header_text(data["text"]):
                del entities[field]

        # 🚫 REMOVE SIGNATURE LEAKS
        for field, data in list(entities.items()):
            text = data["text"].upper()
            if "SIGNATURE" in text or "FIRMA" in text:
                del entities[field]

        # REMOVE HEADER LEAKS
        for field in ["SURNAME", "GIVEN NAME", "NATIONALITY"]:
            if field in entities:
                if self._is_header_text(entities[field]["text"]):
                    del entities[field]

        # CLEANUP WRONG ASSIGNMENTS
        for field, data in list(entities.items()):
            if not self._is_valid_for_field(field, data["text"]):
                del entities[field]

        # -------------------------
        # FORCE NATIONALITY FROM COUNTRY
        # -------------------------
        country_map = {
            "LATVIA": "LVA",
            "ALBANIA": "ALBANIAN",
            "SLOVAKIA": "SVK"
        }

        if "NATIONALITY" not in entities or entities["NATIONALITY"]["text"] == "UNKNOWN":
            if country in country_map:
                entities["NATIONALITY"] = {
                    "text": country_map[country],
                    "bbox": (0, 0, 0, 0),
                    "confidence": 0.99
                }

        # 🚨 FINAL FAILSAFE
        if "NATIONALITY" not in entities or not entities["NATIONALITY"]["text"].strip():
            fallback_nationality = country if country and country != "UNKNOWN" else "UNSPECIFIED"
            entities["NATIONALITY"] = {
                "text": fallback_nationality,
                "bbox": (0, 0, 0, 0),
                "confidence": 0.5
            }

        if required_fields:
            allowed = set(required_fields) | set(optional_fields) | {"MRZ LINE 1", "MRZ LINE 2"}
            entities = {k: v for k, v in entities.items() if k in allowed}

        # Finalize structured Named Entity Recognition (NER) output
        self.ner_entities = {
            k: {
                "text": v["text"],
                "bbox": v["bbox"],
                "confidence": v["confidence"]
            }
            for k, v in entities.items()
        }

        ocr_text_lines = [box["text"] for box in boxes if box.get("text")]
        country = self.detect_country(self.ocr_full_text)

        self.ner_entities = calibrate_entities(
            self.ner_entities,
            country=country,
            raw_lines=ocr_text_lines
        )

        self.ner_entities = derive_nationality(self.ner_entities)
        if "NATIONALITY" not in self.ner_entities or not self.ner_entities["NATIONALITY"].get("text", "").strip() or self.ner_entities["NATIONALITY"].get("text", "").strip().upper() == "UNKNOWN":
            fallback_nationality = country if country and country != "UNKNOWN" else "UNSPECIFIED"
            self.ner_entities["NATIONALITY"] = {
                "text": fallback_nationality,
                "bbox": (0, 0, 0, 0),
                "confidence": 0.99
            }

        self.risk_score, self.risk_issues = compute_risk_score(
            self.ner_entities,
            country=country
        )
        self.ner_source = "REGEX"

        # Calculate Recall Metrics: Evaluate extraction completeness for forensic reporting.
        self._update_ner_metrics()

        if print_summary:
            self.print_ner_fields_summary()

    def parse_mrz_dates(self, mrz_lines):
        try:
            line2 = mrz_lines[1]

            dob_raw = line2[13:19]
            expiry_raw = line2[21:27]

            def format_date(d):
                return f"19{d[0:2]}-{d[2:4]}-{d[4:6]}"

            return {
                "DATE OF BIRTH": format_date(dob_raw),
                "DATE OF EXPIRY": format_date(expiry_raw)
            }
        except:
            return {}
        
    def mrz_parse_(self, mrz_lines):
        """
        Parse MRZ (TD3 format) and extract structured fields.
        Supports standard passport MRZ with 2 lines.
        """
        if len(mrz_lines) < 2:
            return {}

        line1 = mrz_lines[0]["text"].replace(" ", "")
        line2 = mrz_lines[1]["text"].replace(" ", "")

        result = {}

        try:
            # -------------------------
            # LINE 1 (NAMES)
            # -------------------------
            # Format: P<COUNTRYSURNAME<<GIVEN<NAMES
            parts = line1.split("<<")

            if len(parts) >= 2:
                surname = parts[0][5:].replace("<", "").strip()
                given = parts[1].replace("<", " ").strip()

                if surname:
                    result["SURNAME"] = surname
                if given:
                    result["GIVEN NAME"] = given

            # -------------------------
            # LINE 2 (DATA)
            # -------------------------
            # Passport number
            passport_no = line2[0:9].replace("<", "")
            if passport_no:
                result["PASSPORT NO"] = passport_no

            # Nationality
            nationality = line2[10:13]
            result["NATIONALITY"] = nationality

            # Date of birth
            dob = line2[13:19]
            result["DATE OF BIRTH"] = self._format_mrz_date(dob)

            # Sex
            sex = line2[20]
            if sex in ["M", "F"]:
                result["SEX"] = sex

            # Expiry date
            expiry = line2[21:27]
            result["DATE OF EXPIRY"] = self._format_mrz_date(expiry)

        except Exception as e:
            self.log.append(f"- MRZ parsing failed: {e}")

        return result

    def _format_mrz_date(self, date_str):
        """
        Convert YYMMDD → DD-MM-YYYY
        """
        try:
            if len(date_str) != 6:
                return date_str

            year = int(date_str[:2])
            month = date_str[2:4]
            day = date_str[4:6]

            # Handle century
            year += 1900 if year > 30 else 2000

            return f"{day}-{month}-{year}"
        except Exception:
            return date_str

    def _update_ner_metrics(self):
        """Recompute NER completeness metrics from current ner_entities."""
        core_fields = {
            'SURNAME',
            'GIVEN NAME',
            'DATE OF BIRTH',
            'DATE OF ISSUE',
            'DATE OF EXPIRY',
            'SEX',
            'DOCUMENT NO',
            'PERSONAL NO',
            'NATIONALITY',
            'PLACE OF BIRTH',
            'AUTHORITY',
        }

        optional_fields = {
            'HEIGHT',
            'MRZ LINE 1',
            'MRZ LINE 2',
        }

        detected = set(self.ner_entities.keys())
        normalized_detected = set(detected)
        if {'ID CARD NO', 'PASSPORT NO'} & detected:
            normalized_detected.add('DOCUMENT NO')

        detected_core = normalized_detected & core_fields
        missing_core = core_fields - detected_core
        detected_optional = sorted(optional_fields & normalized_detected)

        self.ner_metrics = {
            "detected_fields": sorted(detected),
            "detected_optional_fields": detected_optional,
            "missing_core_fields": sorted(missing_core),
            "detected_core_count": len(detected_core),
            "core_expected_count": len(core_fields),
            "ner_recall": len(detected_core) / len(core_fields),
        }
        self.missing_ner_fields = sorted(missing_core)

    def detect_country(self, ocr_text):
        """Infer document country hints from OCR text."""
        text = str(ocr_text or "").upper()
        if any(k in text for k in ["SHQIP", "REPUBLIKA E SHQIPERISE"]):
            return "ALBANIA"
        if any(k in text for k in ["LATVIJA", "LATVIA", "LVA"]):
            return "LATVIA"
        if any(k in text for k in ["SLOVENSK", "SLOVAKIA", "SVK"]):
            return "SLOVAKIA"
        return "UNKNOWN"

    def validate_fields(self, country, entities):
        """Run strict country-level field validation checks."""
        issues = []

        def get(field):
            return entities.get(field, {}).get("text", "").upper()

        sex_value = get("SEX")
        if sex_value and sex_value not in ["M", "F"]:
            issues.append("Invalid sex value")

        if country == "ALBANIA":
            if get("NATIONALITY") != "ALBANIAN":
                issues.append("Invalid nationality")
            if not re.fullmatch(r"[A-Z]\d{8}[A-Z]", get("PERSONAL NO")):
                issues.append("Invalid Personal No format")
            if not re.fullmatch(r"\d+", get("ID CARD NO")):
                issues.append("Invalid ID Card No format")

        elif country == "LATVIA":
            if get("TYPE") not in ["P", "J"]:
                issues.append("Invalid document type")
            if get("ISSUING STATE CODE") != "LVA":
                issues.append("Invalid issuing state")
            if get("NATIONALITY") != "LATVIJAS":
                issues.append("Invalid nationality")
            if not re.fullmatch(r"[A-Z]{2}\d{7}", get("PASSPORT NO")):
                issues.append("Invalid passport number")
            personal_no = get("PERSONAL NO")
            if personal_no and not re.fullmatch(r"\d{6}-\d{5}", personal_no):
                issues.append("Invalid personal number")

        elif country == "SLOVAKIA":
            if get("NATIONALITY") != "SVK":
                issues.append("Invalid nationality")
            if not re.fullmatch(r"[A-Z]{2}\d{6}", get("ID CARD NO")):
                issues.append("Invalid ID format")
            if not re.fullmatch(r"\d{6}/\d{4}", get("PERSONAL NO")):
                issues.append("Invalid personal number")

        return issues

    def match_field_by_label(self, field, boxes):
        """Find the best value candidate around a detected label anchor."""
        best_candidate = None
        best_score = -999

        for i, box in enumerate(boxes):
            for pattern in LABEL_PATTERNS.get(field, []):
                if re.search(pattern, box["norm"]):
                    for j, candidate in enumerate(boxes):
                        if j == i:
                            continue

                        dy = candidate["bbox"][1] - box["bbox"][3]
                        dx = abs(candidate["bbox"][0] - box["bbox"][0])

                        if 0 <= dy <= 150 and dx <= 250:
                            if not self._is_valid_for_field(field, candidate["text"]):
                                continue

                            score = self._score_candidate(field, candidate, box)

                            if score > best_score:
                                best_score = score
                                best_candidate = candidate

        return best_candidate

    def _is_mostly_alpha(self, text):
        return bool(re.fullmatch(r'[A-Z\s]+', str(text or "")))

    def _is_mostly_numeric(self, text):
        return bool(re.fullmatch(r'[0-9\-./]+', str(text or "")))

    def _is_alphanumeric_id(self, text):
        return bool(re.fullmatch(r'[A-Z0-9]{6,}', str(text or "")))

    def _score_candidate(self, field, candidate, anchor=None):
        score = 0
        text = candidate["text"]
        norm = candidate["norm"]
        bbox = candidate["bbox"]

        # -------------------------
        # 1. OCR CONFIDENCE
        # -------------------------
        score += candidate.get("confidence", 0) * 2

        # -------------------------
        # 2. DISTANCE TO LABEL (VERY IMPORTANT)
        # -------------------------
        if anchor:
            ax, ay = anchor["bbox"][0], anchor["bbox"][1]
            cx, cy = bbox[0], bbox[1]

            dx = abs(cx - ax)
            dy = abs(cy - ay)

            if dx < 200:
                score += 2
            if dy < 50:
                score += 3

        # -------------------------
        # 3. FIELD-SPECIFIC RULES
        # -------------------------

        # NAME
        if field in ["SURNAME", "GIVEN NAME"]:
            if len(text.split()) <= 2:
                score += 2
            if len(text) < 15:
                score += 2
            if "/" in text:
                score -= 3

        # DATE
        if "DATE" in field:
            if re.search(r'\d{2}[./-]\d{2}[./-]\d{4}', text):
                score += 3

        # ID NUMBER
        if field in ["ID CARD NO", "PASSPORT NO"]:
            digits = sum(c.isdigit() for c in text)
            if digits >= 5:
                score += 3

        # PERSONAL NO
        if field == "PERSONAL NO":
            if re.fullmatch(r'[A-Z]\d+[A-Z]', text):
                score += 4

        # NATIONALITY
        if field == "NATIONALITY":
            if "/" in text:
                score += 3

        # AUTHORITY
        if field == "AUTHORITY":
            if re.fullmatch(r'[A-Z]{2,5}', text):
                score += 3

        # -------------------------
        # 4. PENALTIES
        # -------------------------

        # Header penalty
        if any(h in norm for h in ["REPUBLIK", "ALBANIA", "SHQIP"]):
            score -= 10

        # Label penalty
        if any(re.search(p, norm) for patterns in LABEL_PATTERNS.values() for p in patterns):
            score -= 5

        return score

    def _is_valid_for_field(self, field, text):
        """
        Validate if a candidate text is suitable for a specific NER field.
        Prevents wrong assignments like:
        - 'mbiemri/surname' → ID CARD NO
        - 'Bulcar,ALB' → DATE OF BIRTH
        """
        if not text:
            return False

        raw_text = str(text).strip()
        text = raw_text.upper()

        if field in ["DATE OF BIRTH", "DATE OF ISSUE", "DATE OF EXPIRY"]:
            return bool(re.search(r'\b\d{2}[./-]\d{2}[./-]\d{4}\b', text))

        if field == "ID CARD NO":
            cleaned = re.sub(r'[^0-9]', '', text)
            compact = re.sub(r'[^A-Z0-9]', '', text)
            return bool(
                re.fullmatch(r'\d{6,10}', cleaned) or
                re.fullmatch(r'[A-Z]{2}\d{6}', compact)
            )

        if field == "PASSPORT NO":
            cleaned = re.sub(r'[^A-Z0-9]', '', text)
            return bool(re.fullmatch(r'[A-Z]{2}\d{7}', cleaned) or re.fullmatch(r'[A-Z0-9]{7,12}', cleaned))

        if field == "SEX":
            return text in ["M", "F"]

        if field == "SURNAME":
            cleaned = re.sub(r'[^A-Z ]', '', text)
            return len(cleaned.strip()) >= 2

        if field == "GIVEN NAME":
            cleaned = re.sub(r'[^A-Z ]', '', text)
            return len(cleaned.strip()) >= 2

        if field == "PERSONAL NO":
            cleaned = re.sub(r'[^A-Z0-9]', '', text)
            return len(cleaned) >= 6

        if field == "PLACE OF BIRTH":
            return not bool(re.search(r'\d', text))

        if field == "NATIONALITY":
            # Accept MRZ country codes (3 letters)
            if re.fullmatch(r'[A-Z]{3}', text):
                return True

            if text in {"ALBANIAN", "LATVIJAS"}:
                return True

            # Or full text nationality split markers
            if "/" in text:
                return True

            return False

        if field == "HEIGHT":
            return bool(re.fullmatch(r'\d{3}', text))

        if field == "AUTHORITY":
            return bool(re.fullmatch(r"[A-ZÀ-Ž' .\-]{2,80}", text))

        return True

    def _ocr_json_output_path(self, image_path):
        """Build OCR JSON path consistent with save_ocr_json output."""
        json_dir = os.path.join("final_results\\results", "OCR_JSON_results")
        image_name = os.path.basename(image_path)
        json_name = os.path.splitext(image_name)[0] + ".json"
        return os.path.join(json_dir, json_name)

    def _ner_json_output_path(self, image_path):
        """Build NER JSON path for LLM/regex extracted entities."""
        json_dir = os.path.join("final_results\\results", "NER_JSON_results")
        image_name = os.path.basename(image_path)
        json_name = os.path.splitext(image_name)[0] + ".json"
        return os.path.join(json_dir, json_name)

    def save_ner_json(self, image_path):
        """Save extracted NER entities to dedicated JSON output folder."""
        ner_json_path = self._ner_json_output_path(image_path)
        os.makedirs(os.path.dirname(ner_json_path), exist_ok=True)

        data = {
            "image_name": os.path.basename(image_path),
            "image_size": [self.width, self.height],
            "ner_source": "RULE_BASED_REGEX",
            "ner_entities": [
                {
                    "field": field,
                    "text": payload.get("text", ""),
                    "bbox": list(payload.get("bbox", [])) if payload.get("bbox") else None,
                    "confidence": float(payload.get("confidence", 0.0)),
                }
                for field, payload in sorted(self.ner_entities.items())
            ],
            "ner_metrics": getattr(self, "ner_metrics", {}),
        }

        with open(ner_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"[INFO] NER JSON saved → {ner_json_path}")
        self.log.append(f"- NER JSON saved: {ner_json_path}")
        return ner_json_path

    def print_ner_fields_summary(self):
        """Print current NER entity summary to terminal."""
        print("\n[NER FIELDS]")
        for k in sorted(self.ner_entities):
            print(f"  {k:18s}: {self.ner_entities[k]['text']}")
        print("\n[CALIBRATED NER FIELDS]")
        for k in sorted(self.ner_entities):
            print(f"  {k:18s}: {self.ner_entities[k]['text']}")
        print(f"\n[⚠️ RISK SCORE]: {int(self.risk_score)}")
        print(f"[⚠️ ISSUES]: {self.risk_issues}")
        if self.missing_ner_fields:
            print("  Missing fields:", ", ".join(self.missing_ner_fields))
            print(f"  NER Recall: {self.ner_metrics.get('ner_recall', 0.0):.2f}")

    def perform_ocr(self):
        """Initializes and executes the PaddleOCR engine to retrieve raw text and spatial data."""
        self.log.append("- OCR: Multi-pass PaddleOCR started.")

        if self.ocr_engine is None:
            if not PaddleOCR:
                self.ocr_full_text = "OCR NOT AVAILABLE"
                return
            self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)

        all_boxes = []
        variants = self._generate_ocr_variants()

        for idx, img in enumerate(variants):
            try:
                result = self.ocr_engine.ocr(img, cls=True)

                if not result or not result[0]:
                    continue

                for line in result[0]:
                    bbox = line[0]
                    text = line[1][0]
                    conf = line[1][1]

                    if conf < 0.4:
                        continue

                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]

                    x1, y1 = int(min(x_coords)), int(min(y_coords))
                    x2, y2 = int(max(x_coords)), int(max(y_coords))

                    all_boxes.append({
                        "text": text.strip(),
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2)
                    })

            except Exception as e:
                self.log.append(f"- OCR variant {idx} failed: {e}")

        self.ocr_boxes = self._merge_ocr_boxes(all_boxes)
        self.ocr_full_text = " ".join(
            box["text"] for box in self.ocr_boxes if box.get("text")
        )

    def _generate_ocr_variants(self):
        variants = []

        # 1. Original
        variants.append(self.original_image)

        # 2. Grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        variants.append(gray)

        # 3. CLAHE enhanced
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        variants.append(enhanced)

        # 4. Sharpened
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharp = cv2.filter2D(gray, -1, kernel)
        variants.append(sharp)

        # 5. Threshold (binary)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(thresh)

        return variants

    def _merge_ocr_boxes(self, boxes):
        merged = []

        for box in boxes:
            x1, y1, _, _ = box["bbox"]
            duplicate = False

            for m in merged:
                mx1, my1, _, _ = m["bbox"]

                # overlap check
                if abs(x1 - mx1) < 20 and abs(y1 - my1) < 20:
                    # keep higher confidence
                    if box["confidence"] > m["confidence"]:
                        m.update(box)
                    duplicate = True
                    break

            if not duplicate:
                merged.append(box)

        return merged

    def _parse_date_from_text(self, text):
        """Parse a date value from OCR text using common ID formats."""
        if not text:
            return None

        raw = str(text).upper()
        # Try extracting a date-shaped substring first to survive OCR label noise.
        patterns = [
            r"\b\d{1,2}[\-/\.]\d{1,2}[\-/\.]\d{2,4}\b",
            r"\b\d{4}[\-/\.]\d{1,2}[\-/\.]\d{1,2}\b",
            r"\b\d{1,2}\s+[A-Z]{3,9}\s+\d{2,4}\b",
        ]

        candidates = []
        for pat in patterns:
            m = re.search(pat, raw)
            if m:
                candidates.append(m.group(0))

        cleaned = re.sub(r"[^0-9A-Z\-/\. ]", " ", raw).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        candidates.extend([cleaned, cleaned.replace('.', '-'), cleaned.replace('/', '-')])

        formats = (
            "%d-%m-%Y", "%d-%m-%y", "%Y-%m-%d",
            "%d/%m/%Y", "%d/%m/%y", "%Y/%m/%d",
            "%d.%m.%Y", "%d.%m.%y", "%Y.%m.%d",
            "%d %m %Y", "%d %m %y", "%Y %m %d",
            "%d %b %Y", "%d %B %Y", "%d %b %y", "%d %B %y",
        )

        for cand in candidates:
            cand = cand.strip()
            for fmt in formats:
                try:
                    return datetime.datetime.strptime(cand, fmt).date()
                except ValueError:
                    continue
        return None

    def evaluate_logical_consistency(self):
        """Check date ordering rules and store critical logical issues."""
        self.forgery_issues = []
        dob_raw = self.ner_entities.get("DATE OF BIRTH", {}).get("text", "")
        doi_raw = self.ner_entities.get("DATE OF ISSUE", {}).get("text", "")
        doe_raw = self.ner_entities.get("DATE OF EXPIRY", {}).get("text", "")

        dob = self._parse_date_from_text(dob_raw)
        doi = self._parse_date_from_text(doi_raw)
        doe = self._parse_date_from_text(doe_raw)

        if dob and doi and dob >= doi:
            self.forgery_issues.append(
                f"DATE OF BIRTH ({dob_raw}) is not earlier than DATE OF ISSUE ({doi_raw})."
            )
        if doi and doe and doi >= doe:
            self.forgery_issues.append(
                f"DATE OF ISSUE ({doi_raw}) is not earlier than DATE OF EXPIRY ({doe_raw})."
            )

        self.log.append(f"- Logical date checks: {len(self.forgery_issues)} issue(s).")
        return self.forgery_issues

    def calculate_risk_score(self):
        """Aggregate anomaly severity into a normalized 0..100 forensic risk score."""
        char_risk = sum(float(a.get("max_score", 0.0)) for a in self.anomalies)
        bg_risk = sum(float(a.get("z_score", 0.0)) * 2.0 for a in self.background_anomalies)
        ocr_risk = sum(float(a.get("max_score", 0.0)) * 2.5 for a in self.ocr_box_anomalies)
        cluster_risk = sum(float(r.get("severity_proxy", 0.0)) * 1.5 for r in self.suspicious_regions)
        logic_risk = 20.0 * len(self.forgery_issues)

        raw_risk = char_risk + bg_risk + ocr_risk + cluster_risk + logic_risk
        self.risk_score = float(min(100.0, raw_risk))
        self.log.append(f"- Composite risk score: {self.risk_score:.2f}/100")
        return self.risk_score
    
    def generate_training_features(self):
        """
        Derives a forensic feature vector for machine learning integration. 
        """
        self.log.append("- Deriving Comprehensive Feature Vector for ML.")
        
        # Baseline Character Statistics (Global means and STDs)
        stats = self.baseline_stats if self.baseline_stats else {}
        total_chars = len(self.characters)
        font_size_variance = float(np.var([c['height'] for c in self.characters])) if self.characters else 0.0
        ocr_confidence_mean = float(np.mean([b.get('confidence', 0.0) for b in self.ocr_boxes])) if self.ocr_boxes else 0.0

        field_blur_values = []
        for entity in self.ner_entities.values():
            bbox = entity.get('bbox')
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.width, x2), min(self.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = self.gray_original[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            field_blur_values.append(float(cv2.Laplacian(roi, cv2.CV_64F).var()))

        field_blur_variance = float(np.var(field_blur_values)) if field_blur_values else 0.0

        geo_anomalies = sum(
            1 for a in self.anomalies
            if any(t.startswith("GLOBAL") for t in a['types'])
        )

        ink_anomalies = sum(
            1 for a in self.anomalies
            if any(t in ('INK_INTENSITY_ANOMALY', 'EDGE_GRADIENT_ANOMALY') for t in a['types'])
        )


        # Feature Vector Assembly 
        self.forgery_features = {
            'Char_Count': total_chars,
            'NER_Detected_Count': self.ner_metrics.get('detected_count', 0)
                if hasattr(self, 'ner_metrics') else 0,
            'NER_Completeness_Ratio': self.ner_metrics.get('ner_recall', 0.0)
                if hasattr(self, 'ner_metrics') else 0.0,
            'H_Mean': stats.get('height_mean', 0), 'H_STD': stats.get('height_std', 0),
            'W_Mean': stats.get('width_mean', 0), 'W_STD': stats.get('width_std', 0),
            'AR_Mean': stats.get('aspect_ratio_mean', 0), 'AR_STD': stats.get('aspect_ratio_std', 0),
            'Ink_Mean': stats.get('ink_mean', 0), 'Ink_STD': stats.get('ink_std', 0),
            'Grad_Mean': stats.get('grad_mean', 0), 'Grad_STD': stats.get('grad_std', 0),
            'Ink_Density_Mean': stats.get('density_mean', 0),
            'Geo_Anomaly_Ratio': geo_anomalies / (total_chars + 1e-6),
            'Ink_Anomaly_Ratio': ink_anomalies / (total_chars + 1e-6),
            'BG_Mean': self.background_stats.get('mean', 0) if self.background_stats else 0,
            'BG_STD': self.background_stats.get('std', 0) if self.background_stats else 0,
            # Legacy count features kept for backward compatibility with existing datasets/models
            'OCR_Box_Anomalies_Count': len(self.ocr_box_anomalies),
            'BG_Anomaly_Line_Count': len(self.background_anomalies),
            'Clustered_Regions_Count': len(self.suspicious_regions),
            # New engineered features
            'Font_Size_Variance': font_size_variance,
            'OCR_Confidence_Mean': ocr_confidence_mean,
            'Field_Blur_Variance': field_blur_variance,
            'Risk_Score': self.calculate_risk_score(),
        }
        
        self.log.append(f"- Feature vector generated with {len(self.forgery_features)} metrics.")
        return self.forgery_features
    

    def process_document(self, char_sensitivity=2.0, bg_sensitivity=3.0, ocr_sensitivity=2.5, auto_save_png=True):
        """Orchestrates the full detection pipeline with mandatory sequence."""
        try:
            # 1. Start with OCR and Image Preprocessing
            self.perform_ocr()
            self.preprocess_image() 
            
            # 2. Identify Fields and Values using rule-based regex NER
            self.identify_critical_entities_from_ocr(print_summary=False)
            country = self.detect_country(self.ocr_full_text)
            validation_issues = self.validate_fields(country, self.ner_entities)
            if validation_issues:
                self.forgery_issues.extend(validation_issues)
            self.save_ocr_json(self.image_path)

            # 3. Run physical and OCR box checks
            self.detect_ocr_box_anomalies(sensitivity=ocr_sensitivity)
            self.detect_text_regions()
            self.segment_characters()
            self.calculate_background_stats()
            self.detect_background_anomalies(sensitivity=bg_sensitivity)
            self.calculate_baseline_statistics()
            self.evaluate_logical_consistency()
            
            # 4. Detect anomalies and cluster
            self.detect_anomalies(sensitivity=char_sensitivity)
            self.cluster_anomalous_regions()
            
            # 5. Persist NER outputs/features
            self.save_ner_json(self.image_path)
            self.print_ner_fields_summary()
            self.generate_training_features()
            self.generate_report() # This pre-calculates the verdict

            # 6. Save PNG visualization for every processed file
            if auto_save_png:
                self.visualize_results(save_path=default_png_output_path(self.image_path))

        except Exception as e:
            self.log.append(f"[FATAL PIPELINE ERROR] {e}")
            raise
    def visualize_results(self, save_path=None):
        """
        Visualization logic is synchronized with:
        - ML verdict during inference
        - Ground truth during training dataset generation
        """

        # Case 1: ML-based inference
        if hasattr(self, "ml_verdict"):
            verdict = self.ml_verdict
            show_anomalies = verdict == "FORGED"
        # Case 2: Training dataset visualization
        elif hasattr(self, "is_training_doc"):
            verdict = "FORGED" if self.is_forged_gt else "AUTHENTIC"
            show_anomalies = self.is_forged_gt
        else:
            verdict = "AUTHENTIC"
            show_anomalies = False

        prediction_label = self.final_verdict if self.final_verdict else verdict
        is_forged = verdict == "FORGED"
        vis = self.display_image.copy()

        # 1. OCR boxes (Yellow)
        for box in self.ocr_boxes:
            x1, y1, x2, y2 = box['bbox']
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # 2. NER Fields (Green)
        for field, data in self.ner_entities.items():
            bbox = data.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 3. High-severity anomalies (Red) — ONLY WHEN FORGED
        if show_anomalies:
            for a in self.anomalies:
                if a.get("severity") == "high" and "char" in a:
                    x1, y1, x2, y2 = a['char']['bbox']
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)

        label_text = f"GROUND TRUTH: {self.ground_truth_label}"
        color = (0, 255, 0) if self.ground_truth_label == "GENUINE" else (0, 0, 255)
        cv2.putText(
            vis,
            label_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
            cv2.LINE_AA
        )

        pred_text = f"PREDICTION: {prediction_label}"
        cv2.putText(
            vis,
            pred_text,
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 0),
            3,
            cv2.LINE_AA
        )

        if self.ground_truth_label != prediction_label:
            cv2.putText(
                vis,
                "MISCLASSIFIED",
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
                cv2.LINE_AA
            )

        orig_rgb = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
        vis_rgb  = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        fig = plt.figure(figsize=(18, 7))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(orig_rgb)
        ax1.set_title("Original Document", fontsize=14, weight="bold")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(vis_rgb)
        ax2.set_title(
            f"Forgery Detection Verdict: {verdict}",
            fontsize=14,
            color="red" if is_forged else "green",
            weight="bold"
        )
        ax2.axis("off")

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")

        plt.close(fig)
        
    def generate_report(self):
        """Generates a single, clean, comprehensive report matching the target format."""
        report = []
        report.append("\n================================================================================")
        report.append("DOCUMENT FORGERY AUTHENTICATION REPORT (AI-Enhanced)")
        report.append("================================================================================")
        report.append(f"File    : {os.path.basename(self.image_path)}")
        report.append(f"Resolution: {self.width} x {self.height}")
        report.append(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 
        report.append("-" * 80)
        report.append("Execution Log:"); report.extend([f"  {line}" for line in self.log]); report.append("-" * 80)

        # Named Entities (Omitted for brevity, assumed included)
        report.append("Named Entities (scoped fields):")
        ner_keys_order = ['SURNAME', 'GIVEN NAME', 'FULL NAME', 'NATIONALITY', 'PASSPORT NO', 'ID CARD NO', 'PERSONAL NO', 'PLACE OF BIRTH', 'DATE OF BIRTH', 'SEX', 'HEIGHT', 'DATE OF ISSUE', 'DATE OF EXPIRY', 'AUTHORITY', 'SIGNATURE', 'MRZ LINE 1', 'MRZ LINE 2']
        detected_keys = [k for k in ner_keys_order if k in self.ner_entities]
        for entity in detected_keys:
            data = self.ner_entities[entity]
            bbox = data["bbox"]
            entity_value = data["text"]
            report.append(f" {entity.ljust(17)} : {entity_value}")
        report.append("-" * 80)
        
        # Logical Content Issues
        report.append("Logical Content Issues:")
        if self.forgery_issues:
            for issue in self.forgery_issues: report.append(f"  - {issue}")
        else: report.append("  - None detected.")
        report.append("-" * 80)
        
        # Feature Vector (The Training Feature) 
        report.append("ML Training Feature Vector (All Metrics):")
        report.append(f"Total Metrics Collected: {len(self.forgery_features)}")
        for name, value in self.forgery_features.items():
             report.append(f"  - {name.ljust(25)} : {value:.4f}")
        report.append("-" * 80)

        # FINAL VERDICT
        risk_score = self.risk_score if self.risk_score else self.calculate_risk_score()
        
        if self.forgery_issues:
            verdict = "**CRITICAL LOGICAL FORGERY**"; confidence = 99.9
            reason = "Critical logical content inconsistencies detected by OCR analysis."
        elif risk_score > 50:
            verdict = "SUSPICIOUS (ML Ready)"; confidence = min(99.0, 50.0 + risk_score * 0.5)
            reason = f"High cumulative risk score ({risk_score:.2f}/100) indicating likely physical anomalies. Features ready for full ML classification."
        else:
            verdict = "FEATURES COLLECTED (ML Ready)"; confidence = 90.0 - (risk_score * 0.2)
            reason = "Feature vector collected. Minimal physical anomalies detected. Document is ready for final classification by the trained model."
            
        self.final_verdict = verdict.strip('*').split('(')[0].strip()

        report.append("FINAL ASSESSMENT (ML Classification Readiness)")
        report.append("=" * 45)
        report.append(f"  VERDICT              : {verdict}")
        report.append(f"  CONFIDENCE (Heuristic): {confidence:.1f}%")
        report.append(f"  PRIMARY REASON      : {reason}")
        report.append("=" * 45)

        final_report = "\n".join(report)
        return final_report

    def save_ocr_json(self, image_path):
        """
        Save full PaddleOCR output as JSON (one per document)
        """

        # Folder creation
        json_dir = os.path.join("final_results\\results", "OCR_JSON_results")
        os.makedirs(json_dir, exist_ok=True)

        image_name = os.path.basename(image_path)
        json_name = os.path.splitext(image_name)[0] + ".json"
        json_path = os.path.join(json_dir, json_name)

        data = {
            "image_name": image_name,
            "image_size": [self.width, self.height],
            "ocr_engine": "PaddleOCR",
            "ocr_results": []
        }

        for box in self.ocr_boxes:
            data["ocr_results"].append({
                "text": box.get("text", ""),
                "bbox": list(box.get("bbox", [])),
                "confidence": float(box.get("confidence", 0.0))
            })

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"\n\n[INFO] OCR JSON saved → {json_path}")


def ensure_output_folder():
    """Creates the output folder if it doesn't exist."""
    folder = "final_results/PNG_results"
    os.makedirs(folder, exist_ok=True)
    return folder

def default_png_output_path(image_path):
    """Build default PNG output path for a processed document."""
    out_folder = ensure_output_folder()
    doc_name = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(out_folder, f"{doc_name}_analysis.png")

def analyze_single_document(document_path, char_sensitivity=2.0, bg_sensitivity=3.0, ocr_sensitivity=2.5):
    """
    Main function to orchestrate the analysis for a single document.
    """
    output_png = default_png_output_path(document_path)

    print("\n================================================================================")
    print("      STARTING DOCUMENT FORGERY ANALYSIS (AI-ENHANCED)")
    print("================================================================================")

    try:
        detector = DocumentForgeryDetector(document_path) 
    except Exception as e:
        print(f"\nFATAL ERROR loading document: {e}"); return None

    detector.process_document(
        char_sensitivity=char_sensitivity,
        bg_sensitivity=bg_sensitivity,
        ocr_sensitivity=ocr_sensitivity,
        auto_save_png=True,
    )

    print(detector.generate_report())

    print(f"[INFO] Saved visualization to: {output_png}")
    print("================================================================================")
    print("                Analysis Complete!")
    print("================================================================================")

    return detector

def extract_country_code(document_name):
    """Derive a country code prefix from filename (e.g., alb_id_00.jpg -> ALB)."""
    base = os.path.basename(document_name)
    code = base.split("_")[0].strip().upper()
    if not code:
        return "UNK"
    return "".join(ch for ch in code if ch.isalnum())[:3] or "UNK"
