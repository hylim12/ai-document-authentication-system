"""
Project Title: AI-POWERED DOCUMENT AUTHENTICATION FOR ANTI-MONEY LAUNDERING (AML) SYSTEMS
Created By: Eldeena Lim Huey Yinn
Student ID: 1211111904

File: forged_document_detector.py
Functionality: Image preprocessing, character segmentation, and statistical anomaly detection.
"""

# Import necessary libraries and modules
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
import os
from PIL import Image
import re 
import datetime 
import csv
import glob
import shutil
import json
import unicodedata
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None 
warnings.filterwarnings('ignore')


class DocumentForgeryDetector:

    def __init__(self, image_path, ocr_engine=None):
        """Initializes forensic storage and standardizes input resolution."""
        self.image_path = image_path
        self.ocr_engine = ocr_engine
        self.log = []

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        try:
            # Standardize input for consistent feature extraction
            pil_image = Image.open(image_path).convert("RGB")
            target_width = 1500
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


    def identify_critical_entities_from_ocr(self):
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

            boxes.append({
                "text": box.get("text", "").strip(),
                "norm": normalize(box.get("text", "")),
                "bbox": (x1, y1, x2, y2),
                "confidence": box.get("confidence", 0.0)
            })

        self.ocr_boxes = boxes
        if not boxes:
            self.ner_entities = {}
            self.ner_metrics = {}
            self.missing_ner_fields = []
            return

        def y_mid(b): return (b[1] + b[3]) // 2

        def is_label_like(text):
            # Determines if a string matches known document field labels.
            return any(re.search(p, text) for p in (
                r'MBIEMR', r'SURNAME',
                r'EMR', r'GIVEN',
                r'SHTET', r'NATION',
                r'VEND', r'PLACE',
                r'DAT', r'DATE',
                r'NR', r'CARD',
                r'GJIN', r'SEX',
                r'PERSONAL',
                r'AUTOR', r'AUTHOR',
                r'FIRM', r'SIGN'
            ))

        # Direct Entity Extraction: Identify fields with distinct, globally unique patterns
        entities = {}
        used = set()
        strong_patterns = {
            "PERSONAL NO": r'[A-Z]\d{7,9}[A-Z]',
            "ID CARD NO": r'\d{7,10}',
            "SEX": r'\b[MF]\b'
        }

        for i, box in enumerate(boxes):
            for label, pat in strong_patterns.items():
                if label in entities:
                    continue
                if re.fullmatch(pat, box["norm"]):
                    entities[label] = box
                    used.add(i)

        # Label Anchor Detection: Locate specific headers to act as geometric reference points.
        label_map = {
            r'MBIEMR|SURNAME': 'SURNAME',
            r'EMR|GIVEN': 'GIVEN NAME',
            r'SHTET|NATION': 'NATIONALITY',
            r'VEND|PLACE': 'PLACE OF BIRTH',
            r'LINDJ|BIRTH': 'DATE OF BIRTH',
            r'LSHIM|ISSUE': 'DATE OF ISSUE',
            r'SKADIM|EXPIR': 'DATE OF EXPIRY',
            r'GJIN|SEX': 'SEX',
            r'LET|CARD': 'ID CARD NO',
            r'PERSONAL': 'PERSONAL NO',
            r'AUTOR|AUTHOR': 'AUTHORITY',
            r'FIRM|SIGN': 'SIGNATURE'
        }

        anchors = []
        for i, box in enumerate(boxes):
            for pat, label in label_map.items():
                if re.search(pat, box["norm"]):
                    anchors.append((i, label))
                    break

        # Geometric Value Association: Link anchors to adjacent values using a weighted distance cost.
        for idx, label in anchors:
            if label in entities:
                continue

            anchor = boxes[idx]
            ay, ax = y_mid(anchor["bbox"]), anchor["bbox"][2]

            best_idx = None
            best_score = float("inf")

            for j, box in enumerate(boxes):
                if j == idx or j in used:
                    continue
                if is_label_like(box["norm"]):
                    continue
                vy = y_mid(box["bbox"])
                dx = box["bbox"][0] - ax
                dy = box["bbox"][1] - anchor["bbox"][3]
                if not (
                    (0 < dx < 450 and abs(vy - ay) <= 45) or
                    (0 <= dy <= 80)
                ):
                    continue
                score = abs(vy - ay) * 80 + dx
                if score < best_score:
                    best_score = score
                    best_idx = j

            if best_idx is not None:
                entities[label] = boxes[best_idx]
                used.add(best_idx)

        # Finalize structured Named Entity Recognition (NER) output
        self.ner_entities = {
            k: {
                "text": v["text"],
                "bbox": v["bbox"],
                "confidence": v["confidence"]
            }
            for k, v in entities.items()
        }

        # Calculate Recall Metrics: Evaluate extraction completeness for forensic reporting.
        EXPECTED_FIELDS = {
            'SURNAME',
            'GIVEN NAME',
            'DATE OF BIRTH',
            'DATE OF ISSUE',
            'DATE OF EXPIRY',
            'SEX',
            'ID CARD NO',
            'PERSONAL NO',
            'NATIONALITY',
            'PLACE OF BIRTH',
            'AUTHORITY'
        }

        detected = set(self.ner_entities.keys())
        self.ner_metrics = {
            "detected_fields": sorted(detected),
            "missing_fields": sorted(EXPECTED_FIELDS - detected),
            "detected_count": len(detected),
            "expected_count": len(EXPECTED_FIELDS),
            "ner_recall": len(detected) / len(EXPECTED_FIELDS)
        }
        self.missing_ner_fields = self.ner_metrics["missing_fields"]

        # Log findings
        print("\n[NER FIELDS]")
        for k in sorted(self.ner_entities):
            print(f"  {k:18s}: {self.ner_entities[k]['text']}")
        if self.missing_ner_fields:
            print("  Missing fields:", ", ".join(self.missing_ner_fields))
        print(f"  NER Recall: {self.ner_metrics['ner_recall']:.2f}")


    def perform_ocr(self):
        """Initializes and executes the PaddleOCR engine to retrieve raw text and spatial data."""
        self.log.append("- OCR: Starting PaddleOCR.")
        
        # Use the injected engine if available, otherwise fallback to creating one
        if self.ocr_engine is None:
            if not PaddleOCR:
                self.ocr_full_text = "OCR NOT AVAILABLE"
                self.log.append("- OCR failed: Library not found.")
                return
            self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
        try:
            result = self.ocr_engine.ocr(np.array(self.original_image), cls=True)
            self.ocr_boxes = []

            if not result or not result[0]:
                print("[DEBUG] PaddleOCR returned NOTHING.")
                return
            
            for line in result[0]:
                bbox = line[0]
                text = line[1][0]
                conf = line[1][1]

                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]

                x1, y1 = int(min(x_coords)), int(min(y_coords))
                x2, y2 = int(max(x_coords)), int(max(y_coords))

                self.ocr_boxes.append({
                    "text": text.strip(),
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2)
                })

        except Exception as e:
            self.log.append(f"- PaddleOCR Error: {e}")
            self.ocr_full_text = "OCR ERROR"
        
    
    def generate_training_features(self):
        """
        Derives a forensic feature vector for machine learning integration. 
        """
        self.log.append("- Deriving Comprehensive Feature Vector for ML.")
        
        # Baseline Character Statistics (Global means and STDs)
        stats = self.baseline_stats if self.baseline_stats else {}
        total_chars = len(self.characters)
        
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
            'NER_Detected_Count': self.ner_completeness['detected_count']
                if hasattr(self, 'ner_completeness') else 0,
            'NER_Completeness_Ratio': self.ner_completeness['completeness_ratio']
                if hasattr(self, 'ner_completeness') else 0.0,
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
            'OCR_Box_Anomalies_Count': len(self.ocr_box_anomalies),
            'BG_Anomaly_Line_Count': len(self.background_anomalies),
            'Clustered_Regions_Count': len(self.suspicious_regions),
        }
        
        self.log.append(f"- Feature vector generated with {len(self.forgery_features)} metrics.")
        return self.forgery_features
    

    def process_document(self, char_sensitivity=2.0, bg_sensitivity=3.0, ocr_sensitivity=2.5, auto_save_png=True):
        """Orchestrates the full detection pipeline with mandatory sequence."""
        try:
            # 1. Start with OCR and Image Preprocessing
            self.perform_ocr()
            self.preprocess_image() 
            
            # 2. Identify Fields and Values
            self.identify_critical_entities_from_ocr() 
            self.save_ocr_json(self.image_path) 

            # 3. Run physical and OCR box checks
            self.detect_ocr_box_anomalies(sensitivity=ocr_sensitivity)
            self.detect_text_regions()
            self.segment_characters()
            self.calculate_background_stats()
            self.detect_background_anomalies(sensitivity=bg_sensitivity)
            self.calculate_baseline_statistics()
            
            # 4. Detect anomalies and cluster
            self.detect_anomalies(sensitivity=char_sensitivity)
            self.cluster_anomalous_regions()
            
            # 5. Generate final ML features and final report strings
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
        ner_keys_order = ['SURNAME', 'GIVEN NAME', 'NATIONALITY', 'ID CARD NO', 'PLACE OF BIRTH', 'DATE OF BIRTH', 'GENDER', 'DATE OF ISSUE', 'DATE OF EXPIRY', 'SIGNATURE', 'PERSONAL NO', 'AUTHORITY']
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
        total_suspicion_score = (len(self.anomalies) * 1) + (len(self.background_anomalies) * 5) + (len(self.ocr_box_anomalies) * 10) + (len(self.suspicious_regions) * 15)
        
        if self.forgery_issues:
            verdict = "**CRITICAL LOGICAL FORGERY**"; confidence = 99.9
            reason = "Critical logical content inconsistencies detected by OCR analysis."
        elif total_suspicion_score > 50: 
            verdict = "SUSPICIOUS (ML Ready)"; confidence = min(99.0, 50.0 + total_suspicion_score * 0.5)
            reason = f"High cumulative heuristic score ({total_suspicion_score}) indicating many physical anomalies. Features ready for full ML classification."
        else:
            verdict = "FEATURES COLLECTED (ML Ready)"; confidence = 90.0 - (total_suspicion_score * 0.2)
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
        json_dir = os.path.join("results", "OCR_JSON_results")
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

        print(f"[INFO] OCR JSON saved → {json_path}")


def ensure_output_folder():
    """Creates the output folder if it doesn't exist."""
    folder = "PNG_results"
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


def generate_datasets(input_dirs=None):
    """Generate training/test CSV from training folders only."""
    out_folder = ensure_output_folder()
    
    if input_dirs is None:
        input_dirs = ["input_docs"]
    elif isinstance(input_dirs, str):
        input_dirs = [input_dirs]

    shared_engine = None
    if PaddleOCR is not None:
        print("\n[INFO] Initializing shared PaddleOCR Engine...")
        shared_engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
    else:
        print("\n[WARNING] PaddleOCR not installed. Running feature extraction without OCR semantics.")

    image_paths = []
    for input_dir in input_dirs:
        image_paths.extend(glob.glob(os.path.join(input_dir, '*.jpg')))
        image_paths.extend(glob.glob(os.path.join(input_dir, '*.png')))
        image_paths.extend(glob.glob(os.path.join(input_dir, '*.jpeg')))

    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"[FATAL ERROR] No images found in: {input_dirs}")
        return

    training_data = [] 
    test_data = []     
    
    # Summary Table Counters
    summary = {
        'total': len(image_paths),
        'success': 0,
        'failed': 0,
        'actual_genuine': 0,
        'actual_forged': 0,
        'detected_as_forged': 0,
        'country_counts': defaultdict(int)
    }

    print(f"=========================================================")
    print(f"PROCESSING {len(image_paths)} DOCUMENTS FOR DATASETS")
    print(f"=========================================================")

    for i, path in enumerate(image_paths):
        doc_name = os.path.basename(path) 
        doc_root = os.path.splitext(doc_name)[0]
        is_forged = 'fake' in doc_name.lower()
        label = 1 if is_forged else 0
        country_code = extract_country_code(doc_name)
        summary['country_counts'][country_code] += 1

        if is_forged:
            summary['actual_forged'] += 1
        else:
            summary['actual_genuine'] += 1

        try:
            detector = DocumentForgeryDetector(path, ocr_engine=shared_engine)
            detector.is_training_doc = True
            detector.is_forged_gt = is_forged

            detector.process_document(auto_save_png=True)
            print(f"   [OK] Saved outputs: PNG_results/{doc_root}_analysis.png and results/OCR_JSON_results/{doc_root}.json")

            if "FORGED" in detector.final_verdict.upper():
                summary['detected_as_forged'] += 1

            # Prepare features
            data_row = {'Document_ID': doc_name, 'Label': label, 'Country_Code': country_code}
            data_row.update(detector.forgery_features)
            
            training_data.append(data_row)
            test_data.append(data_row)
            
            summary['success'] += 1

        except Exception as e:
            print(f"   FAILED to process {doc_name}. Error: {e}")
            summary['failed'] += 1
            continue

    # Write CSVs
    if training_data:
        write_csv(training_data, "ml_training_data.csv")
    if test_data:
        write_csv(test_data, "ml_test_data.csv")

    # Print Summary Table
    print("\n" + "="*50)
    print("         PROCESSING SUMMARY REPORT")
    print("="*50)
    print(f" Total Documents Found      : {summary['total']}")
    print(f" Successfully Processed     : {summary['success']}")
    print(f" Processing Failures        : {summary['failed']}")
    print("-" * 50)
    print(f" Ground Truth (Genuine)     : {summary['actual_genuine']}")
    print(f" Ground Truth (Forged)      : {summary['actual_forged']}")
    print(f" AI Flagged as Suspicious   : {summary['detected_as_forged']}")
    print(f" Countries Included         : {dict(summary['country_counts'])}")
    print("-" * 50)
    print(f" Files Saved to             : {out_folder}/")
    print(f" CSVs Generated             : ml_training_data.csv, ml_test_data.csv")
    print("="*50 + "\n")


def write_csv(data_list, filename):
    """Helper to write dictionary lists to CSV with consistent headers."""
    if not data_list: return
    fieldnames = ['Document_ID', 'Label'] + list(data_list[0].keys())
    fieldnames = list(dict.fromkeys(fieldnames)) # Remove duplicates
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_list)

def cleanup_results_folder(folder_path="PNG_results"):
    """Deletes all existing files in the results folder before a new batch run."""
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path): os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e: print(f'[ERROR] Cleanup failed: {e}')
    else:
        os.makedirs(folder_path)

if __name__ == "__main__":

    cleanup_results_folder("PNG_results")
    cleanup_results_folder("results")
    # Generate ml_training_data.csv
    generate_datasets(["input_docs"])
    # Run this to generate a single document analysis
