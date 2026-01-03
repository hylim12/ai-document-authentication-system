import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
import os
from PIL import Image
import pytesseract 
import re 
import datetime 
import csv
import glob
import shutil
try:
    from paddleocr import PaddleOCR
except ImportError:
    # Set to None if paddleocr is not installed
    PaddleOCR = None 

warnings.filterwarnings('ignore')


class DocumentForgeryDetector:

    def __init__(self, image_path, ocr_engine=None):
        self.image_path = image_path
        self.ocr_engine = ocr_engine  # Accept a pre-loaded engine to save time
        self.log = [] 
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # Load using PIL
            pil_image = Image.open(image_path)
            
            # --- PERFORMANCE OPTIMIZATION: IMAGE RESIZING ---
            # Standardizing to 1500px width significantly speeds up OCR
            target_width = 1500
            w_percent = (target_width / float(pil_image.size[0]))
            target_height = int((float(pil_image.size[1]) * float(w_percent)))
            
            # Use LANCZOS for high-quality downsampling to keep text clear
            pil_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            image_array = np.array(pil_image)

            if len(image_array.shape) == 2:
                self.gray = image_array
                self.original_image = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
            elif image_array.shape[2] >= 3:
                rgb_array = image_array[:, :, :3]
                self.original_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                self.gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError(f"Unsupported image format: {image_array.shape}")
                
        except Exception as e:
            # Fallback for corrupted PIL loads
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            self.gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        self.height, self.width = self.gray.shape
        self.log.append(f"- Loaded & Resized: {os.path.basename(image_path)} ({self.width}x{self.height})")
        
        # Initialize storage containers
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
        """Enhances contrast internally for OCR without modifying the visual background."""
        self.log.append("- Preprocess: generating internal enhanced copies for OCR logic.")
        
        # Safe local copy - ensures self.original_image stays as the clean input document
        temp_img = self.original_image.copy()
        
        # Internal enhancement for text clarity
        lab = cv2.cvtColor(temp_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Generate the grayscale version for character analysis
        gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY) 
        gray = cv2.fastNlMeansDenoising(gray, None, h=6)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel)
        
        # Update ONLY technical layers
        self.gray = gray
        
        # IMPORTANT: We no longer re-assign self.original_image here
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        self.binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_morph)
        
        self.log.append("- Preprocess: complete (clean original preserved).")
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
        """Uses Contour-based segmentation for individual characters."""
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
        sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        for char in self.characters:
            x1, y1, x2, y2 = char['bbox']
            char_roi_gray = self.gray[y1:y2, x1:x2]
            char_roi_binary = self.binary[y1:y2, x1:x2]
            ink_pixels = char_roi_gray[char_roi_binary > 0]
            char['mean_ink_intensity'] = np.mean(ink_pixels) if ink_pixels.size > 0 else 255
            grad_roi = gradient_magnitude[y1:y2, x1:x2]
            char['mean_gradient'] = np.mean(grad_roi[char_roi_binary > 0]) if grad_roi[char_roi_binary > 0].size > 0 else 0
            
        intensities = [c['mean_ink_intensity'] for c in self.characters]
        gradients = [c['mean_gradient'] for c in self.characters]
        densities = [c['density'] for c in self.characters]
        
        if self.baseline_stats is None: self.baseline_stats = {}
            
        self.baseline_stats.update({
            'ink_mean': np.mean(intensities), 'ink_std': np.std(intensities),
            'grad_mean': np.mean(gradients), 'grad_std': np.std(gradients),
            'density_mean': np.mean(densities), 'density_std': np.std(densities),
        })

    def calculate_background_stats(self):
        """Calculates global background intensity mean and standard deviation."""
        background_mask = cv2.bitwise_not(self.binary)
        background_pixels = self.gray[background_mask > 0]
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
            line_roi_gray = self.gray[y_start:y_end, :]
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
        FORENSIC CHECK: Compares ALL NER fields against 5 characteristics:
        1. Font Size, 2. Ink Density, 3. Style, 4. Color Intensity, 5. Alignment.
        """
        if not self.baseline_stats: 
            self.calculate_baseline_statistics()
        
        # --- THE OFFICIAL GLOBAL STANDARD PROFILE (For ALL NER Fields) ---
        # size: (min_h, max_h), bold: bool, italic: bool, ink: (min, max), align_tol: px
        STANDARD_BASELINE = {
            'SURNAME':         {'size': (14, 20), 'bold': True,  'italic': False, 'ink': (30, 70),  'align': 2},
            'GIVEN NAME':      {'size': (14, 20), 'bold': False, 'italic': False, 'ink': (40, 80),  'align': 2},
            'NATIONALITY':     {'size': (12, 16), 'bold': False, 'italic': False, 'ink': (40, 80),  'align': 1},
            'PLACE OF BIRTH':  {'size': (11, 15), 'bold': False, 'italic': False, 'ink': (40, 80),  'align': 1},
            'DATE OF BIRTH':   {'size': (12, 16), 'bold': False, 'italic': False, 'ink': (30, 60),  'align': 1},
            'GENDER':          {'size': (12, 16), 'bold': False, 'italic': False, 'ink': (40, 80),  'align': 1},
            'DATE OF ISSUE':   {'size': (12, 16), 'bold': False, 'italic': False, 'ink': (30, 60),  'align': 1},
            'DATE OF EXPIRY':  {'size': (12, 16), 'bold': False, 'italic': False, 'ink': (30, 60),  'align': 1},
            'ID CARD NO':      {'size': (13, 18), 'bold': True,  'italic': False, 'ink': (20, 50),  'align': 1},
            'PERSONAL NO':     {'size': (13, 18), 'bold': True,  'italic': False, 'ink': (20, 50),  'align': 1},
            'AUTHORITY':       {'size': (10, 14), 'bold': False, 'italic': True,  'ink': (50, 90),  'align': 2},
            'SIGNATURE':       {'size': (15, 40), 'bold': False, 'italic': True,  'ink': (20, 100), 'align': 10}
        }

        anomalies = []
        stats = self.baseline_stats
        
        # Pre-calculate line averages for Alignment check
        line_baselines = defaultdict(list)
        for c in self.characters:
            line_baselines[c['line_idx']].append(c['y'] + c['height'])
        line_avg_y = {idx: np.mean(vals) for idx, vals in line_baselines.items()}

        for char in self.characters:
            scores = []; types = []
            
            # Find associated NER field
            cx, cy = char['x'] + char['width'] / 2, char['y'] + char['height'] / 2
            current_field = None
            for field, bbox in self.ner_entities.items():
                if bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]:
                    current_field = field; break

            if current_field in STANDARD_BASELINE:
                std = STANDARD_BASELINE[current_field]
                
                # 1. Forensic Char: Font Size (Height)
                if char['height'] > std['size'][1] * 1.15:
                    scores.append(3.5); types.append(f'STD_ERR_{current_field}_TOO_LARGE')
                elif char['height'] < std['size'][0] * 0.85:
                    scores.append(3.5); types.append(f'STD_ERR_{current_field}_TOO_SMALL')

                # 2. Forensic Char: Boldness (Ink Density)
                if std['bold'] and char['density'] < stats['density_mean'] * 0.9:
                    scores.append(2.0); types.append(f'STD_ERR_{current_field}_EXPECTED_BOLD')

                # 3. Forensic Char: Font Style (Italic/Aspect Ratio)
                if not std['italic'] and char['aspect_ratio'] > stats['aspect_ratio_mean'] * 1.3:
                    scores.append(2.0); types.append(f'STD_ERR_{current_field}_EXPECTED_REGULAR_FONT')

                # 4. Forensic Char: Ink Consistency (Mean Intensity)
                if char['mean_ink_intensity'] < std['ink'][0] or char['mean_ink_intensity'] > std['ink'][1]:
                    scores.append(2.5); types.append(f'STD_ERR_{current_field}_INK_TAMPER_SUSPECT')

                # 5. Forensic Char: Vertical Alignment (Baseline Drift)
                char_base = char['y'] + char['height']
                if abs(char_base - line_avg_y.get(char['line_idx'], char_base)) > std['align']:
                    scores.append(2.0); types.append(f'STD_ERR_{current_field}_FLOATING_TEXT')

            # Global Statistical check for anything else
            h_z = abs(char['height'] - stats['height_mean']) / (stats['height_std'] + 1e-6)
            if h_z > sensitivity:
                scores.append(h_z); types.append('GLOBAL_ANOMALY')

            if scores:
                anomalies.append({
                    'char': char, 'scores': scores, 'types': types,
                    'max_score': max(scores),
                    'severity': 'high' if any('STD_ERR' in t for t in types) else 'medium'
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
        Advanced Hybrid NER: Classifies text based on content patterns 
        and robust spatial anchoring for Albanian ID layouts.
        """
        if not self.ocr_boxes: 
            self.log.append("- NER: No OCR boxes available.")
            return
            
        self.log.append("- Running Pattern-Based NER Classifier...")
        critical_entities = {}
        claimed_indices = set()
        
        # --- Stage 1: Pattern Recognition (High Confidence IDs & Dates) ---
        patterns = {
            'PERSONAL NO': r'^[A-Z]\d{8}[A-Z]$',  # Format: J11120296E
            'ID CARD NO': r'^\d{9}$',             # Exactly 9 digits
            'DATE': r'\d{2}-\d{2}-\d{4}',         # DD-MM-YYYY
            'GENDER': r'^[MF]$',                  # Single letter M or F
        }
        
        for i, box in enumerate(self.ocr_boxes):
            text = box['text'].strip()
            for entity, pat in patterns.items():
                if re.search(pat, text) and entity not in critical_entities:
                    critical_entities[entity] = box['bbox']
                    claimed_indices.add(i)

        # --- Stage 2: Spatial Anchor Linker (Label to Value) ---
        label_map = {
            r'mbiemri|surname': 'SURNAME', 
            r'emri|given': 'GIVEN NAME',
            r'shtet.sia|nationality': 'NATIONALITY', 
            r'vendlindja|place': 'PLACE OF BIRTH',
            r'dat.lindja|date\s*of\s*birth': 'DATE OF BIRTH', 
            r'data\s*e\s*l.shimit|date\s*of\s*issue': 'DATE OF ISSUE',
            r'data\s*e\s*skadimit|date\s*of\s*expiry': 'DATE OF EXPIRY', 
            r'autoriteti|authority': 'AUTHORITY', 
            r'firma|signature': 'SIGNATURE'
        }

        for i, anchor in enumerate(self.ocr_boxes):
            if i in claimed_indices: continue
            text = anchor['text'].strip().lower()
            entity_label = next((name for pat, name in label_map.items() if re.search(pat, text)), None)
            
            if entity_label and entity_label not in critical_entities:
                a_bbox = anchor['bbox']
                a_y_mid = (a_bbox[1] + a_bbox[3]) / 2
                a_x_end = a_bbox[2]

                best_val_idx, min_score = -1, float('inf')
                for j, val_box in enumerate(self.ocr_boxes):
                    if i == j or j in claimed_indices: continue
                    v_bbox = val_box['bbox']
                    v_y_mid = (v_bbox[1] + v_bbox[3]) / 2
                    v_x_start = v_bbox[0]
                    
                    y_diff = abs(v_y_mid - a_y_mid)
                    x_dist = v_x_start - a_x_end

                    # STRICTOR GEOMETRY: 
                    # x_dist < 250 prevents jumping to IDs on the far right
                    # y_diff < 10 ensures text is on the exact same row
                    if 0 < x_dist < 250 and y_diff < 10: 
                        score = (60 * y_diff) + x_dist 
                        if score < min_score:
                            if not any(re.search(p, val_box['text'].lower()) for p in label_map.keys()):
                                min_score, best_val_idx = score, j

                if best_val_idx != -1:
                    critical_entities[entity_label] = self.ocr_boxes[best_val_idx]['bbox']
                    claimed_indices.add(best_val_idx)

        # --- Stage 3: Logical Fallback (Upper-Case Names) ---
        if 'SURNAME' not in critical_entities:
            for i, box in enumerate(self.ocr_boxes):
                if i not in claimed_indices and box['text'].isupper() and len(box['text']) > 3:
                    # Typically Surname is in the top-center area
                    if self.height * 0.1 < box['bbox'][1] < self.height * 0.4:
                        critical_entities['SURNAME'] = box['bbox']
                        claimed_indices.add(i)
                        break

        self.ner_entities = critical_entities

    def perform_ocr(self):
        """PaddleOCR implementation using the shared engine."""
        self.log.append("- OCR: Starting PaddleOCR.")
        
        # Use the injected engine if available, otherwise fallback to creating one
        if self.ocr_engine is None:
            if not PaddleOCR:
                self.ocr_full_text = "OCR NOT AVAILABLE"
                self.log.append("- OCR failed: Library not found.")
                return
            self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)

        try:
            # FORCE detection_limit or use_dilation if text is thin
            result = self.ocr_engine.ocr(np.array(self.original_image), cls=True)

            if not result or not result[0]:
                print("[DEBUG] PaddleOCR returned NOTHING.")
                return

            self.ocr_boxes = []
            for line in result[0]:
                box = line[0]
                text = line[1][0]
                conf = line[1][1]
                
                # Convert to (x1, y1, x2, y2)
                x_coords = [p[0] for p in box]; y_coords = [p[1] for p in box]
                x1, y1, x2, y2 = int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))

                print(f"[OCR DETECTED]: {text} (Conf: {conf:.2f})") # CRITICAL DEBUG LINE
                self.ocr_boxes.append({'bbox': (x1, y1, x2, y2), 'text': text, 'conf': conf})

        except Exception as e:
            self.log.append(f"- PaddleOCR Error: {e}")
            self.ocr_full_text = "OCR ERROR"
            
    def generate_training_features(self):
        """
        Derives a comprehensive feature vector containing all quantifiable metrics 
        from character statistics, background analysis, and anomaly counts. 
        """
        self.log.append("- Deriving Comprehensive Feature Vector for ML.")
        
        # --- 1. Baseline Character Statistics (Global means and STDs) ---
        stats = self.baseline_stats if self.baseline_stats else {}
        total_chars = len(self.characters)
        
        # --- 2. Anomaly Counts and Ratios ---
        geo_anomalies = sum(1 for a in self.anomalies if any(t in a['types'] for t in ['height', 'width', 'aspect_ratio']))
        ink_anomalies = sum(1 for a in self.anomalies if any(t in a['types'] for t in ['ink_gradient', 'ink_gradient_NER_focus']))
        density_anomalies = sum(1 for a in self.anomalies if 'density' in a['types'])

        # --- 3. Feature Vector Assembly ---
        self.forgery_features = {
            'Char_Count': total_chars,
            'H_Mean': stats.get('height_mean', 0), 'H_STD': stats.get('height_std', 0),
            'W_Mean': stats.get('width_mean', 0), 'W_STD': stats.get('width_std', 0),
            'AR_Mean': stats.get('aspect_ratio_mean', 0), 'AR_STD': stats.get('aspect_ratio_std', 0),
            'Ink_Mean': stats.get('ink_mean', 0), 'Ink_STD': stats.get('ink_std', 0),
            'Grad_Mean': stats.get('grad_mean', 0), 'Grad_STD': stats.get('grad_std', 0),
            'Ink_Density_Mean': stats.get('density_mean', 0),
            'Geo_Anomaly_Ratio': geo_anomalies / (total_chars + 1e-6),
            'Ink_Anomaly_Ratio': ink_anomalies / (total_chars + 1e-6),
            'Density_Anomaly_Ratio': density_anomalies / (total_chars + 1e-6),
            'BG_Mean': self.background_stats.get('mean', 0) if self.background_stats else 0,
            'BG_STD': self.background_stats.get('std', 0) if self.background_stats else 0,
            'OCR_Box_Anomalies_Count': len(self.ocr_box_anomalies),
            'BG_Anomaly_Line_Count': len(self.background_anomalies),
            'Clustered_Regions_Count': len(self.suspicious_regions),
        }
        
        self.log.append(f"- Feature vector generated with {len(self.forgery_features)} metrics.")
        return self.forgery_features
    
    def check_standard_baseline_violation(self, field_name, detected_h):
        """
        Validates field values against the official standard sizes.
        Target sizes are based on genuine Albanian ID training profiles.
        """
        # --- THE OFFICIAL STANDARD BASES (Derived from genuine docs) ---
        STANDARD_SIZES = {
            "GIVEN NAME": 11.0, "SURNAME": 11.0, "ID CARD NO": 11.0, 
            "PERSONAL NO": 11.0, "NATIONALITY": 9.0, "PLACE OF BIRTH": 9.0,
            "DATE OF BIRTH": 9.0, "GENDER": 9.0, "DATE OF ISSUE": 9.0,
            "DATE OF EXPIRY": 9.0, "AUTHORITY": 8.5, "SIGNATURE": 25.0
        }
        
        target = STANDARD_SIZES.get(field_name.upper())
        if not target: return

        # Define tolerance margin (e.g., 15% allowance)
        upper_limit, lower_limit = target * 1.15, target * 0.85

        if detected_h > upper_limit or detected_h < lower_limit:
            err_type = 'TOO_LARGE' if detected_h > upper_limit else 'TOO_SMALL_THIN'
            self.anomalies.append({
                'field': field_name,
                'types': [f'STD_ERR_{field_name.upper()}_{err_type}'],
                'description': f"Font size ({detected_h:.1f}) deviates from standard ({target})."
            })
            # Force the feature vector to reflect a geometric anomaly for the ML model
            self.forgery_features['Geo_Anomaly_Ratio'] = max(self.forgery_features.get('Geo_Anomaly_Ratio', 0), 0.90)

    def process_document(self, char_sensitivity=2.0, bg_sensitivity=3.0, ocr_sensitivity=2.5):
        """Orchestrates the full detection pipeline with mandatory sequence."""
        # 1. Start with OCR and Image Preprocessing
        self.perform_ocr()
        self.preprocess_image() 
        
        # 2. Identify Fields and Values
        self.identify_critical_entities_from_ocr() 
        
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

    def visualize_results(self, save_path=None):
        """
        High-visibility visualization for FYP results.
        - Yellow (4px): Raw OCR Detections
        - Green (4px): Correctly Identified NER Fields
        - Red (6px): Detected Forensic Anomalies
        - Verdict: Binary (AUTHENTIC/FORGED)
        """
        # Binary Verdict Logic
        has_forensic_anomalies = (len(self.anomalies) > 0 or 
                                  len(self.background_anomalies) > 0 or 
                                  len(self.ocr_box_anomalies) > 0)
        
        status = "FORGED" if has_forensic_anomalies else "AUTHENTIC"
        
        # Start visualization from the clean original image (preserved by fix above)
        vis = self.original_image.copy()

        # 1. Raw OCR Verification (Thick Yellow - 4px)
        for box in self.ocr_boxes:
            b = box['bbox']
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 4)

        # 2. NER Fields (Bold Green - 4px)
        for field, bbox in self.ner_entities.items():
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 4)
            cv2.putText(vis, field, (bbox[0], bbox[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 3. Anomalies (Ultra Thick Red - 6px)
        for a in self.anomalies:
            bbox = None
            field_name = a.get('field')
            if field_name and field_name in self.ner_entities:
                bbox = self.ner_entities[field_name]
            elif 'char' in a:
                bbox = a['char']['bbox']
            
            if bbox:
                cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Left Side: ALWAYS clean original
        axes[0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Document', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Right Side: Analyzed Document (Clean background + boxes)
        axes[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Forgery Detection Verdict: {status}', 
                          color='red' if status == "FORGED" else 'blue', 
                          fontsize=20, fontweight='bold')
        axes[1].axis('off')
        
        if save_path: 
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
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
            bbox = self.ner_entities[entity]
            entity_value = next((box['text'].strip() for box in self.ocr_boxes if box['bbox'] == bbox), "Value Unknown")
            report.append(f" {entity.ljust(17)} : {entity_value}")
        report.append("-" * 80)
        
        # Logical Content Issues
        report.append("Logical Content Issues:")
        if self.forgery_issues:
            for issue in self.forgery_issues: report.append(f"  - {issue}")
        else: report.append("  - None detected.")
        report.append("-" * 80)
        
        # --- Feature Vector (The Training Feature) ---
        report.append("ML Training Feature Vector (All Metrics):")
        report.append(f"Total Metrics Collected: {len(self.forgery_features)}")
        for name, value in self.forgery_features.items():
             report.append(f"  - {name.ljust(25)} : {value:.4f}")
        report.append("-" * 80)

        # --- FINAL VERDICT (Heuristic Placeholder for UNTRAINED ML) ---
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

def ensure_output_folder():
    """Creates the output folder if it doesn't exist."""
    folder = "PNG_results"
    os.makedirs(folder, exist_ok=True)
    return folder

def analyze_single_document(document_path, char_sensitivity=2.0, bg_sensitivity=3.0, ocr_sensitivity=2.5):
    """
    Main function to orchestrate the analysis for a single document.
    """
    out_folder = ensure_output_folder()
    doc_name = os.path.splitext(os.path.basename(document_path))[0]
    output_png = os.path.join(out_folder, f"{doc_name}_enhanced_analysis.PNG")

    print("\n================================================================================")
    print("      STARTING DOCUMENT FORGERY ANALYSIS (AI-ENHANCED)")
    print("================================================================================")

    try:
        detector = DocumentForgeryDetector(document_path) 
    except Exception as e:
        print(f"\nFATAL ERROR loading document: {e}"); return None

    detector.process_document(
        char_sensitivity=char_sensitivity, bg_sensitivity=bg_sensitivity, ocr_sensitivity=ocr_sensitivity
    )

    print(detector.generate_report())

    print("\n[INFO] Generating visualization...")
    detector.visualize_results(save_path=output_png) 
    print(f"[INFO] Saved visualization to: {output_png}")
    print("================================================================================")
    print("                Analysis Complete!")
    print("================================================================================")

    plt.show()
    return detector

def generate_datasets(input_dir="input_docs"):
    out_folder = ensure_output_folder()
    
    print("\n[INFO] Initializing shared PaddleOCR Engine...")
    shared_engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)

    image_paths = (
        glob.glob(os.path.join(input_dir, '*.jpg')) +
        glob.glob(os.path.join(input_dir, '*.png')) +
        glob.glob(os.path.join(input_dir, '*.jpeg'))
    )
    
    if not image_paths:
        print(f"[FATAL ERROR] No images found in '{input_dir}'.")
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
        'detected_as_forged': 0
    }

    print(f"=========================================================")
    print(f"PROCESSING {len(image_paths)} DOCUMENTS FOR DATASETS")
    print(f"=========================================================")

    for i, path in enumerate(image_paths):
        doc_name = os.path.basename(path)
        doc_root = os.path.splitext(doc_name)[0] 
        is_forged = 'fake' in doc_name.lower()
        label = 1 if is_forged else 0
        
        # Update ground truth counters
        if is_forged: summary['actual_forged'] += 1
        else: summary['actual_genuine'] += 1

        print(f"[{i+1}/{len(image_paths)}] Analyzing: {doc_name}")
        
        try:
            detector = DocumentForgeryDetector(path, ocr_engine=shared_engine)
            detector.process_document()
            
            # Save visual result
            output_png = os.path.join(out_folder, f"{doc_root}_analysis.png")
            detector.visualize_results(save_path=output_png)
            
            # Check if our heuristic flagged it as forged
            if "FORGED" in detector.final_verdict.upper():
                summary['detected_as_forged'] += 1

            # Prepare features
            data_row = {'Document_ID': doc_name, 'Label': label}
            data_row.update(detector.forgery_features)
            
            test_data.append(data_row)
            if not is_forged:
                training_data.append(data_row)
            
            summary['success'] += 1

        except Exception as e:
            print(f"   FAILED to process {doc_name}. Error: {e}")
            summary['failed'] += 1
            continue

    # Write CSVs
    if training_data: write_csv(training_data, "ml_training_data.csv")
    if test_data: write_csv(test_data, "ml_test_data.csv")

    # --- PRINT SUMMARY TABLE ---
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
            print(f"[INFO] Cleaning up old results in '{folder_path}'...")
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'[ERROR] Failed to delete {file_path}. Reason: {e}')
        else:
            os.makedirs(folder_path)

if __name__ == "__main__":

    cleanup_results_folder()
    # --- RUN THIS FUNCTION TO GENERATE YOUR ML TRAINING DATASET (ml_training_data.csv) ---
    generate_datasets()
    
    # --- OR UNCOMMENT BELOW TO RUN A SINGLE DOCUMENT ANALYSIS ---
    # document_to_analyze = "input_docs/alb_id_00.jpg"
    # try:
    #     detector = analyze_single_document(
    #         document_to_analyze, char_sensitivity=2.0, bg_sensitivity=3.0, ocr_sensitivity=2.5    
    #     )
    # except FileNotFoundError:
    #     print("\nCRITICAL ERROR: Input document not found. Please check path.")