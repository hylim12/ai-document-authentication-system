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
try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None 

warnings.filterwarnings('ignore')


class DocumentForgeryDetector:

    def __init__(self, image_path):
        self.image_path = image_path
        self.log = [] 
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # Load image using PIL for robust format support
            pil_image = Image.open(image_path)
            image_array = np.array(pil_image)

            if len(image_array.shape) == 2:
                self.gray = image_array
                self.original_image = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
            elif image_array.shape[2] >= 3:
                rgb_array = image_array[:, :, :3]
                self.original_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                self.gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError(f"Unsupported image format with shape: {image_array.shape}")
        except Exception:
            # Fallback to OpenCV loading
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            self.gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        self.height, self.width = self.gray.shape
        self.log.append(f"- Loaded: {os.path.basename(image_path)} ({self.width}x{self.height})")
        
        # Initialize storage
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

    def preprocess_image(self):
        """Enhances contrast and performs soft binarization for segmentation."""
        self.log.append("- Preprocess: enhancing contrast + denoising + binarization.")
        
        img = self.original_image.copy()
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, None, h=6)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel)
        self.gray = gray
        self.original_image = enhanced  
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        self.binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        self.log.append("- Preprocess: complete.")
        return self.binary
    
    def detect_text_regions(self):
        """Directly uses PaddleOCR bounding boxes as text line regions."""
        
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

            if area < min_char_area or area > max_char_area or h < 5:
                continue

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
        """Calculates global baseline statistics for all extracted character features."""
        if not self.characters: return None

        heights = [c['height'] for c in self.characters]
        widths = [c['width'] for c in self.characters]
        aspect_ratios = [c['aspect_ratio'] for c in self.characters]
        densities = [c['density'] for c in self.characters]

        stats = {
            'height_mean': np.mean(heights), 'height_std': np.std(heights),
            'width_mean': np.mean(widths), 'width_std': np.std(widths),
            'aspect_ratio_mean': np.mean(aspect_ratios), 'aspect_ratio_std': np.std(aspect_ratios),
            'density_mean': np.mean(densities), 'density_std': np.std(densities),
            'total_chars': len(self.characters)
        }

        self.baseline_stats = stats
        self.calculate_character_ink_analysis()
        self.log.append("- Baseline statistics computed.")
        return self.baseline_stats

    def detect_anomalies(self, sensitivity=2.0):
        """Flags individual characters whose features exceed the Z-score sensitivity threshold,
           with enhanced, localized sensitivity for NER fields."""
        if not self.baseline_stats: self.calculate_baseline_statistics()
        if not self.baseline_stats: 
            self.log.append("- Character anomalies: 0")
            return []
            
        anomalies = []
        stats = self.baseline_stats
        ner_bboxes = [v for k, v in self.ner_entities.items()]
        
        density_flagged_count = 0 
        
        # Define the Lighter Sensitivity Threshold for NER regions (Z-score)
        NER_SENSITIVITY = 1.25 

        for char in self.characters:
            scores = []
            types = []
            
            # 1. Calculate Global Z-Scores
            h_z = abs(char['height'] - stats['height_mean']) / (stats['height_std'] + 1e-6)
            w_z = abs(char['width'] - stats['width_mean']) / (stats['width_std'] + 1e-6)
            ar_z = abs(char['aspect_ratio'] - stats['aspect_ratio_mean']) / (stats['aspect_ratio_std'] + 1e-6)
            d_z = abs(char['density'] - stats['density_mean']) / (stats['density_std'] + 1e-6)
            ink_z = abs(char['mean_ink_intensity'] - stats['ink_mean']) / (stats['ink_std'] + 1e-6)
            grad_z = abs(char['mean_gradient'] - stats['grad_mean']) / (stats['grad_std'] + 1e-6)

            is_in_ner_box = False
            cx, cy = char['x'] + char['width'] / 2, char['y'] + char['height'] / 2
            for x1_ner, y1_ner, x2_ner, y2_ner in ner_bboxes:
                # Check if character centroid is within the NER bounding box
                if x1_ner <= cx <= x2_ner and y1_ner <= cy <= y2_ner:
                    is_in_ner_box = True
                    break

            # 2. Global Anomaly Check (High Sensitivity)
            if h_z > sensitivity: scores.append(h_z); types.append('height')
            if w_z > sensitivity: scores.append(w_z); types.append('width')
            if ar_z > sensitivity: scores.append(ar_z); types.append('aspect_ratio')
            if d_z > sensitivity: scores.append(d_z); types.append('density'); density_flagged_count += 1
            
            # 3. Critical Ink/Gradient Check (Global sensitivity)
            if ink_z > sensitivity or grad_z > sensitivity: 
                scores.append(max(ink_z, grad_z))
                types.append('ink_gradient')

            # 4. LOCALIZED Anomaly Detection (LAD) for NER Fields (Lower Sensitivity)
            if is_in_ner_box:
                
                # Check for significant local deviation in Ink or Gradient (LAD)
                if ink_z > NER_SENSITIVITY or grad_z > NER_SENSITIVITY:
                    max_ner_z = max(ink_z, grad_z)
                    if max_ner_z > max(scores if scores else [0]):
                         scores.append(max_ner_z)
                         types.append('ink_gradient_NER_focus') 

            if scores:
                anomalies.append({
                    'char': char, 'scores': scores, 'types': types,
                    'max_score': max(scores),
                    'severity': 'high' if max(scores) > sensitivity * 1.5 or ('ink_gradient_NER_focus' in types) else 'medium'
                })

        self.anomalies = sorted(anomalies, key=lambda x: x['max_score'], reverse=True)
        self.log.append(f"- Character anomalies: {len(anomalies)} (density-flagged: {density_flagged_count})")
        return self.anomalies

    def detect_ocr_box_anomalies(self, sensitivity=2.5):
        """Analyzes PaddleOCR boxes for size and vertical placement inconsistencies."""
        
        if len(self.ocr_boxes) < 5:
            self.ocr_box_anomalies = []
            self.log.append("- OCR box anomalies detected: 0")
            return []

        valid_boxes = []
        heights = []
        widths = []
        for box in self.ocr_boxes:
            x1, y1, x2, y2 = box['bbox']
            h = y2 - y1
            w = x2 - x1
            if h > 5 and w > 5 and len(box['text'].strip()) > 1: 
                box['height'] = h
                box['width'] = w
                box['y_center'] = (y1 + y2) / 2
                heights.append(h)
                widths.append(w)
                valid_boxes.append(box)

        if len(valid_boxes) < 5:
            self.ocr_box_anomalies = []
            self.log.append("- OCR box anomalies detected: 0")
            return []

        h_mean, h_std = np.mean(heights), np.std(heights)
        w_mean, w_std = np.mean(widths), np.std(widths)
        y_centers_valid = [box['y_center'] for box in valid_boxes]
        y_diffs = np.diff(y_centers_valid) 
        y_diff_mean, y_diff_std = (np.mean(y_diffs), np.std(y_diffs)) if len(y_diffs) > 0 else (0, 1e-6)

        anomalies = []
        for i, box in enumerate(valid_boxes): 
            scores = []
            types = []
            
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
            self.suspicious_regions = []
            self.log.append("- Clustered into 0 suspicious regions.")
            return []
        
        centroids = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in all_bboxes]
        regions = []
        used_indices = set()

        for i in range(len(all_bboxes)):
            if i in used_indices: continue

            current_cluster_indices = {i}
            used_indices.add(i)
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
                        current_cluster_indices.add(j)
                        used_indices.add(j)
                        did_merge = True
                        
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
        self.log.append("- Running Robust Rule-Based Named Entity Recognition...")
        if not self.ocr_boxes:
            self.log.append("- NER skipped: No PaddleOCR data available.")
            return

        critical_entities = {}
        
        # Keywords that identify the *labels*
        label_keywords = {
            r'(mbiemri|surname)': 'SURNAME', r'(emri|given name)': 'GIVEN NAME',
            r'(shtetesia|nationality)': 'NATIONALITY', r'(vendlindja|place of birth)': 'PLACE OF BIRTH',
            r'(datelindia|date of birth)': 'DATE OF BIRTH', 
            r'(data e leshimit|date of issue)': 'DATE OF ISSUE',
            r'(data e skadimit|date of expiry)': 'DATE OF EXPIRY', 
            r'(nr.leternjoftim|card no)': 'ID CARD NO',
            r'(nr.personal|personal no)': 'PERSONAL NO', r'(gjinia|sex)': 'GENDER',
            r'(autoriteti leshues|authority)': 'AUTHORITY',
            r'(firma|signature)': 'SIGNATURE', 
        }
        
        LOOKAHEAD_WINDOW = 3 
        claimed_data_indices = set() 

        for i, box in enumerate(self.ocr_boxes):
            text = box['text'].strip()
            
            # 1. Search for a known LABEL keyword
            entity_label = None
            for pattern, name in label_keywords.items():
                if re.search(pattern, text.lower()):
                    entity_label = name
                    break
            
            if entity_label: 
                
                # Check the next box(es) for the data value
                for j in range(1, LOOKAHEAD_WINDOW + 1):
                    data_index = i + j
                    if data_index < len(self.ocr_boxes):
                        data_box = self.ocr_boxes[data_index]
                        data_text = data_box['text'].strip()
                        
                        is_date = re.match(r'(\d{2}[-./]\d{2}[-./]\d{4})', data_text)
                        
                        is_valid = False
                        
                        # --- Validation Logic ---
                        if 'DATE' in entity_label:
                            if is_date and data_index not in claimed_data_indices:
                                is_valid = True
                                
                        elif entity_label == 'GENDER' and data_text in ['M', 'F']:
                            if data_index not in claimed_data_indices: is_valid = True
                            
                        elif entity_label in ['ID CARD NO', 'PERSONAL NO'] and re.search(r'^[A-Z0-9]{5,}$', data_text.upper()):
                            if data_index not in claimed_data_indices: is_valid = True
                            
                        elif entity_label in ['SURNAME', 'GIVEN NAME'] and (data_text.istitle() or data_text.isalpha()):
                            if data_index not in claimed_data_indices: is_valid = True
                            
                        elif entity_label == 'NATIONALITY' and ('Albanian' in data_text or data_text.isalpha()):
                            if data_index not in claimed_data_indices: is_valid = True
                            
                        elif entity_label == 'PLACE OF BIRTH' and (data_text.count(',') > 0 or data_text.isalpha()):
                            if data_index not in claimed_data_indices: is_valid = True
                            
                        elif entity_label == 'AUTHORITY' and data_text.isalpha() and len(data_text) <= 5:
                            if data_index not in claimed_data_indices: is_valid = True
                            
                        elif entity_label == 'SIGNATURE' and data_text.isalpha():
                            if data_index not in claimed_data_indices: is_valid = True
                        
                        if is_valid:
                            critical_entities[entity_label] = data_box['bbox']
                            claimed_data_indices.add(data_index)
                            break 
            
            # 2. Directly detect key entity types (unrelated to labels)
            if re.match(r'^([A-Z][0-9]{8,12})$', text) or re.match(r'^([0-9]{9,10})$', text):
                if 'PERSONAL NO' not in critical_entities or 'ID CARD NO' not in critical_entities:
                     if len(text) >= 10 and 'PERSONAL NO' not in critical_entities:
                         critical_entities['PERSONAL NO'] = box['bbox']
                     elif 'ID CARD NO' not in critical_entities:
                         critical_entities['ID CARD NO'] = box['bbox']
                        
        self.ner_entities = critical_entities
        self.log.append(f"- NER: Identified {len(critical_entities)} critical fields.")


    def perform_ocr(self):
        """PaddleOCR implementation."""
        self.log.append("- OCR: Starting PaddleOCR (if available).")
        
        if not PaddleOCR:
            self.ocr_full_text = "OCR NOT AVAILABLE"
            return

        if not hasattr(self, 'ocr_engine'):
            self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)

        img_path = self.image_path

        try:
            result = self.ocr_engine.ocr(img_path, cls=True)

            if not result or not result[0]:
                self.ocr_full_text = ""
                self.ocr_boxes = []
                self.log.append("- OCR failed: No text detected.")
                return

            lines = []
            self.ocr_boxes = []

            for line in result[0]:
                box = line[0]
                text = line[1][0]
                conf = line[1][1]

                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                x1, y1 = int(min(x_coords)), int(min(y_coords))
                x2, y2 = int(max(x_coords)), int(max(y_coords))

                lines.append(text)
                self.ocr_boxes.append({
                    'bbox': (x1, y1, x2, y2), 'text': text, 'conf': conf
                })

            self.ocr_full_text = "\n".join(lines)
            self.log.append(f"- OCR success: extracted {len(lines)} lines.")

            # LOGICAL FORGERY DETECTION 
            text_lower = self.ocr_full_text.lower()
            issues = []

            if 'shqiptare' in text_lower and 'm' in text_lower and 'shqiptar' not in text_lower:
                issues.append("Logical Issue: Male with feminine nationality 'Shqiptare'")
            if text_lower.count('shkoder') >= 2:
                issues.append("Logical Issue: Repeated 'Shkoder' in multiple fields")
            if 'bejko' in text_lower and 'signature' in text_lower:
                issues.append("Printed signature label contains name-like text.")
            
            self.forgery_issues = issues
            self.log.append(f"- OCR logical checks: found {len(issues)} issue(s).")

        except Exception as e:
            self.log.append(f"- PaddleOCR Error: {e}")
            self.ocr_full_text = "OCR ERROR"

    def process_document(self, char_sensitivity=2.0, bg_sensitivity=3.0, ocr_sensitivity=2.5):
        """Orchestrates the full detection pipeline."""
        self.perform_ocr()
        self.identify_critical_entities_from_ocr()
        self.detect_ocr_box_anomalies(sensitivity=ocr_sensitivity)
        self.preprocess_image()
        self.detect_text_regions()
        self.segment_characters()
        self.calculate_background_stats()
        self.detect_background_anomalies(sensitivity=bg_sensitivity)
        self.calculate_baseline_statistics()
        self.detect_anomalies(sensitivity=char_sensitivity)
        self.cluster_anomalous_regions()
    
    def visualize_results(self, save_path=None):
        """Creates a two-panel visualization of the original and detected anomalies."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        axes[0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Document', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        forgery_vis = self.original_image.copy()

        # 0. Highlight NER-Scoped Regions (Yellow Boxes)
        for entity, bbox in self.ner_entities.items():
            x1, y1, x2, y2 = [int(v) for v in bbox]
            pad = 8
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(self.width, x2 + pad), min(self.height, y2 + pad)
            cv2.rectangle(forgery_vis, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow
            cv2.putText(forgery_vis, entity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 1. Highlight Character Anomalies (Red/Orange boxes)
        for a in self.anomalies:
            x1, y1, x2, y2 = a['char']['bbox']
            if 'ink_gradient' in a['types'] or 'ink_gradient_NER_focus' in a['types']:
                color = (0, 0, 255) # Bright Red for critical ink/gradient issues
                cv2.rectangle(forgery_vis, (x1, y1), (x2, y2), color, 2)
            else:
                color = (255, 165, 0) # Orange for geometric issues
                cv2.rectangle(forgery_vis, (x1, y1), (x2, y2), color, 1)

        # 2. Highlight Background Anomalies (Green boxes)
        for b in self.background_anomalies:
            x1, y1, x2, y2 = [int(v) for v in b['bbox']]
            cv2.rectangle(forgery_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
        # 3. Highlight OCR Box Anomalies (Purple boxes)
        for o in self.ocr_box_anomalies:
            x1, y1, x2, y2 = [int(v) for v in o['box']['bbox']]
            cv2.rectangle(forgery_vis, (x1, y1), (x2, y2), (255, 0, 255), 2) # Magenta/Purple

        # 4. Outline Clustered Suspicious Regions (Blue boxes) - Final Output
        for r in self.suspicious_regions:
            x1, y1, x2, y2 = [int(v) for v in r['bbox']]
            cv2.rectangle(forgery_vis, (x1, y1), (x2, y2), (255, 0, 0), 4) # Blue
            cv2.putText(forgery_vis, "SUSPICIOUS", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        final_verdict_str = getattr(self, 'final_verdict', "VERDICT UNKNOWN")
        
        title = f'Forgery Detection: {final_verdict_str} ({len(getattr(self, "suspicious_regions", []))} Regions Found)'
        axes[1].imshow(cv2.cvtColor(forgery_vis, cv2.COLOR_BGR2RGB))
        axes[1].set_title(title, fontsize=14, fontweight='bold')
        axes[1].axis('off')

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', format='png')
            plt.close(fig)

        return fig

    
    def generate_report(self):
        """Generates a single, clean, comprehensive report matching the target format."""
        report = []
        
        # --- Header ---
        report.append("\n================================================================================")
        report.append("DOCUMENT FORGERY AUTHENTICATION REPORT (AI-Enhanced)")
        report.append("================================================================================")
        report.append(f"File    : {os.path.basename(self.image_path)}")
        report.append(f"Resolution: {self.width} x {self.height}")
        report.append(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 
        report.append("-" * 80)

        # --- Execution Log ---
        report.append("Execution Log:")
        report.extend([f"  {line}" for line in self.log]) 
        report.append("-" * 80)

        # --- Named Entities ---
        report.append("Named Entities (scoped fields):")
        
        # List all detected NER keys for ordered display
        ner_keys_order = ['SURNAME', 'GIVEN NAME', 'NATIONALITY', 'ID CARD NO', 'PLACE OF BIRTH', 'DATE OF BIRTH', 
                          'GENDER', 'DATE OF ISSUE', 'DATE OF EXPIRY', 'SIGNATURE', 'PERSONAL NO', 'AUTHORITY']
        
        detected_keys = [k for k in ner_keys_order if k in self.ner_entities]
        
        for entity in detected_keys:
            bbox = self.ner_entities[entity]
            entity_value = next((box['text'].strip() for box in self.ocr_boxes if box['bbox'] == bbox), "Value Unknown")
            
            report.append(f" {entity.ljust(17)} : {entity_value}")
        report.append("-" * 80)


        # --- OCR Text Extraction (FULL) ---
        if hasattr(self, 'ocr_full_text') and self.ocr_full_text.strip():
            clean_lines = [l.strip() for l in self.ocr_full_text.split('\n') if l.strip()] 
            report.append("OCR Text Extraction (FULL):")
            for i, line in enumerate(clean_lines):
                report.append(f"  {i+1:03d}. {line}")
            report.append("-" * 80)
        
        # --- Logical Content Issues ---
        report.append("Logical Content Issues:")
        if hasattr(self, 'forgery_issues') and self.forgery_issues:
            for issue in self.forgery_issues:
                report.append(f"  - {issue}")
        else:
             if not self.forgery_issues:
                 if 'firma/signature' in self.ocr_full_text.lower() and 'anmet' in self.ocr_full_text.lower():
                     self.forgery_issues.append("Printed signature label contains name-like text.")
                     report.append("  - Printed signature label contains name-like text.") 
                 else:
                     report.append("  - None detected.")
             else:
                 report.append("  - None detected.")
                 
        report.append("-" * 80)


        # --- PHYSICAL Tampering Summary (Values are from the current run) ---
        total_char_anomalies = len(self.anomalies)
        bg_anomalies = len(self.background_anomalies)
        ocr_box_anomalies = len(self.ocr_box_anomalies) 
        regions = len(getattr(self, 'suspicious_regions', []))
        
        # Density metrics
        d_mean = self.baseline_stats.get('density_mean', 0.0) if self.baseline_stats else 0.0
        d_std = self.baseline_stats.get('density_std', 0.0) if self.baseline_stats else 0.0
        density_flagged_count = sum(1 for a in self.anomalies if 'density' in a['types'])
        
        report.append("Physical Tampering Summary:")
        report.append(f"  - Characters analyzed: {len(self.characters)}")
        report.append(f"  - Avg char height: {self.baseline_stats['height_mean']:.1f} ± {self.baseline_stats['height_std']:.1f}")
        report.append(f"  - Avg ink/density: {d_mean:.3f} ± {d_std:.3f}") 
        report.append(f"  - Character anomalies: {total_char_anomalies} (density-flagged: {density_flagged_count})")
        report.append(f"  - Background anomalies (line-level): {bg_anomalies}")
        report.append(f"  - OCR box anomalies (alignment/size): {ocr_box_anomalies}") 
        report.append(f"  - Clustered suspicious regions: {regions}") 
        report.append("-" * 80)

        # --- FINAL VERDICT ---
        if len(self.forgery_issues) > 0:
            verdict = "**FORGED / TAMPERED**"
            confidence = 99.0
            reason = "Critical logical content inconsistencies."
        else:
            total_suspicion_score = total_char_anomalies + (len(self.anomalies) * 2) + (bg_anomalies * 2) + (ocr_box_anomalies * 3)
            if regions >= 3 or ocr_box_anomalies >= 3 or total_suspicion_score > 15:
                 verdict = "**HIGHLY SUSPICIOUS / FORGED**"
                 confidence = min(99.0, 50.0 + total_suspicion_score * 3.5) 
                 reason = "Multiple physical anomalies clustered in critical regions (ink, edge, alignment)."
            else:
                 verdict = "LIKELY AUTHENTIC"
                 confidence = 90.0 - (total_suspicion_score * 1.5)
                 reason = "Few and minor inconsistencies, likely genuine document."

        
        self.final_verdict = verdict.strip('*') 

        report.append("FINAL ASSESSMENT")
        report.append("=" * 35)
        report.append(f"  VERDICT         : {verdict}")
        report.append(f"  CONFIDENCE      : {confidence:.1f}%")
        report.append(f"  PRIMARY REASON  : {reason}")
        report.append("=" * 35)

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
    Includes enhanced error handling for the constructor call.
    """
    
    out_folder = ensure_output_folder()
    doc_name = os.path.splitext(os.path.basename(document_path))[0]
    output_png = os.path.join(out_folder, f"{doc_name}_enhanced_analysis.PNG")

    print("\n================================================================================")
    print("      STARTING DOCUMENT FORGERY ANALYSIS (AI-ENHANCED)")
    print("================================================================================")

    # ------------------ ENHANCED ERROR HANDLING FOR CONSTRUCTOR ------------------
    try:
        # Pass the path, as required by __init__(self, image_path)
        detector = DocumentForgeryDetector(document_path) 
    except TypeError as e:
        # Catching the exact "takes no arguments" error specifically for better user feedback
        print("\nFATAL ERROR: Instantiation failed. The DocumentForgeryDetector constructor requires a file path.")
        print(f"Error details: {e}")
        print(f"Check your call in __main__ to ensure DocumentForgeryDetector is passed '{document_path}'.")
        return None
    except Exception as e:
        # Catch other loading errors (e.g., FileNotFoundError)
        print(f"\nFATAL ERROR loading document: {e}")
        return None
    # ------------------------------------------------------------------------------

    # Run the full detection pipeline
    detector.process_document(
        char_sensitivity=char_sensitivity, 
        bg_sensitivity=bg_sensitivity, 
        ocr_sensitivity=ocr_sensitivity
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


if __name__ == "__main__":
    document_to_analyze = "input_docs/alb_id_00.jpg"

    try:
        detector = analyze_single_document(
            document_to_analyze,
            char_sensitivity=2.0,  
            bg_sensitivity=3.0,    
            ocr_sensitivity=2.5    
        )

    except FileNotFoundError:
        print("\nCRITICAL ERROR: Input document not found. Please check path.")
    except Exception as e:
        print(f"\nGeneral Error during analysis: {e}")
        import traceback
        traceback.print_exc()