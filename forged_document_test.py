import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
import os
from PIL import Image
import datetime 
import csv
import glob
import sys 
import pandas as pd

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
                raise ValueError(f"Unsupported image format: {image_array.shape}")
        except Exception:
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
        self.ocr_boxes = [] 
        self.ner_entities = {} 
        self.suspicious_regions = [] 
        self.forgery_features = {} 
        self.forgery_issues = [] 

    def preprocess_image(self):
        img = self.original_image.copy()
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY) 
        gray = cv2.fastNlMeansDenoising(gray, None, h=6)
        self.gray = gray
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
        return self.binary
    
    def detect_text_regions(self):
        if not self.ocr_boxes:
            if not hasattr(self, 'binary'): self.preprocess_image()
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(self.width * 0.05), 1))
            dilated = cv2.dilate(self.binary, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            lines = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w > int(self.width * 0.10): lines.append((max(0, y-5), min(self.height, y+h+5)))
            self.text_lines = sorted(lines, key=lambda l: l[0])
        return self.text_lines

    def segment_characters(self):
        contours, _ = cv2.findContours(self.binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        chars = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if (w*h) < (self.height*self.width*0.00001): continue
            line_idx = next((i for i, (ys, ye) in enumerate(self.text_lines) if ys <= y+h//2 <= ye), -1)
            if line_idx != -1:
                chars.append({'x':x, 'y':y, 'width':w, 'height':h, 'line_idx':line_idx, 'bbox':(x,y,x+w,y+h), 
                              'aspect_ratio':w/h, 'density':np.sum(self.binary[y:y+h, x:x+w]>0)/(w*h)})
        self.characters = sorted(chars, key=lambda c: (c['line_idx'], c['x']))
        return self.characters

    def calculate_character_ink_analysis(self):
        gx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=5)
        mag = np.sqrt(gx**2 + gy**2)
        for c in self.characters:
            x1, y1, x2, y2 = c['bbox']
            ink = self.gray[y1:y2, x1:x2][self.binary[y1:y2, x1:x2]>0]
            c['mean_ink_intensity'] = np.mean(ink) if ink.size > 0 else 255
            c['mean_gradient'] = np.mean(mag[y1:y2, x1:x2][self.binary[y1:y2, x1:x2]>0]) if ink.size > 0 else 0

    def calculate_background_stats(self):
        bg_px = self.gray[cv2.bitwise_not(self.binary) > 0]
        filt = bg_px[bg_px < 250]
        self.background_stats = {'mean': np.mean(filt if filt.size > 100 else bg_px), 'std': np.std(filt if filt.size > 100 else bg_px)}

    def calculate_baseline_statistics(self):
        if not self.characters: return None
        self.calculate_character_ink_analysis()
        self.baseline_stats = {
            'height_mean': np.mean([c['height'] for c in self.characters]), 'height_std': np.std([c['height'] for c in self.characters]),
            'width_mean': np.mean([c['width'] for c in self.characters]), 'width_std': np.std([c['width'] for c in self.characters]),
            'ink_mean': np.mean([c['mean_ink_intensity'] for c in self.characters]), 'ink_std': np.std([c['mean_ink_intensity'] for c in self.characters]),
            'grad_mean': np.mean([c['mean_gradient'] for c in self.characters]), 'grad_std': np.std([c['mean_gradient'] for c in self.characters]),
            'aspect_ratio_mean': np.mean([c['aspect_ratio'] for c in self.characters]), 'aspect_ratio_std': np.std([c['aspect_ratio'] for c in self.characters]),
            'density_mean': np.mean([c['density'] for c in self.characters]), 'density_std': np.std([c['density'] for c in self.characters]),
        }

    def detect_anomalies(self, sensitivity=2.0):
        if not self.baseline_stats: self.calculate_baseline_statistics()
        for c in self.characters:
            s = self.baseline_stats
            z_scores = [abs(c['height']-s['height_mean'])/s['height_std'], abs(c['mean_gradient']-s['grad_mean'])/s['grad_std']]
            if max(z_scores) > sensitivity:
                self.anomalies.append({'char': c, 'max_score': max(z_scores), 'types': ['outlier']})

    def generate_training_features(self):
        s = self.baseline_stats if self.baseline_stats else {}
        self.forgery_features = {
            'Char_Count': len(self.characters),
            'H_Mean': s.get('height_mean', 0), 'H_STD': s.get('height_std', 0),
            'W_Mean': s.get('width_mean', 0), 'W_STD': s.get('width_std', 0),
            'Ink_Mean': s.get('ink_mean', 0), 'Ink_STD': s.get('ink_std', 0),
            'Grad_Mean': s.get('grad_mean', 0), 'Grad_STD': s.get('grad_std', 0),
            'Geo_Anomaly_Ratio': len(self.anomalies) / (len(self.characters) + 1e-6),
            'BG_Mean': self.background_stats.get('mean', 0) if self.background_stats else 0,
            'BG_STD': self.background_stats.get('std', 0) if self.background_stats else 0,
            'AR_Mean': s.get('aspect_ratio_mean', 0), 'AR_STD': s.get('aspect_ratio_std', 0),
            'Ink_Density_Mean': s.get('density_mean', 0),
            'Ink_Anomaly_Ratio': 0, 'Density_Anomaly_Ratio': 0, 'OCR_Box_Anomalies_Count': 0,
            'BG_Anomaly_Line_Count': 0, 'Clustered_Regions_Count': 0
        }
        return self.forgery_features

    def process_document(self):
        self.preprocess_image()
        self.detect_text_regions()
        self.segment_characters()
        self.calculate_background_stats()
        self.calculate_baseline_statistics()
        self.detect_anomalies()
        self.generate_training_features()

    def visualize_results(self, save_path=None):
        plt.figure(figsize=(10,5))
        plt.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        if save_path: plt.savefig(save_path); plt.close()

def generate_ml_dataset(input_dir="input_docs", output_csv="ml_training_data.csv"):
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    all_data = []
    for path in image_paths:
        try:
            print(f"Processing {os.path.basename(path)}...")
            det = DocumentForgeryDetector(path)
            det.process_document()
            row = {'Document_ID': os.path.basename(path), 'Label': 1 if 'fake' in path.lower() else 0}
            row.update(det.forgery_features)
            all_data.append(row)
        except Exception as e: print(f"Error on {path}: {e}")

    if all_data:
        pd.DataFrame(all_data).to_csv(output_csv, index=False)
        print(f"Saved dataset to {output_csv}")

if __name__ == "__main__":
    generate_ml_dataset()