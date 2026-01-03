import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pytesseract

# UPDATE THIS PATH IF TESSERACT IS ELSEWHERE
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class DocumentForgeryDetector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise FileNotFoundError("Image not found!")
        self.h, self.w = self.original.shape[:2]
        self.gray = None
        self.binary = None
        self.text_lines = []
        self.chars = []
        self.anomalies = 0
        self.ocr_text = ""

    def preprocess(self):
        print("Preprocessing with maximum clarity...")
        img = self.original.copy()
        
        # LAB enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        
        # Ensure dark text
        if np.mean(gray) < 100:
            gray = 255 - gray
            
        self.gray = gray
        
        # Otsu binary
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        self.binary = binary

    def find_text_lines(self):
        proj = np.sum(self.binary == 255, axis=1)
        threshold = max(80, np.max(proj) * 0.06)
        lines = []
        start = None
        for i, v in enumerate(proj):
            if v > threshold and start is None:
                start = i
            elif v <= threshold and start is not None:
                if i - start > 20:
                    lines.append((max(0, start-20), min(self.h, i+20)))
                start = None
        if start is not None:
            lines.append((max(0, start-20), self.h))
        self.text_lines = lines
        print(f"Found {len(lines)} text lines")

    def segment_chars(self):
        chars = []
        for y1, y2 in self.text_lines:
            roi = self.binary[y1:y2, :]
            proj = np.sum(roi == 255, axis=0)
            thresh = max(40, np.max(proj) * 0.1)
            x1 = None
            for x, val in enumerate(proj):
                if val > thresh and x1 is None:
                    x1 = x
                elif val <= thresh and x1 is not None:
                    if 15 < x - x1 < 180:
                        chars.append((x1, y1, x, y2))
                    x1 = None
            if x1 is not None and self.w - x1 > 15:
                chars.append((x1, y1, self.w, y2))
        self.chars = chars
        print(f"Segmented {len(chars)} characters")

    def ocr(self):
        img = cv2.resize(self.gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        img = cv2.convertScaleAbs(img, alpha=1.15, beta=10)
        config = r'--oem 3 --psm 6 -l eng+sqi'
        text = pytesseract.image_to_string(Image.fromarray(img), config=config)
        self.ocr_text = text.strip()
        print(f"OCR Done → {len(text)} chars")

    def detect_forgery(self):
        if len(self.chars) < 10:
            self.anomalies = 20  # Force flag if segmentation failed
            return
        
        ink_vals = []
        grad = cv2.Sobel(self.gray, cv2.CV_64F, 1, 1, ksize=3)
        grad_mag = np.sqrt(grad**2)
        
        for x1,y1,x2,y2 in self.chars:
            mask = self.binary[y1:y2, x1:x2] == 255
            if np.sum(mask) == 0: continue
            ink = np.mean(self.gray[y1:y2, x1:x2][mask])
            edge = np.mean(grad_mag[y1:y2, x1:x2][mask])
            ink_vals.append((ink, edge))
        
        if len(ink_vals) < 5:
            self.anomalies = 15
            return
            
        inks = [x[0] for x in ink_vals]
        mean_ink = np.mean(inks)
        std_ink = np.std(inks) + 1e-6
        
        outliers = sum(1 for ink, _ in ink_vals if abs(ink - mean_ink) / std_ink > 2.2)
        self.anomalies = outliers

    def run(self):
        self.preprocess()
        self.find_text_lines()
        self.segment_chars()
        self.ocr()
        self.detect_forgery()

        print("\n" + "="*70)
        print("           FINAL FORGERY DETECTION RESULT")
        print("="*70)
        print(f"File: {os.path.basename(self.image_path)}")
        print(f"OCR Text Preview:\n{self.ocr_text[:400]}{'...' if len(self.ocr_text)>400 else ''}")
        print(f"\nCharacters segmented: {len(self.chars)}")
        print(f"Ink/Gradient anomalies: {self.anomalies}")
        
        verdict = "FORGED / TAMPERED" if self.anomalies >= 8 or len(self.chars) < 20 else "AUTHENTIC"
        confidence = max(50, 99 - self.anomalies * 4)
        
        print("="*70)
        print(f"FINAL VERDICT: {verdict}")
        print(f"CONFIDENCE: {confidence}%")
        print("="*70)

        # Visual output
        vis = self.original.copy()
        for bbox in self.chars[:100]:
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
        if verdict == "FORGED / TAMPERED":
            cv2.putText(vis, "FORGED", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,0,255), 15)
        
        os.makedirs("RESULTS", exist_ok=True)
        out_path = f"RESULTS/RESULT_{os.path.basename(self.image_path)}"
        cv2.imwrite(out_path, vis)
        print(f"Result image saved: {out_path}")

# RUN IT
if __name__ == "__main__":
    detector = DocumentForgeryDetector("input_docs/alb_id_21_fake_6_70.jpg")
    detector.run()