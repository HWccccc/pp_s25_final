import cv2
import numpy as np

class ImageProcessor:
    @staticmethod
    def resize_image(image, width=None, height=None):
        """調整圖片大小，保持比例"""
        if width is None and height is None:
            return image
        
        h, w = image.shape[:2]
        if width is None:
            aspect_ratio = height / h
            new_width = int(w * aspect_ratio)
            new_height = height
        elif height is None:
            aspect_ratio = width / w
            new_width = width
            new_height = int(h * aspect_ratio)
        else:
            new_width = width
            new_height = height
        
        return cv2.resize(image, (new_width, new_height), 
                         interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def enhance_number_region(image):
        """增強數字區域的圖像品質"""
        # 轉換到灰度圖
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 自適應直方圖均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 二值化
        _, binary = cv2.threshold(enhanced, 0, 255, 
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary

    @staticmethod
    def denoise_image(image):
        """去除圖像噪點"""
        return cv2.fastNlMeansDenoisingColored(image)

    @staticmethod
    def draw_detection_overlay(image, detections, color=(0, 255, 0), thickness=2):
        """在圖像上繪製檢測結果的疊加層"""
        overlay = image.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            if 'label' in det:
                cv2.putText(overlay, det['label'], (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        return overlay

    @staticmethod
    def create_mask(image, polygons):
        """創建遮罩"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for polygon in polygons:
            points = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
        return mask

    @staticmethod
    def apply_mask(image, mask):
        """應用遮罩到圖像"""
        return cv2.bitwise_and(image, image, mask=mask)

    @staticmethod
    def extract_features(image):
        """提取圖像特徵"""
        # 創建 SIFT 特徵提取器
        sift = cv2.SIFT_create()
        # 找到關鍵點和描述符
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return keypoints, descriptors

    @staticmethod
    def match_features(desc1, desc2, ratio=0.75):
        """特徵匹配"""
        # 創建 FLANN 匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # 應用比率測試
        good_matches = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_matches.append(m)
        
        return good_matches