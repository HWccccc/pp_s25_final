import cv2
import numpy as np

class NumberRecognizer:
    def __init__(self, ocr_model):
        self.ocr = ocr_model

    def check_box_intersection(self, box1, box2):
        """檢查兩個邊界框是否相交"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        return not (x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2)

    def process_number_recognition_from_teams(self, frame, team_boxes):
        """使用 team_boxes 的邊界框進行數字識別"""
        detected_numbers = []
        for box in team_boxes:
            x1, y1, x2, y2 = map(int, box)  # 使用球隊框的座標

            # 裁剪並放大球隊區域
            cropped_img = frame[y1:y2 + 10, x1 - 10:x2 + 10]
            if cropped_img is None or cropped_img.size == 0:
                cropped_img = frame[y1:y2, x1:x2]

            resized = cv2.resize(cropped_img, None, fx=2.0, fy=2.0,
                                 interpolation=cv2.INTER_LINEAR)

            # OCR識別
            result = self.ocr.ocr(resized, det=True, cls=True)
            for res in result:
                if res is None:
                    continue
                for line in res:
                    text = line[1][0]  # OCR檢測出的文本
                    numbers_only = ''.join(filter(str.isdigit, text))  # 過濾出數字
                    if numbers_only:
                        detected_numbers.append({
                            "number": numbers_only,
                            "box": (x1, y1, x2, y2),
                            "confidence": line[1][1]  # OCR置信度
                        })

        return detected_numbers

    def process_number_recognition(self, frame, number_boxes):
        """處理號碼識別（僅識別數字）"""
        detected_numbers = []
        for box in number_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # 裁剪並放大號碼區域
            cropped_img = frame[y1:y2 + 10, x1 - 10:x2 + 10]
            if cropped_img is None or cropped_img.size == 0:
                cropped_img = frame[y1:y2, x1:x2]

            resized = cv2.resize(cropped_img, None, fx=2.0, fy=2.0,
                               interpolation=cv2.INTER_LINEAR)

            # OCR識別
            result = self.ocr.ocr(resized, det=True, cls=True)
            for res in result:
                if res is None:
                    continue
                for line in res:
                    text = line[1][0]
                    numbers_only = ''.join(filter(str.isdigit, text))
                    if numbers_only:
                        detected_numbers.append({
                            "number": numbers_only,
                            "box": (x1, y1, x2, y2),
                            "confidence": line[1][1]  # 添加置信度
                        })

        return detected_numbers

    def match_numbers_to_players_using_teams(self, frame, player_boxes, number_boxes, team_boxes):
        """將使用 team_boxes 檢測到的數字匹配到球員"""
        detected_numbers = self.process_number_recognition_from_teams(frame, team_boxes)
        player_number_matches = []

        for player_box in player_boxes:
            player_id, px1, py1, px2, py2 = player_box
            player_box_coords = (px1, py1, px2, py2)

            # 檢查與 team_boxes 的交集
            for tx1, ty1, tx2, ty2 in team_boxes:
                team_box_coords = (tx1, ty1, tx2, ty2)

                if self.check_box_intersection(player_box_coords, team_box_coords):
                    # 檢查與檢測到的數字框的交集
                    for number_info in detected_numbers:
                        number_box = number_info['box']
                        number = number_info['number']
                        confidence = number_info['confidence']

                        if self.check_box_intersection(player_box_coords, number_box):
                            player_number_matches.append({
                                "player_id": player_id,
                                "team_box": team_box_coords,
                                "number": number,
                                "confidence": confidence,
                                "box": player_box_coords
                            })

        return player_number_matches

    def match_numbers_to_players(self, frame, player_boxes, number_boxes, team_boxes):
        """將識別到的號碼匹配到球員"""
        detected_numbers = self.process_number_recognition(frame, number_boxes)
        player_number_matches = []

        for player_box in player_boxes:
            player_id, px1, py1, px2, py2 = player_box
            player_box_coords = (px1, py1, px2, py2)

            # 檢查與team boxes的交集
            for tx1, ty1, tx2, ty2, team_name in team_boxes:
                team_box_coords = (tx1, ty1, tx2, ty2)

                if self.check_box_intersection(player_box_coords, team_box_coords):
                    # 檢查與number boxes的交集
                    for number_info in detected_numbers:
                        number_box = number_info['box']
                        number = number_info['number']
                        confidence = number_info['confidence']

                        if (self.check_box_intersection(player_box_coords, number_box) and
                            self.check_box_intersection(team_box_coords, number_box)):

                            player_number_matches.append({
                                "player_id": player_id,
                                "team": team_name,
                                "number": number,
                                "confidence": confidence,
                                "box": player_box_coords
                            })

        return player_number_matches

    def filter_matches(self, matches, confidence_threshold=0.5):
        """過濾並處理重複的匹配"""
        # 按置信度排序
        sorted_matches = sorted(matches, key=lambda x: x['confidence'], reverse=True)

        # 去除重複的player_id，保留置信度最高的
        unique_matches = {}
        for match in sorted_matches:
            player_id = match['player_id']
            if (player_id not in unique_matches and
                match['confidence'] >= confidence_threshold):
                unique_matches[player_id] = match

        return list(unique_matches.values())