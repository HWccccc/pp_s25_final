import cv2
import numpy as np
import torch
from threading import Lock
from paddleocr import PaddleOCR
import pandas as pd
import os
import base64
from ultralytics import YOLO
from .court_analyzer import extract_geometric_points
from .number_recognizer import NumberRecognizer
from .scoring_analyzer import ScoringAnalyzer
from .coordinate_mapper import CoordinateMapper
from utils.data_manager import DataManager
import time
from collections import defaultdict
import multiprocessing as mp
import sys

# ===== Pipeline 階段處理 Workers (保持原有不變) =====
def pipeline_stage1_worker(input_q, output_q, player_model_path, court_model_path, device):
    """Pipeline 階段1：模型推理 + 物件分類"""
    print("[Pipeline階段1] 模型推理 + 物件分類 Worker 啟動...")
    
    # 載入模型
    player_model = YOLO(player_model_path).to(device)
    court_model = YOLO(court_model_path).to(device)
    
    # 類別映射
    team_mapping = {0: '3-s', 2: 'biv'}
    
    with torch.no_grad():
        while True:
            task = input_q.get()
            if task is None:
                print("[Pipeline階段1] 收到結束訊號")
                output_q.put(None)
                break
            
            frame_id, frame = task
            start_time = time.time()
            
            # 執行實際的 YOLO 推理
            player_results = player_model.track(source=frame, conf=0.3, persist=True, tracker="bytetrack.yaml")
            court_results = court_model.track(source=frame, conf=0.25, persist=True, tracker="bytetrack.yaml")
            
            # 初始化分類結果
            player_boxes = []
            team_boxes = []
            number_boxes = []
            basketball_data = None
            hoop_data = None
            
            # 處理 player model 結果並分類
            if player_results:
                for result in player_results:
                    if result.boxes is not None:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            cls = int(box.cls[0].item())
                            track_id = int(box.id[0].item()) if box.id is not None else -1
                            
                            # 根據類別分類處理
                            if cls == 1:  # ball
                                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                                left = (x1, y1)
                                right = (x2, y2)
                                basketball_data = {
                                    'center': center,
                                    'left': left, 
                                    'right': right
                                }
                                
                            elif cls == 3:  # hoop
                                center = ((x1 + x2) / 2, y1)
                                left = (x1, y1)
                                right = (x2, y2)
                                hoop_data = {
                                    'center': center,
                                    'left': left,
                                    'right': right
                                }
                                
                            elif cls == 5:  # player
                                player_boxes.append({
                                    'track_id': track_id,
                                    'x1': float(x1), 'y1': float(y1),
                                    'x2': float(x2), 'y2': float(y2)
                                })
                                
                            elif cls == 0 or cls == 2:  # team boxes
                                team_name = team_mapping.get(cls, 'unknown')
                                team_boxes.append({
                                    'x1': float(x1), 'y1': float(y1),
                                    'x2': float(x2), 'y2': float(y2),
                                    'team': team_name
                                })
                                
                            elif cls == 4:  # number
                                number_boxes.append({
                                    'x1': float(x1), 'y1': float(y1),
                                    'x2': float(x2), 'y2': float(y2),
                                    'cls': cls,
                                    'track_id': track_id
                                })
            
            # 處理 court model 結果
            court_boxes_data = []
            if court_results:
                for result in court_results:
                    if result.boxes is not None:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            cls = int(box.cls[0].item())
                            track_id = int(box.id[0].item()) if box.id is not None else -1
                            
                            court_boxes_data.append({
                                'x1': float(x1), 'y1': float(y1), 
                                'x2': float(x2), 'y2': float(y2),
                                'cls': cls, 'track_id': track_id
                            })
            
            inference_time = time.time() - start_time
            
            # 傳遞已分類的資料 + 原始幀
            results = {
                'frame_id': frame_id,
                'original_frame': frame,  # 加入原始幀
                'player_boxes': player_boxes,
                'team_boxes': team_boxes,
                'number_boxes': number_boxes,
                'basketball_data': basketball_data,
                'hoop_data': hoop_data,
                'court_boxes': court_boxes_data,
                'inference_time': inference_time,
                'status': 'classified'
            }
            
            output_q.put((frame_id, results))
            print(f"[Pipeline階段1] 完成 frame_{frame_id}，分類: {len(player_boxes)}球員, {len(team_boxes)}隊伍, 球:{basketball_data is not None}, 籃框:{hoop_data is not None}")

def pipeline_stage2_worker(input_q, output_q):
    """Pipeline 階段2：OCR 號碼識別"""
    print("[Pipeline階段2] OCR 號碼識別 Worker 啟動...")
    
    # 初始化 OCR
    from paddleocr import PaddleOCR
    from .number_recognizer import NumberRecognizer
    import cv2
    import numpy as np
    
    ocr = PaddleOCR(lang='en',
                   rec_model_dir="./path_to/en_PP-OCRv3_rec_infer", 
                   cls_model_dir="./path_to/ch_ppocr_mobile_v2.0_cls_infer",
                   use_angle_cls=False,
                   use_gpu=True,
                   show_log=False)
    
    number_recognizer = NumberRecognizer(ocr)
    print("[Pipeline階段2] OCR 模型載入完成")
    
    while True:
        task = input_q.get()
        if task is None:
            print("[Pipeline階段2] 收到結束訊號")
            output_q.put(None)
            break
        
        frame_id, stage1_results = task
        start_time = time.time()
        
        print(f"[Pipeline階段2] 處理 frame_{frame_id} OCR 號碼識別")
        
        # 提取號碼識別需要的資料
        number_boxes = stage1_results.get('number_boxes', [])
        player_boxes = stage1_results.get('player_boxes', [])
        team_boxes = stage1_results.get('team_boxes', [])
        original_frame = stage1_results.get('original_frame')
        
        ocr_matches = []
        
        if number_boxes and original_frame is not None:
            print(f"[Pipeline階段2] 找到 {len(number_boxes)} 個號碼區域，開始 OCR 識別")
            
            try:
                # 簡化版本：直接對號碼區域進行 OCR
                for i, num_box in enumerate(number_boxes):
                    x1, y1, x2, y2 = int(num_box['x1']), int(num_box['y1']), int(num_box['x2']), int(num_box['y2'])
                    
                    # 確保座標在有效範圍內
                    h, w = original_frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        # 裁切號碼區域
                        number_region = original_frame[y1:y2, x1:x2]
                        
                        # 進行 OCR 識別
                        ocr_result = ocr.ocr(number_region, cls=False)
                        
                        if ocr_result and ocr_result[0]:
                            for line in ocr_result[0]:
                                text = line[1][0]
                                confidence = line[1][1]
                                
                                # 檢查是否為數字
                                if text.isdigit() and confidence > 0.5:
                                    # 尋找最近的球員
                                    min_dist = float('inf')
                                    closest_player = None
                                    num_center_x = (x1 + x2) / 2
                                    num_center_y = (y1 + y2) / 2
                                    
                                    for player_box in player_boxes:
                                        px1, py1, px2, py2 = player_box['x1'], player_box['y1'], player_box['x2'], player_box['y2']
                                        player_center_x = (px1 + px2) / 2
                                        player_center_y = (py1 + py2) / 2
                                        
                                        dist = ((num_center_x - player_center_x) ** 2 + (num_center_y - player_center_y) ** 2) ** 0.5
                                        if dist < min_dist:
                                            min_dist = dist
                                            closest_player = player_box
                                    
                                    # 尋找最近的隊伍
                                    closest_team = "unknown"
                                    min_team_dist = float('inf')
                                    for team_box in team_boxes:
                                        tx1, ty1, tx2, ty2 = team_box['x1'], team_box['y1'], team_box['x2'], team_box['y2']
                                        team_center_x = (tx1 + tx2) / 2
                                        team_center_y = (ty1 + ty2) / 2
                                        
                                        dist = ((num_center_x - team_center_x) ** 2 + (num_center_y - team_center_y) ** 2) ** 0.5
                                        if dist < min_team_dist:
                                            min_team_dist = dist
                                            closest_team = team_box['team']
                                    
                                    if closest_player and min_dist < 100:  # 距離閾值
                                        ocr_matches.append({
                                            'player_id': closest_player['track_id'],
                                            'team': closest_team,
                                            'number': int(text),
                                            'confidence': confidence,
                                            'ocr_text': text,
                                            'distance': min_dist
                                        })
                                        
                                        print(f"[Pipeline階段2] 識別號碼: {text} (信心度: {confidence:.2f}) -> 球員ID {closest_player['track_id']} 隊伍 {closest_team}")
                
                print(f"[Pipeline階段2] 成功識別 {len(ocr_matches)} 個號碼匹配")
                
            except Exception as e:
                print(f"[Pipeline階段2] OCR 識別錯誤: {e}")
                import traceback
                traceback.print_exc()
                ocr_matches = []
        
        elif number_boxes:
            print(f"[Pipeline階段2] 找到 {len(number_boxes)} 個號碼區域，但缺少原始幀")
        else:
            print(f"[Pipeline階段2] 未找到號碼區域")
        
        ocr_time = time.time() - start_time
        
        # 將 OCR 結果加入 stage1 結果中，保持原始幀傳遞
        results = stage1_results.copy()
        results.update({
            'ocr_matches': ocr_matches,
            'ocr_time': ocr_time,
            'stage2_status': 'ocr_completed'
        })
        
        output_q.put((frame_id, results))
        print(f"[Pipeline階段2] 完成 frame_{frame_id} OCR 處理，耗時: {ocr_time:.3f}s")

def pipeline_stage3_worker(input_q, output_q, frame_completion_events, config):
    """Pipeline 階段3：進籃分析 + 球場映射 + 狀態更新 + 畫面繪製 (必須按順序)"""
    print("[Pipeline階段3] 進籃分析 + 球場映射 + 狀態更新 Worker 啟動...")
    
    # 初始化分析器 (在 Worker 內部建立)
    from .scoring_analyzer import ScoringAnalyzer
    from .coordinate_mapper import CoordinateMapper
    import cv2
    import numpy as np
    
    scoring_analyzer = ScoringAnalyzer()
    coordinate_mapper = CoordinateMapper()
    
    # 從 config 取得設定
    output_width = config['output_width']
    output_height = config['output_height']
    class_names = config['class_names']
    team_mapping = config['team_mapping']
    
    # 球場相關設定
    court_image = None
    image_points = None
    if config.get('court_image_path'):
        court_image = cv2.imread(config['court_image_path'])
        image_points = config.get('image_points')
        if image_points is not None:
            coordinate_mapper.set_reference(config['court_image_path'], image_points)
    
    # 顏色生成函數
    def get_color_from_id(track_id):
        if track_id is None:
            return (255, 255, 255)
        seed_value = abs(hash(str(track_id))) % (2 ** 32 - 1)
        np.random.seed(seed_value)
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        return color
    
    while True:
        task = input_q.get()
        if task is None:
            print("[Pipeline階段3] 收到結束訊號")
            output_q.put(None)
            break
        
        frame_id, stage2_results = task
        
        # 等待前一幀完成 (保持時序)
        previous_event, current_event = frame_completion_events.get(frame_id, (None, None))
        if previous_event:
            print(f"[Pipeline階段3] frame_{frame_id} 等待前一幀...")
            previous_event.wait()
        
        start_time = time.time()
        print(f"[Pipeline階段3] 處理 frame_{frame_id} 分析與狀態更新")
        
        # 提取所需資料
        basketball_data = stage2_results.get('basketball_data')
        hoop_data = stage2_results.get('hoop_data')
        player_boxes = stage2_results.get('player_boxes', [])
        court_boxes = stage2_results.get('court_boxes', [])
        team_boxes = stage2_results.get('team_boxes', [])
        ocr_matches = stage2_results.get('ocr_matches', [])
        original_frame = stage2_results.get('original_frame')  # 需要從 stage1 傳遞原始幀
        
        # === 1. 進籃分析 ===
        scores_updated = False
        if basketball_data and hoop_data:
            # 轉換資料格式為原本的 tuple 格式
            ball_tuple = (basketball_data['center'], basketball_data['left'], basketball_data['right'])
            hoop_tuple = (hoop_data['center'], hoop_data['left'], hoop_data['right'])
            
            scoring_analyzer.update_positions(ball_tuple, hoop_tuple)
            
            # 檢查得分
            scored, team = scoring_analyzer.check_scoring(basketball_data['center'], hoop_data['center'])
            if scored:
                print(f"[Pipeline階段3] 進球！得分方：{team}")
                scores_updated = True
        
        # 預測籃球位置（如果沒有偵測到）
        if not basketball_data:
            predicted_pos = scoring_analyzer.predict_basketball_position()
            if predicted_pos:
                basketball_data = {'center': predicted_pos, 'left': None, 'right': None}
        
        # === 2. 球場映射與軌跡 ===
        court_frame = None
        if court_image is not None:
            court_frame = court_image.copy()
            
            # 處理球場映射
            current_video_points = None
            for court_box in court_boxes:
                if court_box['cls'] == 1:  # RA area (假設)
                    # 這裡需要 extract_geometric_points 邏輯
                    # 暫時用簡化版本
                    x1, y1, x2, y2 = court_box['x1'], court_box['y1'], court_box['x2'], court_box['y2']
                    current_video_points = np.array([
                        [x1, y1], [x1, y2], [x2, y1], [x2, y2]
                    ], dtype=np.float32)
                    break
            
            # 處理球員軌跡
            if current_video_points is not None and image_points is not None:
                try:
                    H, _ = cv2.findHomography(current_video_points, image_points)
                    
                    for player_box in player_boxes:
                        track_id = player_box['track_id']
                        x1, y1, x2, y2 = player_box['x1'], player_box['y1'], player_box['x2'], player_box['y2']
                        
                        # 計算球員腳部位置
                        player_point = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32).reshape(1, 1, 2)
                        projected_point = cv2.perspectiveTransform(player_point, H)
                        x, y = int(projected_point[0][0][0]), int(projected_point[0][0][1])
                        
                        # 追蹤移動
                        total_distance = coordinate_mapper.track_movement(track_id, [x, y])
                        
                        # 繪製軌跡
                        if track_id in coordinate_mapper.tracking_data:
                            positions = coordinate_mapper.tracking_data[track_id]['positions']
                            color = coordinate_mapper.tracking_data[track_id]['color']
                            
                            # 繪製軌跡線
                            if len(positions) > 1:
                                for i in range(len(positions) - 1):
                                    pt1 = tuple(map(int, positions[i]))
                                    pt2 = tuple(map(int, positions[i + 1]))
                                    cv2.line(court_frame, pt1, pt2, color, 2)
                            
                            # 繪製當前位置
                            cv2.circle(court_frame, (x, y), 5, color, -1)
                            cv2.putText(court_frame, f"ID:{track_id} Dist:{total_distance:.1f}m",
                                       (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                except Exception as e:
                    print(f"[Pipeline階段3] 球場映射錯誤: {e}")
        
        # === 4. 處理 OCR 號碼匹配結果 ===
        if ocr_matches:
            print(f"[Pipeline階段3] 處理 {len(ocr_matches)} 個 OCR 號碼匹配")
            # 這裡需要更新球員註冊表和關聯
            # 但由於我們在 Worker 中，無法直接存取主進程的 player_registry
            # 暫時先記錄，或者考慮其他方式同步
            for match in ocr_matches:
                print(f"[Pipeline階段3] 識別到球員: ID {match['player_id']}, 隊伍 {match['team']}, 號碼 {match['number']}")
        
        # === 3. 畫面繪製 ===
        # 使用原始影片作為底圖，如果沒有就建立黑色畫面
        if original_frame is not None:
            processed_frame = original_frame.copy()
            # 縮放到輸出尺寸
            processed_frame = cv2.resize(processed_frame, (output_width, output_height))
        else:
            # 如果沒有原始幀，建立黑色畫面
            processed_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # 繪製籃球
        if basketball_data and basketball_data['center']:
            center = basketball_data['center']
            left = basketball_data['left']
            right = basketball_data['right']
            if left and right:
                # 調整座標到輸出尺寸
                x1 = int(left[0] * output_width / (original_frame.shape[1] if original_frame is not None else output_width))
                y1 = int(left[1] * output_height / (original_frame.shape[0] if original_frame is not None else output_height))
                x2 = int(right[0] * output_width / (original_frame.shape[1] if original_frame is not None else output_width))
                y2 = int(right[1] * output_height / (original_frame.shape[0] if original_frame is not None else output_height))
                
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_frame, "BALL", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 繪製球員
        for player_box in player_boxes:
            track_id = player_box['track_id']
            x1, y1, x2, y2 = player_box['x1'], player_box['y1'], player_box['x2'], player_box['y2']
            
            # 調整座標到輸出尺寸
            if original_frame is not None:
                x1 = int(x1 * output_width / original_frame.shape[1])
                y1 = int(y1 * output_height / original_frame.shape[0])
                x2 = int(x2 * output_width / original_frame.shape[1])
                y2 = int(y2 * output_height / original_frame.shape[0])
            
            color = get_color_from_id(track_id)
            
            # 檢查是否有對應的 OCR 識別結果
            player_label = f"Player {track_id}"
            for match in ocr_matches:
                if match['player_id'] == track_id:
                    player_label = f"{match['team']} #{match['number']}"
                    break
            
            cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(processed_frame, player_label,
                       (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 繪製隊伍
        for team_box in team_boxes:
            x1, y1, x2, y2 = team_box['x1'], team_box['y1'], team_box['x2'], team_box['y2']
            team = team_box['team']
            
            # 調整座標到輸出尺寸
            if original_frame is not None:
                x1 = int(x1 * output_width / original_frame.shape[1])
                y1 = int(y1 * output_height / original_frame.shape[0])
                x2 = int(x2 * output_width / original_frame.shape[1])
                y2 = int(y2 * output_height / original_frame.shape[0])
            
            color = (255, 0, 0) if team == '3-s' else (0, 0, 255)
            
            cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(processed_frame, team, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 繪製籃框
        if hoop_data and hoop_data['center']:
            center = hoop_data['center']
            left = hoop_data['left']
            right = hoop_data['right']
            if left and right:
                # 調整座標到輸出尺寸
                x1 = int(left[0] * output_width / (original_frame.shape[1] if original_frame is not None else output_width))
                y1 = int(left[1] * output_height / (original_frame.shape[0] if original_frame is not None else output_height))
                x2 = int(right[0] * output_width / (original_frame.shape[1] if original_frame is not None else output_width))
                y2 = int(right[1] * output_height / (original_frame.shape[0] if original_frame is not None else output_height))
                
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(processed_frame, "HOOP", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # 縮放球場畫面
        if court_frame is not None:
            court_frame = cv2.resize(court_frame, (output_width, output_height))
        
        # 獲取當前得分
        current_scores = scoring_analyzer.get_scores()
        
        final_results = {
            'processed_frame': processed_frame,
            'court_frame': court_frame,
            'detected_players': [],  # 簡化版本先不處理
            'scores': current_scores,
            'scores_updated': scores_updated,
            'stage3_time': time.time() - start_time,
            'stage3_status': 'completed'
        }
        
        output_q.put((frame_id, final_results))
        
        # 通知下一幀可以開始
        if current_event:
            current_event.set()
            print(f"[Pipeline階段3] frame_{frame_id} 完成，得分: {current_scores}")

class BasketballTracker:
    def __init__(self, player_model_path, court_model_path, data_folder, max_parallel_frames=3):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用設備: {self.device}")

        # 儲存模型路徑（用於多進程）
        self.player_model_path = player_model_path
        self.court_model_path = court_model_path

        # ===== 新增：並行控制參數 =====
        self.max_parallel_frames = max_parallel_frames
        self.frames_in_pipeline = 0
        self.pending_results = {}  # frame_id -> 等待的結果
        self.next_expected_frame = 1  # 下一個期待的輸出 frame

        # ===== Pipeline 多進程設置 =====
        if sys.platform in ['win32', 'darwin']:
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass

        self.ctx = mp.get_context("spawn")

        # 建立 Pipeline 三階段佇列（支援並行）
        self.stage1_input_q = self.ctx.Queue(maxsize=max_parallel_frames)
        self.stage1_to_stage2_q = self.ctx.Queue(maxsize=max_parallel_frames)  
        self.stage2_to_stage3_q = self.ctx.Queue(maxsize=max_parallel_frames)
        self.stage3_output_q = self.ctx.Queue(maxsize=max_parallel_frames)

        # 幀完成事件字典 (用於階段3順序控制)
        self.frame_completion_events = {}
        self.previous_frame_event = None

        # 啟動 Pipeline 三個 Workers
        self.stage1_process = self.ctx.Process(
            target=pipeline_stage1_worker,
            args=(self.stage1_input_q, self.stage1_to_stage2_q, 
                  player_model_path, court_model_path, self.device)
        )

        self.stage2_process = self.ctx.Process(
            target=pipeline_stage2_worker,
            args=(self.stage1_to_stage2_q, self.stage2_to_stage3_q)
        )

        # 準備 Stage3 需要的分析器和配置
        stage3_config = {
            'output_width': 640,
            'output_height': 480,
            'class_names': ['3-s', 'ball', 'biv', 'hoop', 'number', 'player'],
            'team_mapping': {0: '3-s', 2: 'biv'},
            'court_image_path': None,  # 會在 set_court_reference 時更新
            'image_points': None
        }

        self.stage3_process = self.ctx.Process(
            target=pipeline_stage3_worker,
            args=(self.stage2_to_stage3_q, self.stage3_output_q, 
                  self.frame_completion_events, stage3_config)
        )

        self.stage1_process.start()
        self.stage2_process.start()
        self.stage3_process.start()

        print(f"Pipeline 多進程模式已啟動，最大並行幀數: {max_parallel_frames}")

        import paddle
        print(paddle.is_compiled_with_cuda())  # 如果返回 False，表示 Paddle 未啟用 GPU 支援
        print(paddle.device.get_device())  # 檢查當前使用的設備

        self.ocr = PaddleOCR(lang='en',
                             rec_model_dir="./path_to/en_PP-OCRv3_rec_infer",  # 英文識別模型（高精度）
                             cls_model_dir="./path_to/ch_ppocr_mobile_v2.0_cls_infer",  # 若要啟用方向分類器
                             use_angle_cls=False,
                             use_gpu=True,
                             show_log=False)
        print(f"OCR模型載入完成")

        # 初始化資料管理器
        self.data_folder = data_folder
        self.data_manager = DataManager(data_folder)

        # 初始化各種分析器
        self.number_recognizer = NumberRecognizer(self.ocr)
        self.scoring_analyzer = ScoringAnalyzer()
        self.coordinate_mapper = CoordinateMapper()

        # 性能分析相關
        self.performance_metrics = defaultdict(list)
        self.total_frames_processed = 0
        self.total_processing_time = 0

        # 用來儲存長期的球員狀態（例如分數、體能）
        self.player_states = {}  # player_key -> {score, stamina, total_distance, max_stamina}
        self.max_distance = 100  # 最大跑動距離設為1000m

        # track_id -> player_key (ex: "3-s_5")
        self.player_associations = {}

        # player_key -> {team, number, track_id, last_seen, ...}
        self.player_registry = {}

        self.frame_count = 0
        self.association_timeout = 30

        # 追踪相關設定
        self.class_names = ['3-s', 'ball', 'biv', 'hoop', 'number', 'player']
        self.team_mapping = {
            0: '3-s',
            2: 'biv'
        }
        self.lock = Lock()

        # 移除 score 從顯示選項中
        self.display_options = {
            'player': True,
            'ball': True,
            'team': True,
            'number': True,
            'trajectory': True
        }

        # 座標映射相關
        self.court_image = None
        self.image_points = None
        self.current_video_points = None

        # 播放控制相關
        self.is_playing = False
        self.video_capture = None
        self.current_frame = None
        self.frame_buffer = []
        self.buffer_size = 60
        self.current_frame_index = 0

        # 新增得分回調
        self.score_callback = None

        # 設定輸出解析度
        self.output_width = 640
        self.output_height = 480

    def process_frame(self, frame):
        """修改版：支援流水線並行處理"""
        if frame is None:
            return None, None, None

        frame_start_time = time.time()
        self.frame_count += 1
        self.total_frames_processed += 1

        # 1. 檢查是否需要先收集結果以控制並行數量
        if self.frames_in_pipeline >= self.max_parallel_frames:
            print(f"[主進程] 達到最大並行數 {self.max_parallel_frames}，先收集一個結果")
            self._collect_one_result()

        # 2. 設置當前幀的完成事件
        current_frame_event = self.ctx.Event()
        self.frame_completion_events[self.frame_count] = (self.previous_frame_event, current_frame_event)
        self.previous_frame_event = current_frame_event

        # 3. 送入新 frame 到 Pipeline（非阻塞）
        frame_copy = frame.copy()
        self.stage1_input_q.put((self.frame_count, frame_copy))
        self.frames_in_pipeline += 1
        print(f"[主進程] 將 frame_{self.frame_count} 送入 Pipeline，目前並行: {self.frames_in_pipeline}")

        # 4. 嘗試取得已完成的結果（依序輸出）
        result = self._get_next_sequential_result()
        
        if result is not None:
            frame_id, final_results = result
            print(f"[主進程] 輸出 frame_{frame_id} 結果")
            
            # 取得 Pipeline 處理結果
            processed_frame = final_results.get('processed_frame')
            court_frame = final_results.get('court_frame')
            detected_players = final_results.get('detected_players', [])
            scores = final_results.get('scores', {})
            scores_updated = final_results.get('scores_updated', False)
            
            # 如果得分更新，觸發回調
            if scores_updated and self.score_callback:
                self.score_callback(scores)
            
            # 更新總處理時間
            self.total_processing_time += time.time() - frame_start_time
            
            return processed_frame, court_frame, detected_players
        else:
            # 如果沒有可輸出的結果，返回 None（表示需要等待）
            print(f"[主進程] frame_{self.frame_count} 已送入，等待結果...")
            return None, None, None

    def _collect_one_result(self):
        """收集一個結果以控制並行數量"""
        try:
            frame_id, final_results = self.stage3_output_q.get(timeout=30)
            self.frames_in_pipeline -= 1
            
            # 暫存結果，等待依序輸出
            self.pending_results[frame_id] = final_results
            print(f"[主進程] 收集到 frame_{frame_id} 結果，等待依序輸出")
            
        except Exception as e:
            print(f"[主進程] 收集結果錯誤: {e}")
            self.frames_in_pipeline = max(0, self.frames_in_pipeline - 1)

    def _get_next_sequential_result(self):
        """取得下一個依序的結果"""
        # 先檢查是否有新完成的結果
        while not self.stage3_output_q.empty():
            try:
                frame_id, final_results = self.stage3_output_q.get_nowait()
                self.frames_in_pipeline -= 1
                self.pending_results[frame_id] = final_results
                print(f"[主進程] 收集到 frame_{frame_id} 結果")
            except:
                break

        # 檢查是否可以輸出下一個期待的 frame
        if self.next_expected_frame in self.pending_results:
            result = (self.next_expected_frame, self.pending_results[self.next_expected_frame])
            del self.pending_results[self.next_expected_frame]
            self.next_expected_frame += 1
            return result
        
        return None

    def get_pending_results(self):
        """強制取得所有 pending 的結果（用於視頻結束時）"""
        results = []
        
        # 收集剩餘的所有結果
        while self.frames_in_pipeline > 0:
            try:
                frame_id, final_results = self.stage3_output_q.get(timeout=30)
                self.frames_in_pipeline -= 1
                self.pending_results[frame_id] = final_results
                print(f"[主進程] 最終收集 frame_{frame_id} 結果")
            except Exception as e:
                print(f"[主進程] 最終收集錯誤: {e}")
                break

        # 依序輸出所有結果
        while self.pending_results:
            if self.next_expected_frame in self.pending_results:
                frame_id = self.next_expected_frame
                final_results = self.pending_results[frame_id]
                del self.pending_results[frame_id]
                
                processed_frame = final_results.get('processed_frame')
                court_frame = final_results.get('court_frame')
                detected_players = final_results.get('detected_players', [])
                
                results.append((frame_id, processed_frame, court_frame, detected_players))
                self.next_expected_frame += 1
                print(f"[主進程] 最終輸出 frame_{frame_id}")
            else:
                # 如果有跳號，停止輸出
                break
                
        return results

    def reset_pipeline(self):
        """重置 pipeline 狀態"""
        self.frames_in_pipeline = 0
        self.pending_results.clear()
        self.next_expected_frame = 1
        self.frame_completion_events.clear()
        self.previous_frame_event = None
        self.frame_count = 0

    def get_pipeline_status(self):
        """取得 pipeline 狀態資訊"""
        return {
            'frames_in_pipeline': self.frames_in_pipeline,
            'pending_results_count': len(self.pending_results),
            'next_expected_frame': self.next_expected_frame,
            'max_parallel_frames': self.max_parallel_frames,
            'current_frame_count': self.frame_count
        }

    def get_player_info(self, team, number):
        """從DataManager獲取球員信息"""
        player_data, error = self.data_manager.get_player_data(team, number)
        if error:
            print(f"讀取球員資料錯誤: {error}")
            return pd.DataFrame(), []

        # 轉換成DataFrame格式
        if player_data:
            df = pd.DataFrame([player_data])
            return df, df.columns.tolist()
        return pd.DataFrame(), []

    def get_player_image(self, team, player_name):
        """從DataManager獲取球員圖片"""
        image_data, error = self.data_manager.get_player_image(team, player_name)
        if error:
            print(f"讀取球員圖片錯誤: {error}")
            return None
        return image_data

    def get_team_images(self, team):
        """從DataManager獲取球隊圖片"""
        team_image, error = self.data_manager.get_player_image(team, team)
        if error:
            print(f"讀取球隊圖片錯誤: {error}")
            return None

        if team_image:
            try:
                # 將 base64 圖片轉換為 OpenCV 格式
                image_data = team_image.split(',')[1]
                nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return image
            except Exception as e:
                print(f"轉換球隊圖片錯誤: {e}")
        return None

    def get_color_from_id(self, track_id):
        """根據 track_id 生成固定顏色"""
        try:
            if track_id is None:
                return (255, 255, 255)  # 返回白色作為預設顏色

            # 確保 track_id 在有效範圍內
            seed_value = abs(hash(str(track_id))) % (2 ** 32 - 1)
            np.random.seed(seed_value)
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            return color
        except Exception as e:
            print(f"生成顏色錯誤: {e}")
            return (255, 255, 255)  # 發生錯誤時返回白色

    def set_court_reference(self, court_image_path):
        """設置球場參考圖和參考點"""
        self.court_image = cv2.imread(court_image_path)
        self.image_points = np.array([
            [497, 347],  # top_left
            [497, 567],  # bottom_left
            [853, 347],  # top_right
            [853, 567]  # bottom_right
        ], dtype=np.float32)
        self.coordinate_mapper.set_reference(court_image_path, self.image_points)
        
        # 更新 Stage3 的球場設定 (需要重新啟動 Stage3 進程)
        # 暫時先不動態更新，等架構穩定後再優化

    def update_display_options(self, player=True, ball=True, team=True, number=True, trajectory=True):
        """更新顯示選項"""
        with self.lock:
            self.display_options['player'] = player
            self.display_options['ball'] = ball
            self.display_options['team'] = team
            self.display_options['number'] = number
            self.display_options['trajectory'] = trajectory

    def clean_old_associations(self):
        """清理過期的關聯"""
        current_frame = self.frame_count
        expired_ids = []
        for track_id, player_key in self.player_associations.items():
            if player_key in self.player_registry:
                last_seen = self.player_registry[player_key]['last_seen']
                if current_frame - last_seen > self.association_timeout:
                    expired_ids.append(track_id)

        for track_id in expired_ids:
            del self.player_associations[track_id]

    def set_score_callback(self, callback):
        """設置得分更新的回調函數"""
        self.score_callback = callback
        # 立即發送當前得分
        if callback:
            callback(self.scoring_analyzer.get_scores())

    def load_video(self, video_path):
        """載入視頻文件"""
        try:
            self.video_capture = cv2.VideoCapture(video_path)
            if not self.video_capture.isOpened():
                raise Exception("無法打開視頻文件")
            return True
        except Exception as e:
            print(f"載入視頻錯誤: {e}")
            return False

    def play(self):
        """開始播放"""
        self.is_playing = True

    def pause(self):
        """暫停播放"""
        self.is_playing = False

    def next_frame(self):
        """修改版：支援流水線處理"""
        if self.video_capture is None:
            return None, None, None

        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame = frame
            if len(self.frame_buffer) >= self.buffer_size:
                self.frame_buffer.pop(0)
            self.frame_buffer.append(frame)
            self.current_frame_index = len(self.frame_buffer) - 1
            
            result = self.process_frame(frame)
            
            # 如果當前幀沒有立即結果，嘗試取得 pending 結果
            if result[0] is None and result[1] is None and result[2] is None:
                pending_result = self._get_next_sequential_result()
                if pending_result:
                    frame_id, final_results = pending_result
                    processed_frame = final_results.get('processed_frame')
                    court_frame = final_results.get('court_frame') 
                    detected_players = final_results.get('detected_players', [])
                    return processed_frame, court_frame, detected_players
            
            return result
        else:
            # 視頻結束，輸出所有 pending 結果
            pending_results = self.get_pending_results()
            if pending_results:
                frame_id, processed_frame, court_frame, detected_players = pending_results[0]
                return processed_frame, court_frame, detected_players
            
        return None, None, None

    def prev_frame(self):
        """後退一幀"""
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.current_frame = self.frame_buffer[self.current_frame_index]
            return self.process_frame(self.current_frame)
        return None, None, None

    def get_current_frame(self):
        """獲取當前幀"""
        return self.current_frame

    def is_video_loaded(self):
        """檢查視頻是否已載入"""
        return self.video_capture is not None and self.video_capture.isOpened()

    def get_video_info(self):
        """獲取視頻信息"""
        if self.video_capture is None:
            return None

        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return {
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'current_frame_index': self.current_frame_index
        }

    def set_frame_position(self, frame_number):
        """設置播放位置"""
        if self.video_capture is None:
            return False

        # 重置 pipeline 狀態
        self.reset_pipeline()
        
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame = frame
            self.current_frame_index = frame_number
            return True
        return False

    def release(self):
        """釋放資源"""
        # ===== 關閉 Pipeline 多進程 =====
        try:
            # 發送結束訊號給所有階段
            self.stage1_input_q.put(None)
            self.stage1_to_stage2_q.put(None)
            self.stage2_to_stage3_q.put(None)
            
            # 等待進程結束
            for process in [self.stage1_process, self.stage2_process, self.stage3_process]:
                process.join(timeout=3)
                if process.is_alive():
                    process.terminate()
                    process.join()
                    
            print("Pipeline 多進程已關閉")
        except Exception as e:
            print(f"關閉 Pipeline 多進程時發生錯誤: {e}")

        # 重置 pipeline 狀態
        self.reset_pipeline()

        if self.video_capture is not None:
            self.video_capture.release()
        self.video_capture = None
        self.frame_buffer.clear()
        self.current_frame = None
        self.current_frame_index = 0
        self.is_playing = False
        self.coordinate_mapper.clear_trajectories()

    def update_player_stamina(self, player_key, distance):
        """更新球員體力值"""
        if player_key in self.player_states:
            stamina = max(0, (self.max_distance - distance) / self.max_distance * 100)
            self.player_states[player_key]['stamina'] = stamina
            self.player_states[player_key]['total_distance'] = distance

    def get_stamina_color(self, stamina):
        """根據體力值回傳對應的顏色"""
        if stamina <= 25:
            return (0, 0, 255)  # 紅色
        elif stamina <= 50:
            return (0, 255, 255)  # 黃色
        else:
            return (0, 255, 0)  # 綠色

    def get_performance_stats(self):
        """獲取性能統計數據"""
        stats = {
            "total_frames": self.total_frames_processed,
            "total_time": self.total_processing_time,
            "avg_time_per_frame": self.total_processing_time / max(1, self.total_frames_processed),
            "fps": self.total_frames_processed / max(0.001, self.total_processing_time)
        }

        # 計算每個步驟的平均時間
        for step, times in self.performance_metrics.items():
            if times:
                stats[f"{step}_avg"] = sum(times) / len(times)
                stats[f"{step}_max"] = max(times)
                stats[f"{step}_min"] = min(times)

        return stats

    def print_performance_report(self):
        """打印性能報告"""
        stats = self.get_performance_stats()
        print("\n=== Performance Report ===")
        print(f"Total Frames Processed: {stats['total_frames']}")
        print(f"Total Processing Time: {stats['total_time']:.2f} seconds")
        print(f"Average Time per Frame: {stats['avg_time_per_frame'] * 1000:.2f} ms")
        print(f"Average FPS: {stats['fps']:.2f}")
        print("\nBreakdown by Steps:")

        for key, value in stats.items():
            if key.endswith('_avg'):
                step = key.replace('_avg', '')
                print(f"\n{step}:")
                print(f"  Average: {value * 1000:.2f} ms")
                print(f"  Maximum: {stats[f'{step}_max'] * 1000:.2f} ms")
                print(f"  Minimum: {stats[f'{step}_min'] * 1000:.2f} ms")

    def reset_performance_metrics(self):
        """重置性能指標"""
        self.performance_metrics.clear()
        self.total_frames_processed = 0
        self.total_processing_time = 0

    def reset_scores(self):
        """重置所有得分"""
        self.scoring_analyzer.reset_scores()
        # 重置分數時也要通知更新
        if self.score_callback:
            self.score_callback(self.scoring_analyzer.get_scores())

    def get_current_scores(self):
        """獲取當前比分"""
        return self.scoring_analyzer.get_scores()

    def save_frame(self, frame, filename):
        """儲存當前畫面"""
        try:
            cv2.imwrite(filename, frame)
            return True
        except Exception as e:
            print(f"儲存畫面錯誤: {e}")
            return False

    def get_player_statistics(self):
        """獲取所有球員的統計資料"""
        player_stats = {}
        for player_key, registry_data in self.player_registry.items():
            stats = {}
            if player_key in self.player_states:
                stats = self.player_states[player_key].copy()
            track_id = registry_data.get('track_id', None)
            if track_id in self.coordinate_mapper.tracking_data:
                stats['total_distance'] = self.coordinate_mapper.tracking_data[track_id]['total_distance']

            stats['last_seen'] = registry_data.get('last_seen', 0)
            stats['team'] = registry_data.get('team', '')
            stats['number'] = registry_data.get('number', '')
            stats['stamina_color'] = self.get_stamina_color(stats.get('stamina', 100))
            player_stats[player_key] = stats

        return player_stats

    def clear_trajectories(self):
        """清除所有軌跡"""
        self.coordinate_mapper.clear_trajectories()

    def remove_trajectory(self, track_id):
        """移除指定 track_id 的軌跡"""
        self.coordinate_mapper.remove_trajectory(track_id)

    def toggle_trajectory_display(self, show_trajectory=True):
        """切換軌跡顯示"""
        with self.lock:
            self.display_options['trajectory'] = show_trajectory
            if not show_trajectory:
                self.clear_trajectories()

    def reset_player_states(self):
        """重置所有球員狀態"""
        for player_key in self.player_states:
            self.player_states[player_key] = {
                'score': 0,
                'stamina': 100,
                'total_distance': 0
            }
        self.clear_trajectories()

    def get_player_state(self, player_key):
        """獲取特定球員的狀態"""
        return self.player_states.get(player_key, {
            'score': 0,
            'stamina': 100,
            'total_distance': 0
        })

    def update_player_state(self, player_key, state_updates):
        """更新球員狀態"""
        if player_key in self.player_states:
            self.player_states[player_key].update(state_updates)

    def _create_mock_frame(self, text):
        """建立模擬的輸出畫面"""
        frame = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

    def _handle_scoring(self, scoring_info):
        """
        處理得分事件
        Args:
            scoring_info: dict containing scoring information
        """
        track_id = scoring_info.get('track_id')
        team = scoring_info.get('team')

        if track_id and track_id in self.player_associations:
            pkey = self.player_associations[track_id]
            if pkey in self.player_states:
                # 更新個人得分
                self.player_states[pkey]['personal_scores'] += 1
                print(f"進球！得分方：{team} #{self.player_registry[pkey]['number']}, "
                      f"個人第 {self.player_states[pkey]['personal_scores']} 球")
        else:
            print(f"進球！得分方：{team} (未識別球員)")

        # 更新總分顯示
        if self.score_callback:
            self.score_callback(self.scoring_analyzer.get_scores())

    def _process_court_mapping(self, court_results, player_boxes, court_frame):
        """處理球場映射"""
        for result in court_results:
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.xy
                classes = result.boxes.cls.cpu().numpy()

                for mask, cls_id in zip(masks, classes):
                    if len(mask) < 3:
                        continue

                    if cls_id == 1:  # RA area
                        geometric_points = extract_geometric_points(mask)
                        self.current_video_points = np.array([
                            geometric_points["RA"]["points"][0],
                            geometric_points["RA"]["points"][1],
                            geometric_points["RA"]["points"][2],
                            geometric_points["RA"]["points"][3]
                        ], dtype=np.float32)

                        if (self.current_video_points is not None and
                                self.image_points is not None and
                                court_frame is not None):
                            try:
                                H, _ = cv2.findHomography(self.current_video_points, self.image_points)
                                self._process_player_trajectories(player_boxes, H, court_frame)
                            except Exception as e:
                                print(f"球場映射錯誤: {e}")

    def _process_player_trajectories(self, player_boxes, homography_matrix, court_frame):
        """處理球員軌跡"""
        for player_id, x1, y1, x2, y2 in player_boxes:
            player_point = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32).reshape(1, 1, 2)
            try:
                projected_point = cv2.perspectiveTransform(player_point, homography_matrix)
                x, y = int(projected_point[0][0][0]), int(projected_point[0][0][1])
                total_distance = self.coordinate_mapper.track_movement(player_id, [x, y])

                if player_id in self.coordinate_mapper.tracking_data:
                    positions = self.coordinate_mapper.tracking_data[player_id]['positions']
                    color = self.coordinate_mapper.tracking_data[player_id]['color']

                    if len(positions) > 1:
                        for i in range(len(positions) - 1):
                            pt1 = tuple(map(int, positions[i]))
                            pt2 = tuple(map(int, positions[i + 1]))
                            cv2.line(court_frame, pt1, pt2, color, 2)

                    cv2.circle(court_frame, (x, y), 5, color, -1)
                    cv2.putText(court_frame,
                                f"ID: {player_id} Dist: {total_distance:.1f}m",
                                (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                1)

                    if player_id in self.player_associations:
                        pkey = self.player_associations[player_id]
                        if pkey in self.player_states:
                            stamina = self.player_states[pkey]['stamina']
                            stamina_color = self.get_stamina_color(stamina)
            except Exception as e:
                print(f"軌跡處理錯誤: {e}")

    def _process_number_matches(self, filtered_matches):
        """處理號碼匹配結果"""
        for match in filtered_matches:
            track_id = match['player_id']
            team = match['team']
            number = match['number']
            player_key = f"{team}_{number}"

            if player_key not in self.player_registry:
                self.player_registry[player_key] = {
                    "team": team,
                    "number": number,
                    "track_id": track_id,
                    "last_seen": self.frame_count
                }
                if player_key not in self.player_states:
                    self.player_states[player_key] = {
                        'score': 0,
                        'stamina': 100,
                        'total_distance': 0
                    }
            else:
                self.player_registry[player_key]['track_id'] = track_id
                self.player_registry[player_key]['last_seen'] = self.frame_count

            old_tid = None
            for tid, pkey in self.player_associations.items():
                if pkey == player_key and tid != track_id:
                    old_tid = tid
                    break
            if old_tid is not None:
                del self.player_associations[old_tid]

            self.player_associations[track_id] = player_key

    def _update_player_states(self, player_boxes):
        """
        更新球員狀態
        Args:
            player_boxes: List of (track_id, x1, y1, x2, y2)
        Returns:
            detected_players: List of player information
        """
        detected_players = []
        for (t_id, x1, y1, x2, y2) in player_boxes:
            if t_id in self.player_associations:
                pkey = self.player_associations[t_id]
                if pkey in self.player_registry:
                    team = self.player_registry[pkey]['team']
                    number = self.player_registry[pkey]['number']
                    player_info, _ = self.get_player_info(team, int(number))

                    # 確保 player_states 有該球員的記錄
                    if pkey not in self.player_states:
                        self.player_states[pkey] = {
                            'score': 0,
                            'stamina': 100,
                            'total_distance': 0.0,
                            'is_last_touch': False,
                            'personal_scores': 0
                        }

                    extra_info = self.player_states[pkey]

                    # 更新移動距離和體力值
                    track_data = {}
                    if t_id in self.coordinate_mapper.tracking_data:
                        total_distance = self.coordinate_mapper.tracking_data[t_id]['total_distance']
                        track_data['total_distance'] = total_distance
                        self.update_player_stamina(pkey, total_distance)

                    # 組合所有資訊
                    new_player = {
                        'team': team,
                        'number': int(number),
                        'info': player_info,
                        'score': extra_info.get('score', 0),
                        'stamina': extra_info.get('stamina', 100),
                        'total_distance': track_data.get('total_distance', 0.0),
                        'stamina_color': self.get_stamina_color(extra_info.get('stamina', 100)),
                        'is_last_touch': extra_info.get('is_last_touch', False),
                        'personal_scores': extra_info.get('personal_scores', 0)
                    }

                    if not any(p['team'] == new_player['team'] and
                               p['number'] == new_player['number']
                               for p in detected_players):
                        detected_players.append(new_player)

        return detected_players