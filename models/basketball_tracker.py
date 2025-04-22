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

class BasketballTracker:
    def __init__(self, player_model_path, court_model_path, data_folder):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用設備: {self.device}")

        # 載入模型
        self.player_model = YOLO(player_model_path).to(self.device)
        self.court_model = YOLO(court_model_path).to(self.device)
        print(f"模型載入完成")

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
        """前進一幀"""
        if self.video_capture is None:
            return None, None, None

        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame = frame
            if len(self.frame_buffer) >= self.buffer_size:
                self.frame_buffer.pop(0)
            self.frame_buffer.append(frame)
            self.current_frame_index = len(self.frame_buffer) - 1
            return self.process_frame(frame)
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

        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.video_capture.read()
        if ret:
            self.current_frame = frame
            self.current_frame_index = frame_number
            return True
        return False

    def release(self):
        """釋放資源"""
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

    def process_frame(self, frame):
        """處理單一幀"""
        if frame is None:
            return None, None, None

        frame_start_time = time.time()
        self.frame_count += 1
        self.total_frames_processed += 1

        # # 添加幀計數顯示
        # frame_info = f"Current Frame: {self.frame_count}"
        # cv2.putText(frame, frame_info, (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #
        # # 添加處理延遲顯示
        # if hasattr(self, 'last_frame_time'):
        #     processing_time = time.time() - self.last_frame_time
        #     fps = 1.0 / processing_time if processing_time > 0 else 0
        #     delay_info = f"FPS: {fps:.2f}"
        #     cv2.putText(frame, delay_info, (10, 60),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        self.last_frame_time = time.time()

        processed_frame = frame.copy()
        court_frame = self.court_image.copy() if self.court_image is not None else None
        detected_players = []

        with self.lock:
            display_options = self.display_options.copy()

        basketball_data = None
        hoop_data = None
        player_positions = {}

        # 1) 處理球場檢測
        start_time = time.time()
        court_results = self.court_model.track(source=frame, conf=0.25, persist=True, tracker="bytetrack.yaml")
        self.performance_metrics['court_detection'].append(time.time() - start_time)

        # 2) 處理球員、球、籃框等偵測
        start_time = time.time()
        player_results = self.player_model.track(source=frame, conf=0.3, persist=True, tracker="bytetrack.yaml")
        self.performance_metrics['player_detection'].append(time.time() - start_time)

        player_boxes = []
        team_boxes = []
        number_boxes = []

        # 3) 物件分析與追蹤
        start_time = time.time()
        if player_results:
            for result in player_results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0].item())
                    track_id = int(box.id[0].item()) if box.id is not None else -1

                    # 修改類別名稱映射
                    class_display = {
                        0: "3-s",  # 3-s -> 3fm
                        1: "ball",
                        2: "biv",  # biv -> b
                        3: "hoop",
                        4: "Num",  # number -> N
                        5: "Player"  # player -> P
                    }

                    if cls == 1:  # ball
                        center = ((x1 + x2) / 2, (y1 + y2) / 2)
                        left = (x1, y1)
                        right = (x2, y2)
                        basketball_data = (center, left, right)
                        if display_options['ball']:
                            cv2.rectangle(processed_frame,
                                          (int(x1), int(y1)),
                                          (int(x2), int(y2)),
                                          (0, 255, 0), 2)

                    elif cls == 3:  # hoop
                        center = ((x1 + x2) / 2, y1)
                        left = (x1, y1)
                        right = (x2, y2)
                        hoop_data = (center, left, right)

                    elif cls == 5:  # player
                        player_boxes.append((track_id, x1, y1, x2, y2))

                    # 先檢查是否有關聯的球員資訊
                    team = None
                    if track_id in self.player_associations:
                        pkey = self.player_associations[track_id]
                        if pkey in self.player_registry:
                            self.player_registry[pkey]['last_seen'] = self.frame_count
                            team = self.player_registry[pkey]['team']
                            player_positions[track_id] = (x1, y1, x2, y2, team)

                    elif cls == 0 or cls == 2:  # team boxes
                        team_name = self.team_mapping.get(cls, 'unknown')
                        team_boxes.append((x1, y1, x2, y2, team_name))

                    elif cls == 4:  # number
                        number_boxes.append(box)

                    # 3) 繪製標記 (若符合對應的顯示選項)
                    should_draw = ((cls == 5 and display_options['player']) or
                                   ((cls == 0 or cls == 2) and display_options['team']) or
                                   (cls == 4 and display_options['number']))

                    if should_draw:
                        color = self.get_color_from_id(track_id)
                        cv2.rectangle(processed_frame,
                                      (int(x1), int(y1)),
                                      (int(x2), int(y2)),
                                      color, 2)

                        # 修改後的標籤邏輯：只顯示簡化的類別名稱
                        label = class_display[cls]
                        if track_id in self.player_associations:
                            player_key = self.player_associations[track_id]
                            if player_key in self.player_registry:
                                assoc_data = self.player_registry[player_key]
                                label = f"{class_display[cls]} {assoc_data['team']}#{assoc_data['number']}"

                        cv2.putText(processed_frame,
                                    label,
                                    (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    2)

        self.performance_metrics['object_tracking'].append(time.time() - start_time)

        # 4) 進籃分析
        start_time = time.time()
        if basketball_data and hoop_data:
            self.scoring_analyzer.update_positions(basketball_data, hoop_data)

        if not basketball_data:
            predicted_pos = self.scoring_analyzer.predict_basketball_position()
            if predicted_pos:
                basketball_data = (predicted_pos, None, None)

        if basketball_data and basketball_data[0]:
            self.scoring_analyzer.update_last_touch(basketball_data[0], player_positions)

        # 檢查得分並通過回調更新
        if basketball_data and hoop_data:
            scored, team = self.scoring_analyzer.check_scoring(basketball_data[0], hoop_data[0])
            if scored:
                print(f"進球！得分方：{team}")
                if self.score_callback:
                    scores = self.scoring_analyzer.get_scores()
                    self.score_callback(scores)
        self.performance_metrics['scoring_analysis'].append(time.time() - start_time)

        # 5) 球場分析與映射
        start_time = time.time()
        if court_results and display_options['trajectory']:
            self._process_court_mapping(court_results, player_boxes, court_frame)
        self.performance_metrics['court_mapping'].append(time.time() - start_time)

        # 6) 號碼識別
        start_time = time.time()
        if number_boxes:
            matches = self.number_recognizer.match_numbers_to_players(
                frame, player_boxes, number_boxes, team_boxes)
            filtered_matches = self.number_recognizer.filter_matches(matches)
            self._process_number_matches(filtered_matches)
        self.performance_metrics['number_recognition'].append(time.time() - start_time)

        # 7) 更新球員狀態
        start_time = time.time()
        detected_players = self._update_player_states(player_boxes)
        self.performance_metrics['player_state_update'].append(time.time() - start_time)

        # 8) 最終處理
        start_time = time.time()
        if processed_frame is not None:
            processed_frame = cv2.resize(processed_frame, (self.output_width, self.output_height))
        if court_frame is not None:
            court_frame = cv2.resize(court_frame, (self.output_width, self.output_height))
        self.performance_metrics['final_processing'].append(time.time() - start_time)

        # 更新總處理時間
        self.total_processing_time += time.time() - frame_start_time

        return processed_frame, court_frame, detected_players

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
            11