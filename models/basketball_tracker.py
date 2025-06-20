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
from collections import defaultdict, deque
import multiprocessing as mp
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

# ===== FPS 監控類別 =====
class FPSMonitor:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.time()
        self.current_fps = 0.0
        self.avg_fps = 0.0
        
    def update(self):
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        
        if frame_time > 0:
            self.current_fps = 1.0 / frame_time
            self.frame_times.append(frame_time)
            
            if len(self.frame_times) > 0:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                self.avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_fps(self):
        return self.current_fps
    
    def get_avg_fps(self):
        return self.avg_fps

def draw_fps_on_frame(frame, fps_monitor):
    """在影格上繪製 FPS 資訊"""
    if frame is None:
        return frame
    
    current_fps = fps_monitor.get_fps()
    avg_fps = fps_monitor.get_avg_fps()
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 255, 0)
    thickness = 2
    
    fps_text = f"FPS: {current_fps:.1f}"
    avg_fps_text = f"Avg: {avg_fps:.1f}"
    
    height, width = frame.shape[:2]
    
    # 繪製半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (width - 150, 10), (width - 10, 80), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # 繪製 FPS 文字
    cv2.putText(frame, fps_text, (width - 140, 35), font, font_scale, color, thickness)
    cv2.putText(frame, avg_fps_text, (width - 140, 60), font, font_scale, color, thickness)
    
    return frame

# ===== ThreadPool 模型管理函數（移到最外層）=====
# 全局的 thread_local 存儲
_thread_local = threading.local()

def get_player_model(player_model_path, device):
    if not hasattr(_thread_local, 'player_model'):
        print(f"[Pipeline階段1] 載入球員模型到線程 {threading.current_thread().ident}")
        _thread_local.player_model = YOLO(player_model_path).to(device)
    return _thread_local.player_model
    
def get_court_model(court_model_path, device):
    if not hasattr(_thread_local, 'court_model'):
        print(f"[Pipeline階段1] 載入球場模型到線程 {threading.current_thread().ident}")
        _thread_local.court_model = YOLO(court_model_path).to(device)
    return _thread_local.court_model

def process_player_model(frame, player_model_path, device):
    model = get_player_model(player_model_path, device)
    with torch.no_grad():
        return model.track(source=frame, conf=0.3, persist=True, tracker="bytetrack.yaml")

def process_court_model(frame, court_model_path, device):
    model = get_court_model(court_model_path, device)
    with torch.no_grad():
        return model.track(source=frame, conf=0.25, persist=True, tracker="bytetrack.yaml")

def ocr_pool_worker_process(args):
    """OCR 池工作函式 - 處理單個 OCR 任務"""
    frame_id, original_frame, number_boxes, player_boxes, team_boxes = args
    
    # 檢查是否已在這個進程中載入過 OCR 模型
    if not hasattr(ocr_pool_worker_process, 'ocr_model'):
        print(f"[OCR Pool] 進程 {os.getpid()} 載入 OCR 模型")
        try:
            from paddleocr import PaddleOCR
            from .number_recognizer import NumberRecognizer
            
            ocr_pool_worker_process.ocr_model = PaddleOCR(
                lang='en',
                rec_model_dir="./path_to/en_PP-OCRv3_rec_infer", 
                cls_model_dir="./path_to/ch_ppocr_mobile_v2.0_cls_infer",
                use_angle_cls=False,
                use_gpu=True,
                show_log=False
            )
            
            ocr_pool_worker_process.number_recognizer = NumberRecognizer(ocr_pool_worker_process.ocr_model)
            print(f"[OCR Pool] 進程 {os.getpid()} OCR 模型載入完成")
            
        except Exception as e:
            print(f"[OCR Pool] 進程 {os.getpid()} OCR 初始化失敗: {e}")
            ocr_pool_worker_process.ocr_model = None
            ocr_pool_worker_process.number_recognizer = None
    
    # 使用載入的模型進行處理
    if ocr_pool_worker_process.number_recognizer is None:
        print(f"[OCR Pool] 進程 {os.getpid()} OCR 模型未載入，跳過處理")
        return frame_id, []
    
    start_time = time.time()
    ocr_matches = []
    
    try:
        if number_boxes and original_frame is not None:
            print(f"[OCR Pool] 進程 {os.getpid()} 處理 frame_{frame_id}，{len(number_boxes)} 個號碼區域")
            
            # ===== 格式轉換（與原本邏輯相同）=====
            import torch
            
            class MockBox:
                def __init__(self, data):
                    self.xyxy = torch.tensor([[data['x1'], data['y1'], data['x2'], data['y2']]])
                    self.cls = torch.tensor([data['cls']])
                    self.id = torch.tensor([data['track_id']]) if data['track_id'] != -1 else None
            
            yolo_number_boxes = [MockBox(num_box) for num_box in number_boxes]
            player_tuples = [(p['track_id'], p['x1'], p['y1'], p['x2'], p['y2']) for p in player_boxes]
            team_tuples = [(t['x1'], t['y1'], t['x2'], t['y2'], t['team']) for t in team_boxes]
            
            # 使用 NumberRecognizer 處理
            matches = ocr_pool_worker_process.number_recognizer.match_numbers_to_players(
                original_frame, player_tuples, yolo_number_boxes, team_tuples
            )
            filtered_matches = ocr_pool_worker_process.number_recognizer.filter_matches(matches)
            
            # 轉換結果格式
            for match in filtered_matches:
                ocr_matches.append({
                    'player_id': match.get('player_id'),
                    'team': match.get('team'),
                    'number': match.get('number'),
                    'confidence': match.get('confidence', 1.0),
                    'ocr_text': str(match.get('number', '')),
                    'distance': match.get('distance', 0.0)
                })
                
                print(f"[OCR Pool] 進程 {os.getpid()} 識別: #{match.get('number')} -> 球員ID {match.get('player_id')} 隊伍 {match.get('team')}")
        
        processing_time = time.time() - start_time
        print(f"[OCR Pool] 進程 {os.getpid()} frame_{frame_id} 完成，耗時: {processing_time:.3f}s，識別 {len(ocr_matches)} 個號碼")
        
        return frame_id, ocr_matches
        
    except Exception as e:
        print(f"[OCR Pool] 進程 {os.getpid()} 處理 frame_{frame_id} 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return frame_id, []


# ===== Pipeline 階段處理 Workers =====
def pipeline_stage1_worker(input_q, output_q, player_model_path, court_model_path, device):
    """Pipeline 階段1：模型推理 + 物件分類"""
    print("[Pipeline階段1] 模型推理 + 物件分類 Worker 啟動...")
    
    # 類別映射
    team_mapping = {0: '3-s', 2: 'biv'}
    
    # 使用 ThreadPoolExecutor 進行並行推理
    with ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            task = input_q.get()
            if task is None:
                print("[Pipeline階段1] 收到結束訊號")
                output_q.put(None)
                break
            
            frame_id, frame = task
            start_time = time.time()
            
            # 並行執行兩個模型的推理
            player_future = executor.submit(process_player_model, frame, player_model_path, device)
            court_future = executor.submit(process_court_model, frame, court_model_path, device)
            
            # 等待兩個推理結果
            player_results = player_future.result()
            court_results = court_future.result()
            
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

# ===== OCR Worker 進程函數 =====
def ocr_worker_process(worker_id, task_queue, result_queue):
    """OCR 工作進程 - 預載入模型並持續處理任務"""
    print(f"[OCR Worker {worker_id}] 進程啟動，PID: {os.getpid()}")
    
    # ===== 預載入 OCR 模型 =====
    try:
        from paddleocr import PaddleOCR
        from .number_recognizer import NumberRecognizer
        
        print(f"[OCR Worker {worker_id}] 開始載入 OCR 模型...")
        
        ocr_model = PaddleOCR(
            lang='en',
            rec_model_dir="./path_to/en_PP-OCRv3_rec_infer", 
            cls_model_dir="./path_to/ch_ppocr_mobile_v2.0_cls_infer",
            use_angle_cls=False,
            use_gpu=True,
            show_log=False
        )
        
        number_recognizer = NumberRecognizer(ocr_model)
        print(f"[OCR Worker {worker_id}] OCR 模型載入完成")
        
    except Exception as e:
        print(f"[OCR Worker {worker_id}] OCR 初始化失敗: {e}")
        return
    
    print(f"[OCR Worker {worker_id}] 準備就緒，等待任務...")
    
    # ===== 持續處理任務 =====
    while True:
        try:
            # 從任務佇列取得工作
            task = task_queue.get()
            
            # 檢查結束訊號
            if task is None:
                print(f"[OCR Worker {worker_id}] 收到結束訊號")
                break
            
            frame_id, original_frame, number_boxes, player_boxes, team_boxes = task
            start_time = time.time()
            ocr_matches = []
            
            try:
                if number_boxes and original_frame is not None:
                    print(f"[OCR Worker {worker_id}] 處理 frame_{frame_id}，{len(number_boxes)} 個號碼區域")
                    
                    # ===== 格式轉換 =====
                    import torch
                    
                    class MockBox:
                        def __init__(self, data):
                            self.xyxy = torch.tensor([[data['x1'], data['y1'], data['x2'], data['y2']]])
                            self.cls = torch.tensor([data['cls']])
                            self.id = torch.tensor([data['track_id']]) if data['track_id'] != -1 else None
                    
                    yolo_number_boxes = [MockBox(num_box) for num_box in number_boxes]
                    player_tuples = [(p['track_id'], p['x1'], p['y1'], p['x2'], p['y2']) for p in player_boxes]
                    team_tuples = [(t['x1'], t['y1'], t['x2'], t['y2'], t['team']) for t in team_boxes]
                    
                    # 使用預載入的 NumberRecognizer 處理
                    matches = number_recognizer.match_numbers_to_players(
                        original_frame, player_tuples, yolo_number_boxes, team_tuples
                    )
                    filtered_matches = number_recognizer.filter_matches(matches)
                    
                    # 轉換結果格式
                    for match in filtered_matches:
                        ocr_matches.append({
                            'player_id': match.get('player_id'),
                            'team': match.get('team'),
                            'number': match.get('number'),
                            'confidence': match.get('confidence', 1.0),
                            'ocr_text': str(match.get('number', '')),
                            'distance': match.get('distance', 0.0)
                        })
                        
                        print(f"[OCR Worker {worker_id}] 識別: #{match.get('number')} -> 球員ID {match.get('player_id')} 隊伍 {match.get('team')}")
                
                processing_time = time.time() - start_time
                print(f"[OCR Worker {worker_id}] frame_{frame_id} 完成，耗時: {processing_time:.3f}s，識別 {len(ocr_matches)} 個號碼")
                
                # 回傳結果
                result_queue.put((frame_id, ocr_matches))
                
            except Exception as e:
                print(f"[OCR Worker {worker_id}] 處理 frame_{frame_id} 錯誤: {e}")
                import traceback
                traceback.print_exc()
                result_queue.put((frame_id, []))
                
        except Exception as e:
            print(f"[OCR Worker {worker_id}] 任務處理錯誤: {e}")
            break
    
    print(f"[OCR Worker {worker_id}] 進程結束")

def pipeline_stage2_worker(input_q, output_q):
    """修改版 Pipeline 階段2：使用 multiprocessing + Queue 的 OCR 處理"""
    print("[Pipeline階段2] OCR multiprocessing + Queue Worker 啟動...")
    
    import multiprocessing as mp
    import time
    from collections import deque
    
    # ===== 建立 OCR 工作進程池 =====
    max_workers = min(2, mp.cpu_count())
    print(f"[Pipeline階段2] 啟動 {max_workers} 個 OCR 工作進程...")
    
    # 建立進程間通訊的 Queue
    ctx = mp.get_context("spawn")
    ocr_task_queue = ctx.Queue()
    ocr_result_queue = ctx.Queue()
    
    # 啟動 OCR 工作進程
    ocr_workers = []
    for i in range(max_workers):
        worker = ctx.Process(
            target=ocr_worker_process,
            args=(i, ocr_task_queue, ocr_result_queue)
        )
        worker.start()
        ocr_workers.append(worker)
        print(f"[Pipeline階段2] OCR Worker {i} 已啟動")
    
    # 等待所有 OCR 進程完成初始化
    import time
    print("[Pipeline階段2] 等待 OCR 工作進程完成模型載入...")
    time.sleep(3)  # 給予足夠時間載入模型
    print("[Pipeline階段2] OCR 工作進程已準備就緒")
    
    # ===== 任務處理邏輯 =====
    pending_tasks = {}  # frame_id -> task_data
    completed_results = {}  # frame_id -> ocr_results
    task_counter = 0
    
    try:
        while True:
            # 1. 檢查是否有新的輸入任務
            try:
                task = input_q.get_nowait()
                if task is None:
                    print("[Pipeline階段2] 收到結束訊號")
                    break
                
                frame_id, stage1_results = task
                
                # 提取 OCR 需要的資料
                number_boxes = stage1_results.get('number_boxes', [])
                player_boxes = stage1_results.get('player_boxes', [])
                team_boxes = stage1_results.get('team_boxes', [])
                original_frame = stage1_results.get('original_frame')
                
                if number_boxes and original_frame is not None:
                    # 送給 OCR 工作進程
                    ocr_task_queue.put((frame_id, original_frame, number_boxes, player_boxes, team_boxes))
                    pending_tasks[frame_id] = stage1_results
                    task_counter += 1
                    print(f"[Pipeline階段2] 送出 frame_{frame_id} 給 OCR 工作進程，待處理: {len(pending_tasks)}")
                else:
                    # 沒有號碼需要處理，直接傳遞
                    results = stage1_results.copy()
                    results.update({
                        'ocr_matches': [],
                        'ocr_time': 0,
                        'stage2_status': 'no_numbers'
                    })
                    output_q.put((frame_id, results))
                    print(f"[Pipeline階段2] frame_{frame_id} 無號碼，直接傳遞")
                    
            except:
                pass  # 沒有新任務
            
            # 2. 檢查 OCR 結果
            try:
                result_frame_id, ocr_matches = ocr_result_queue.get_nowait()
                completed_results[result_frame_id] = ocr_matches
                print(f"[Pipeline階段2] 收到 frame_{result_frame_id} OCR 結果，識別 {len(ocr_matches)} 個號碼")
            except:
                pass  # 沒有新結果
            
            # 3. 處理完成的結果
            completed_frames = []
            for frame_id in list(pending_tasks.keys()):
                if frame_id in completed_results:
                    stage1_results = pending_tasks[frame_id]
                    ocr_matches = completed_results[frame_id]
                    
                    # 組合最終結果
                    results = stage1_results.copy()
                    results.update({
                        'ocr_matches': ocr_matches,
                        'ocr_time': 0,
                        'stage2_status': 'ocr_completed'
                    })
                    
                    output_q.put((frame_id, results))
                    print(f"[Pipeline階段2] 完成 frame_{frame_id}，識別 {len(ocr_matches)} 個號碼")
                    
                    completed_frames.append(frame_id)
            
            # 清理已完成的任務
            for frame_id in completed_frames:
                del pending_tasks[frame_id]
                del completed_results[frame_id]
            
            # 避免 CPU 過度使用
            if not pending_tasks and task_counter == 0:
                time.sleep(0.001)
        
        # ===== 處理剩餘任務 =====
        print(f"[Pipeline階段2] 等待剩餘 {len(pending_tasks)} 個任務完成...")
        while pending_tasks:
            try:
                result_frame_id, ocr_matches = ocr_result_queue.get(timeout=10)
                
                if result_frame_id in pending_tasks:
                    stage1_results = pending_tasks[result_frame_id]
                    
                    results = stage1_results.copy()
                    results.update({
                        'ocr_matches': ocr_matches,
                        'ocr_time': 0,
                        'stage2_status': 'ocr_completed'
                    })
                    
                    output_q.put((result_frame_id, results))
                    print(f"[Pipeline階段2] 最終完成 frame_{result_frame_id}")
                    
                    del pending_tasks[result_frame_id]
                    
            except Exception as e:
                print(f"[Pipeline階段2] 等待剩餘任務錯誤: {e}")
                break
        
        # ===== 關閉 OCR 工作進程 =====
        print("[Pipeline階段2] 關閉 OCR 工作進程...")
        for i in range(max_workers):
            ocr_task_queue.put(None)  # 發送結束訊號
        
        for worker in ocr_workers:
            worker.join(timeout=5)
            if worker.is_alive():
                print(f"[Pipeline階段2] 強制終止 OCR Worker")
                worker.terminate()
        
        print("[Pipeline階段2] OCR 工作進程已關閉")
        output_q.put(None)
        
    except Exception as e:
        print(f"[Pipeline階段2] 錯誤: {e}")
        import traceback
        traceback.print_exc()
        
        # 確保關閉所有進程
        for worker in ocr_workers:
            if worker.is_alive():
                worker.terminate()
        
        output_q.put(None)

def pipeline_stage3_worker(input_q, output_q, frame_completion_events, config):
    """Pipeline 階段3：進籃分析 + 球場映射 + 狀態更新 + 畫面繪製 (必須按順序)"""
    print("[Pipeline階段3] 進籃分析 + 球場映射 + 狀態更新 Worker 啟動...")
    
    # 初始化分析器 (在 Worker 內部建立)
    from .scoring_analyzer import ScoringAnalyzer
    from .coordinate_mapper import CoordinateMapper
    from .court_analyzer import extract_geometric_points
    from utils.data_manager import DataManager
    import cv2
    import numpy as np
    import pandas as pd
    
    scoring_analyzer = ScoringAnalyzer()
    coordinate_mapper = CoordinateMapper()
    data_manager = DataManager(config.get('data_folder', 'big3_data'))
    
    # 從 config 取得設定
    output_width = config['output_width']
    output_height = config['output_height']
    class_names = config['class_names']
    team_mapping = config['team_mapping']
    
    # class_display 映射
    class_display = {
        0: "3-s",    # 3-s -> 3fm
        1: "ball",
        2: "biv",    # biv -> b
        3: "hoop",
        4: "Num",    # number -> N
        5: "Player"  # player -> P
    }
    
    # 球場相關設定
    court_image = None
    image_points = None
    if config.get('court_image_path'):
        court_image = cv2.imread(config['court_image_path'])
        image_points = config.get('image_points')
        if image_points is not None:
            image_points = np.array(image_points, dtype=np.float32)
            coordinate_mapper.set_reference(config['court_image_path'], image_points)
        print(f"[Pipeline階段3] 成功載入球場圖片: {config['court_image_path']}")
    else:
        print(f"[Pipeline階段3] 警告：球場圖片未設定")
    
    # 顏色生成函數
    def get_color_from_id(track_id):
        if track_id is None:
            return (255, 255, 255)
        seed_value = abs(hash(str(track_id))) % (2 ** 32 - 1)
        np.random.seed(seed_value)
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        return color
    
    # 初始化 FPS 監控器
    fps_monitor = FPSMonitor(window_size=30)
    
    # ===== Worker 內部的球員狀態管理（參考正常版本）=====
    display_options = {'player': True, 'ball': True, 'team': True, 'number': True, 'trajectory': True}
    player_associations = {}  # track_id -> player_key
    player_registry = {}     # player_key -> {team, number, track_id, last_seen}
    player_states = {}       # player_key -> {score, stamina, total_distance, etc.}
    frame_count = 0
    association_timeout = 30
    max_distance = 100
    
    # ===== 參考正常版本的函數 =====
    def update_player_stamina(player_key, distance):
        """更新球員體力值（參考正常版本）"""
        if player_key in player_states:
            stamina = max(0, (max_distance - distance) / max_distance * 100)
            player_states[player_key]['stamina'] = stamina
            player_states[player_key]['total_distance'] = distance

    def get_stamina_color(stamina):
        """根據體力值回傳對應的顏色（參考正常版本）"""
        if stamina <= 25:
            return (0, 0, 255)  # 紅色
        elif stamina <= 50:
            return (0, 255, 255)  # 黃色
        else:
            return (0, 255, 0)  # 綠色

    def clean_old_associations():
        """清理過期的關聯（完全參考正常版本）"""
        current_frame = frame_count
        expired_ids = []
        for track_id, player_key in player_associations.items():
            if player_key in player_registry:
                last_seen = player_registry[player_key]['last_seen']
                if current_frame - last_seen > association_timeout:
                    expired_ids.append(track_id)

        for track_id in expired_ids:
            del player_associations[track_id]

    def process_number_matches(filtered_matches):
        """處理號碼匹配結果（完全參考正常版本）"""
        for match in filtered_matches:
            track_id = match['player_id']
            team = match['team']
            number = match['number']
            player_key = f"{team}_{number}"

            if player_key not in player_registry:
                player_registry[player_key] = {
                    "team": team,
                    "number": number,
                    "track_id": track_id,
                    "last_seen": frame_count
                }
                if player_key not in player_states:
                    player_states[player_key] = {
                        'score': 0,
                        'stamina': 100,
                        'total_distance': 0
                    }
            else:
                player_registry[player_key]['track_id'] = track_id
                player_registry[player_key]['last_seen'] = frame_count

            old_tid = None
            for tid, pkey in player_associations.items():
                if pkey == player_key and tid != track_id:
                    old_tid = tid
                    break
            if old_tid is not None:
                del player_associations[old_tid]

            player_associations[track_id] = player_key

    def update_player_states(player_boxes):
        """更新球員狀態（完全參考正常版本 _update_player_states）"""
        detected_players = []
        
        # ===== 關鍵：只處理當前幀的 player_boxes =====
        for player_box in player_boxes:
            t_id = player_box['track_id']
            x1, y1, x2, y2 = player_box['x1'], player_box['y1'], player_box['x2'], player_box['y2']
            
            if t_id in player_associations:
                pkey = player_associations[t_id]
                if pkey in player_registry:
                    team = player_registry[pkey]['team']
                    number = player_registry[pkey]['number']
                    
                    # 從 DataManager 獲取球員資料
                    try:
                        player_data, error = data_manager.get_player_data(team, int(number))
                        if error:
                            print(f"[Pipeline階段3] 讀取球員資料錯誤: {error}")
                            player_info = pd.DataFrame()
                        else:
                            player_info = pd.DataFrame([player_data]) if player_data else pd.DataFrame()
                    except Exception as e:
                        print(f"[Pipeline階段3] DataManager 錯誤: {e}")
                        player_info = pd.DataFrame()

                    # 確保 player_states 有該球員的記錄
                    if pkey not in player_states:
                        player_states[pkey] = {
                            'score': 0,
                            'stamina': 100,
                            'total_distance': 0.0,
                            'is_last_touch': False,
                            'personal_scores': 0
                        }

                    extra_info = player_states[pkey]

                    # 更新移動距離和體力值
                    track_data = {}
                    if t_id in coordinate_mapper.tracking_data:
                        total_distance = coordinate_mapper.tracking_data[t_id]['total_distance']
                        track_data['total_distance'] = total_distance
                        update_player_stamina(pkey, total_distance)

                    # 組合所有資訊（完全參考正常版本）
                    new_player = {
                        'team': team,
                        'number': int(number),
                        'info': player_info,
                        'score': extra_info.get('score', 0),
                        'stamina': extra_info.get('stamina', 100),
                        'total_distance': track_data.get('total_distance', 0.0),
                        'stamina_color': get_stamina_color(extra_info.get('stamina', 100)),
                        'is_last_touch': extra_info.get('is_last_touch', False),
                        'personal_scores': extra_info.get('personal_scores', 0)
                    }

                    # 避免重複加入相同球員（參考正常版本）
                    if not any(p['team'] == new_player['team'] and
                               p['number'] == new_player['number']
                               for p in detected_players):
                        detected_players.append(new_player)

        return detected_players
    
    while True:
        task = input_q.get()
        if task is None:
            print("[Pipeline階段3] 收到結束訊號")
            output_q.put(None)
            break
        
                # 檢查是否為配置更新消息
        if isinstance(task, tuple) and task[0] == 'CONFIG_UPDATE':
            updated_config = task[1]
            print(f"[Pipeline階段3] 收到配置更新: {updated_config}")
            
            # 更新本地的配置和相關變量
            config.update(updated_config)
            
            if config.get('court_image_path'):
                court_image = cv2.imread(config['court_image_path'])
                image_points_list = config.get('image_points')
                if image_points_list is not None:
                    image_points = np.array(image_points_list, dtype=np.float32)
                    coordinate_mapper.set_reference(config['court_image_path'], image_points)
                    print(f"[Pipeline階段3] 成功載入球場圖片: {config['court_image_path']}")
                else:
                    print(f"[Pipeline階段3] 警告：配置中未提供 image_points")
            continue # 處理完配置後，繼續等待下一條消息
        
        frame_id, stage2_results = task
        frame_count = frame_id  # 更新幀計數
        
        # 更新 FPS 計算
        fps_monitor.update()
        
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
        original_frame = stage2_results.get('original_frame')
        
        # ===== 1. 處理 OCR 號碼匹配結果（參考正常版本）=====
        if ocr_matches:
            print(f"[Pipeline階段3] 處理 {len(ocr_matches)} 個 OCR 號碼匹配")
            process_number_matches(ocr_matches)
        
        # ===== 2. 清理過期關聯（參考正常版本）=====
        clean_old_associations()
        
        # ===== 3. 構建 player_positions =====
        player_positions = {}
        for player_box in player_boxes:
            track_id = player_box['track_id']
            x1, y1, x2, y2 = player_box['x1'], player_box['y1'], player_box['x2'], player_box['y2']
            
            # 確定球員所屬隊伍
            team = None
            if track_id in player_associations:
                player_key = player_associations[track_id]
                if player_key in player_registry:
                    team = player_registry[player_key]['team']
            
            if not team:
                # 從最近的team_box推斷隊伍
                min_dist = float('inf')
                player_center_x = (x1 + x2) / 2
                player_center_y = (y1 + y2) / 2
                
                for team_box in team_boxes:
                    tx1, ty1, tx2, ty2 = team_box['x1'], team_box['y1'], team_box['x2'], team_box['y2']
                    team_center_x = (tx1 + tx2) / 2
                    team_center_y = (ty1 + ty2) / 2
                    
                    dist = ((player_center_x - team_center_x) ** 2 + (player_center_y - team_center_y) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        team = team_box['team']
            
            player_positions[track_id] = (x1, y1, x2, y2, team or 'unknown')
        
        # === 4. 進籃分析 ===
        scores_updated = False
        if basketball_data and hoop_data:
            ball_tuple = (basketball_data['center'], basketball_data['left'], basketball_data['right'])
            hoop_tuple = (hoop_data['center'], hoop_data['left'], hoop_data['right'])
            
            scoring_analyzer.update_positions(ball_tuple, hoop_tuple)
            
            if basketball_data['center']:
                scoring_analyzer.update_last_touch(basketball_data['center'], player_positions)
            
            scored, team = scoring_analyzer.check_scoring(basketball_data['center'], hoop_data['center'])
            if scored:
                print(f"[Pipeline階段3] 進球！得分方：{team}")
                scores_updated = True
        
        # 預測籃球位置
        if not basketball_data:
            predicted_pos = scoring_analyzer.predict_basketball_position()
            if predicted_pos:
                basketball_data = {'center': predicted_pos, 'left': None, 'right': None}
                scoring_analyzer.update_last_touch(predicted_pos, player_positions)
        
        # === 5. 球場映射與軌跡 ===
        court_frame = None
        if court_image is not None:
            court_frame = court_image.copy()
            
            current_video_points = None
            for court_box in court_boxes:
                if court_box['cls'] == 1:  # RA area
                    x1, y1, x2, y2 = court_box['x1'], court_box['y1'], court_box['x2'], court_box['y2']
                    mask_points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                    
                    try:
                        geometric_points = extract_geometric_points(mask_points)
                        if "RA" in geometric_points and "points" in geometric_points["RA"]:
                            current_video_points = np.array([
                                geometric_points["RA"]["points"][0],
                                geometric_points["RA"]["points"][1],
                                geometric_points["RA"]["points"][2],
                                geometric_points["RA"]["points"][3]
                            ], dtype=np.float32)
                        else:
                            current_video_points = mask_points
                    except:
                        current_video_points = mask_points
                    break
            
            # 處理球員軌跡
            if current_video_points is not None and image_points is not None:
                try:
                    H, _ = cv2.findHomography(current_video_points, image_points)
                    
                    for player_box in player_boxes:
                        track_id = player_box['track_id']
                        x1, y1, x2, y2 = player_box['x1'], player_box['y1'], player_box['x2'], player_box['y2']
                        
                        player_point = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32).reshape(1, 1, 2)
                        projected_point = cv2.perspectiveTransform(player_point, H)
                        x, y = int(projected_point[0][0][0]), int(projected_point[0][0][1])
                        
                        total_distance = coordinate_mapper.track_movement(track_id, [x, y])
                        
                        if track_id in coordinate_mapper.tracking_data:
                            positions = coordinate_mapper.tracking_data[track_id]['positions']
                            color = coordinate_mapper.tracking_data[track_id]['color']
                            
                            if len(positions) > 1:
                                for i in range(len(positions) - 1):
                                    pt1 = tuple(map(int, positions[i]))
                                    pt2 = tuple(map(int, positions[i + 1]))
                                    cv2.line(court_frame, pt1, pt2, color, 2)
                            
                            cv2.circle(court_frame, (x, y), 5, color, -1)
                            cv2.putText(court_frame, f"ID:{track_id} Dist:{total_distance:.1f}m",
                                       (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                except Exception as e:
                    print(f"[Pipeline階段3] 球場映射錯誤: {e}")
        
        # === 6. 畫面繪製 ===
        if original_frame is not None:
            processed_frame = original_frame.copy()
            processed_frame = cv2.resize(processed_frame, (output_width, output_height))
        else:
            processed_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # 繪製籃球
        if display_options['ball'] and basketball_data and basketball_data['center']:
            center = basketball_data['center']
            left = basketball_data['left']
            right = basketball_data['right']
            if left and right:
                x1 = int(left[0] * output_width / (original_frame.shape[1] if original_frame is not None else output_width))
                y1 = int(left[1] * output_height / (original_frame.shape[0] if original_frame is not None else output_height))
                x2 = int(right[0] * output_width / (original_frame.shape[1] if original_frame is not None else output_width))
                y2 = int(right[1] * output_height / (original_frame.shape[0] if original_frame is not None else output_height))
                
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_frame, "BALL", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 繪製球員
        if display_options['player']:
            for player_box in player_boxes:
                track_id = player_box['track_id']
                x1, y1, x2, y2 = player_box['x1'], player_box['y1'], player_box['x2'], player_box['y2']
                
                if original_frame is not None:
                    x1 = int(x1 * output_width / original_frame.shape[1])
                    y1 = int(y1 * output_height / original_frame.shape[0])
                    x2 = int(x2 * output_width / original_frame.shape[1])
                    y2 = int(y2 * output_height / original_frame.shape[0])
                
                color = get_color_from_id(track_id)
                
                if track_id in player_associations:
                    player_key = player_associations[track_id]
                    if player_key in player_registry:
                        assoc_data = player_registry[player_key]
                        player_label = f"{class_display[5]} {assoc_data['team']}#{assoc_data['number']}"
                    else:
                        player_label = f"{class_display[5]} {track_id}"
                else:
                    player_label = f"{class_display[5]} {track_id}"
                
                cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(processed_frame, player_label,
                           (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 繪製隊伍
        if display_options['team']:
            for team_box in team_boxes:
                x1, y1, x2, y2 = team_box['x1'], team_box['y1'], team_box['x2'], team_box['y2']
                team = team_box['team']
                
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
                x1 = int(left[0] * output_width / (original_frame.shape[1] if original_frame is not None else output_width))
                y1 = int(left[1] * output_height / (original_frame.shape[0] if original_frame is not None else output_height))
                x2 = int(right[0] * output_width / (original_frame.shape[1] if original_frame is not None else output_width))
                y2 = int(right[1] * output_height / (original_frame.shape[0] if original_frame is not None else output_height))
                
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(processed_frame, "HOOP", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # 繪製 FPS 資訊
        processed_frame = draw_fps_on_frame(processed_frame, fps_monitor)
        
        # 縮放球場畫面
        if court_frame is not None:
            court_frame = cv2.resize(court_frame, (output_width, output_height))
        
        # ===== 7. 更新球員狀態（參考正常版本）=====
        detected_players = update_player_states(player_boxes)
        
        # 獲取當前得分
        current_scores = scoring_analyzer.get_scores()
        
        final_results = {
            'processed_frame': processed_frame,
            'court_frame': court_frame,
            'detected_players': detected_players,
            'scores': current_scores,
            'scores_updated': scores_updated,
            'stage3_time': time.time() - start_time,
            'stage3_status': 'completed'
        }
        
        output_q.put((frame_id, final_results))
        
        # 通知下一幀可以開始
        if current_event:
            current_event.set()
            print(f"[Pipeline階段3] frame_{frame_id} 完成，得分: {current_scores}，球員數: {len(detected_players)}")

class BasketballTracker:
    def __init__(self, player_model_path, court_model_path, data_folder, max_parallel_frames=3):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用設備: {self.device}")

        # 儲存模型路徑（用於多進程）
        self.player_model_path = player_model_path
        self.court_model_path = court_model_path

        # ===== 新增：FPS 監控 =====
        self.fps_monitor = FPSMonitor(window_size=30)
        self.show_fps = True  # 控制是否顯示 FPS

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

        self.stage3_config = {  # ===== 改為 self.stage3_config =====
            'output_width': 640,
            'output_height': 480,
            'class_names': ['3-s', 'ball', 'biv', 'hoop', 'number', 'player'],
            'team_mapping': {0: '3-s', 2: 'biv'},
            'court_image_path': None,  # 會在 set_court_reference 時更新
            'image_points': None,
            'data_folder': data_folder
        }

        self.stage3_process = self.ctx.Process(
            target=pipeline_stage3_worker,
            args=(self.stage2_to_stage3_q, self.stage3_output_q, 
                  self.frame_completion_events, self.stage3_config)
        )

        self.stage1_process.start()
        self.stage2_process.start()
        self.stage3_process.start()

        print(f"Pipeline 多進程模式已啟動，最大並行幀數: {max_parallel_frames}")

        import paddle
        print(paddle.is_compiled_with_cuda())  # 如果返回 False，表示 Paddle 未啟用 GPU 支援
        print(paddle.device.get_device())  # 檢查當前使用的設備

        # ===== 關鍵修改：移除主類別中的OCR初始化 =====
        # 註解掉原本的OCR初始化，因為已移至Pipeline Stage2
        # self.ocr = PaddleOCR(...)
        # self.number_recognizer = NumberRecognizer(self.ocr)
        print(f"OCR模型載入已移至Pipeline Stage2，使用NumberRecognizer邏輯")

        # 初始化資料管理器
        self.data_folder = data_folder
        self.data_manager = DataManager(data_folder)

        # 初始化各種分析器
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

        # 更新 FPS 計算
        self.fps_monitor.update()

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
        self.fps_monitor = FPSMonitor(window_size=30)  # 重置 FPS 監控

    def get_pipeline_status(self):
        """取得 pipeline 狀態資訊"""
        return {
            'frames_in_pipeline': self.frames_in_pipeline,
            'pending_results_count': len(self.pending_results),
            'next_expected_frame': self.next_expected_frame,
            'max_parallel_frames': self.max_parallel_frames,
            'current_frame_count': self.frame_count,
            'current_fps': self.fps_monitor.get_fps(),
            'avg_fps': self.fps_monitor.get_avg_fps()
        }

    def toggle_fps_display(self, show_fps=True):
        """切換 FPS 顯示開關"""
        self.show_fps = show_fps

    def get_fps_info(self):
        """獲取 FPS 資訊"""
        return {
            'current_fps': self.fps_monitor.get_fps(),
            'avg_fps': self.fps_monitor.get_avg_fps()
        }

    # 以下保持原有的所有其他方法不變...
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
        import cv2
        import numpy as np
        
        # 更新主類別屬性
        self.court_image = cv2.imread(court_image_path)
        self.image_points = np.array([
            [497, 347],  # top_left
            [497, 567],  # bottom_left
            [853, 347],  # top_right
            [853, 567]  # bottom_right
        ], dtype=np.float32)
        self.coordinate_mapper.set_reference(court_image_path, self.image_points)
            # 建立一個新的配置字典，準備發送給 worker
        new_config = {
            'court_image_path': court_image_path,
            'image_points': self.image_points.tolist()
        }
        
        # 通過隊列將配置更新消息發送給 stage3 worker
        # 我們將消息包裝在一個元組中，用一個特殊的標記 'CONFIG_UPDATE' 來區分
        self.stage2_to_stage3_q.put(('CONFIG_UPDATE', new_config))
        print(f"[主進程] 已發送球場配置更新指令到 Pipeline Stage 3")
        
        # ===== 新增：更新 Pipeline Stage3 配置 =====
        if hasattr(self, 'stage3_config'):
            self.stage3_config['court_image_path'] = court_image_path
            self.stage3_config['image_points'] = self.image_points.tolist()  # 轉為 list 供多進程使用
            print(f"[修正] 已更新 Pipeline Stage3 球場設定: {court_image_path}")
            
            # ===== 檢查球場圖片是否成功載入 =====
            if self.court_image is not None:
                print(f"[修正] 球場圖片載入成功，尺寸: {self.court_image.shape}")
            else:
                print(f"[錯誤] 球場圖片載入失敗: {court_image_path}")
        else:
            print("[警告] stage3_config 不存在，無法更新球場設定")

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
        """修改版：改善初始播放體驗"""
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
            
            # ===== 新增：改善初始響應 =====
            if result[0] is None and result[1] is None and result[2] is None:
                # 如果 Pipeline 還沒有結果，嘗試多次取得
                for attempt in range(3):  # 最多嘗試 3 次
                    pending_result = self._get_next_sequential_result()
                    if pending_result:
                        frame_id, final_results = pending_result
                        processed_frame = final_results.get('processed_frame')
                        court_frame = final_results.get('court_frame') 
                        detected_players = final_results.get('detected_players', [])
                        print(f"[改善] 第 {attempt+1} 次嘗試成功取得結果 frame_{frame_id}")
                        return processed_frame, court_frame, detected_players
                    
                    # 如果沒有結果，短暫等待
                    import time
                    time.sleep(0.1)  # 等待 100ms
                
                # ===== 如果還是沒有結果，返回當前幀作為暫時顯示 =====
                print(f"[改善] Pipeline 處理中，返回原始幀作為暫時顯示")
                display_frame = cv2.resize(frame, (self.output_width, self.output_height))
                
                # 在畫面上顯示處理狀態
                cv2.putText(display_frame, "Processing...", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(display_frame, f"Frame {self.frame_count}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # 返回原始幀，讓使用者知道系統在運行
                return display_frame, None, []
            
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
        """釋放資源（改善版）"""
        print("[關閉] 開始關閉 Pipeline...")
        
        # ===== 關閉 Pipeline 多進程 =====
        try:
            # 1. 先發送結束訊號給所有階段
            print("[關閉] 發送結束訊號...")
            try:
                self.stage1_input_q.put(None, timeout=1)
            except:
                pass
            
            # 2. 等待進程結束，設定較短的超時時間
            print("[關閉] 等待進程結束...")
            processes = [self.stage1_process, self.stage2_process, self.stage3_process]
            
            for i, process in enumerate(processes):
                if process and process.is_alive():
                    print(f"[關閉] 等待進程 {i+1} 結束...")
                    process.join(timeout=2)  # 縮短超時時間
                    
                    if process.is_alive():
                        print(f"[關閉] 強制終止進程 {i+1}")
                        process.terminate()
                        process.join(timeout=1)
                        
                        if process.is_alive():
                            print(f"[關閉] 強制殺死進程 {i+1}")
                            try:
                                process.kill()
                            except:
                                pass
            
            print("[關閉] Pipeline 多進程已關閉")
            
        except Exception as e:
            print(f"[關閉] 關閉 Pipeline 多進程時發生錯誤: {e}")

        # 3. 重置 pipeline 狀態
        try:
            self.reset_pipeline()
        except:
            pass

        # 4. 釋放視頻資源
        try:
            if self.video_capture is not None:
                self.video_capture.release()
            self.video_capture = None
            self.frame_buffer.clear()
            self.current_frame = None
            self.current_frame_index = 0
            self.is_playing = False
            if hasattr(self, 'coordinate_mapper'):
                self.coordinate_mapper.clear_trajectories()
        except Exception as e:
            print(f"[關閉] 釋放視頻資源時發生錯誤: {e}")
        
        print("[關閉] 資源釋放完成")

    # 其他所有原有方法保持不變...
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
            "fps": self.total_frames_processed / max(0.001, self.total_processing_time),
            "current_fps": self.fps_monitor.get_fps(),
            "avg_fps": self.fps_monitor.get_avg_fps()
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
        print(f"Current FPS: {stats['current_fps']:.2f}")
        print(f"Average FPS (Window): {stats['avg_fps']:.2f}")
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
        self.fps_monitor = FPSMonitor(window_size=30)

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
