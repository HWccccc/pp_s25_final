import os
import sys
import argparse
import threading
import multiprocessing as mp
import time
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import torch
from ultralytics import YOLO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
from concurrent.futures import ThreadPoolExecutor

# Set environment variable to avoid OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 設置matplotlib支持中文顯示
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# ===== 多進程工作函式 =====
def player_process_worker(frames, result_dict, player_model_path, device):
    """球員偵測多進程工作函式"""
    model = YOLO(player_model_path).to(device)
    results = {}
    times = {}
    for i, frame in enumerate(frames):
        with torch.no_grad():
            start = time.time()
            result = model.track(source=frame, conf=0.3, persist=True, tracker="bytetrack.yaml")
            end = time.time()
        inference_time = end - start
        results[i] = result
        times[i] = inference_time
        print(f"[多進程] 球員偵測 Frame-{i} 完成，耗時: {inference_time:.3f} 秒")
    result_dict['player_results'] = results
    result_dict['player_times'] = times

def court_process_worker(frames, result_dict, court_model_path, device):
    """球場偵測多進程工作函式"""
    model = YOLO(court_model_path).to(device)
    results = {}
    times = {}
    for i, frame in enumerate(frames):
        with torch.no_grad():
            start = time.time()
            result = model.track(source=frame, conf=0.25, persist=True, tracker="bytetrack.yaml")
            end = time.time()
        inference_time = end - start
        results[i] = result
        times[i] = inference_time
        print(f"[多進程] 球場偵測 Frame-{i} 完成，耗時: {inference_time:.3f} 秒")
    result_dict['court_results'] = results
    result_dict['court_times'] = times

# ===== 佇列工作函式 =====
def player_queue_worker(task_queue, result_queue, player_model_path, device):
    """球員偵測佇列工作函式"""
    print("[球員佇列工作進程] 啟動，載入模型中...")
    model = YOLO(player_model_path).to(device)
    with torch.no_grad():
        while True:
            task = task_queue.get()
            if task is None:
                print("[球員佇列工作進程] 收到結束訊號，退出。")
                break
            idx, frame = task
            start = time.time()
            res = model.track(source=frame, conf=0.3, persist=True, tracker="bytetrack.yaml")
            t = time.time() - start
            print(f"[球員佇列工作進程] Frame-{idx} 推理完成，耗時: {t:.3f} 秒")
            result_queue.put(('player', idx, res, t))

def court_queue_worker(task_queue, result_queue, court_model_path, device):
    """球場偵測佇列工作函式"""
    print("[球場佇列工作進程] 啟動，載入模型中...")
    model = YOLO(court_model_path).to(device)
    with torch.no_grad():
        while True:
            task = task_queue.get()
            if task is None:
                print("[球場佇列工作進程] 收到結束訊號，退出。")
                break
            idx, frame = task
            start = time.time()
            res = model.track(source=frame, conf=0.25, persist=True, tracker="bytetrack.yaml")
            t = time.time() - start
            print(f"[球場佇列工作進程] Frame-{idx} 推理完成，耗時: {t:.3f} 秒")
            result_queue.put(('court', idx, res, t))

# ===== 多進程池工作函式（支援模型快取）=====
def player_pool_worker_frame_sync(args):
    """球員偵測池工作函式 - 支援模型快取"""
    frame, idx, model_path, device = args

    # 檢查是否已在這個進程中載入過模型
    if not hasattr(player_pool_worker_frame_sync, 'model'):
        print(f"[進程 {os.getpid()}] 載入球員模型")
        player_pool_worker_frame_sync.model = YOLO(model_path).to(device)

    with torch.no_grad():
        start = time.time()
        result = player_pool_worker_frame_sync.model.track(source=frame, conf=0.3, persist=True, tracker="bytetrack.yaml")
        inference_time = time.time() - start

    print(f"[多進程池] 球員偵測 Frame-{idx} 完成，耗時: {inference_time:.3f} 秒")
    return ('player', idx, result, inference_time)

def court_pool_worker_frame_sync(args):
    """球場偵測池工作函式 - 支援模型快取"""
    frame, idx, model_path, device = args

    # 檢查是否已在這個進程中載入過模型
    if not hasattr(court_pool_worker_frame_sync, 'model'):
        print(f"[進程 {os.getpid()}] 載入球場模型")
        court_pool_worker_frame_sync.model = YOLO(model_path).to(device)

    with torch.no_grad():
        start = time.time()
        result = court_pool_worker_frame_sync.model.track(source=frame, conf=0.25, persist=True, tracker="bytetrack.yaml")
        inference_time = time.time() - start

    print(f"[多進程池] 球場偵測 Frame-{idx} 完成，耗時: {inference_time:.3f} 秒")
    return ('court', idx, result, inference_time)

# 功能選擇宏定義
MODE_SERIAL = 0
MODE_THREADING = 1
MODE_MULTIPROCESS = 2
MODE_QUEUE = 3
MODE_IMPROVED_THREADING = 4
MODE_MP_POOL = 5

class YOLODetector:
    def __init__(self, player_model_path, court_model_path):
        self.player_model_path = player_model_path
        self.court_model_path = court_model_path
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"使用裝置: {self.device}")

        self.player_model = None
        self.court_model = None
        
    def load_models(self):
        """預先載入兩個模型"""
        print("預先載入模型中...")
        start = time.time()
        self.player_model = YOLO(self.player_model_path).to(self.device)
        self.court_model = YOLO(self.court_model_path).to(self.device)

        with torch.no_grad():
            dummy_img = torch.zeros((1, 3, 640, 640), device=self.device)
            self.player_model.predict(source=dummy_img)
            self.court_model.predict(source=dummy_img)

        end = time.time()
        print(f"模型載入完成，耗時: {end - start:.3f} 秒")
    
    def warmup_mp_pool_models(self):
        """預熱多進程池模型"""
        from concurrent.futures import ProcessPoolExecutor

        print("[多進程池] 開始預熱模型...")
        warmup_start = time.time()

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with ProcessPoolExecutor(max_workers=2) as executor:
            player_warmup = executor.submit(
                player_pool_worker_frame_sync,
                (dummy_frame, -1, self.player_model_path, self.device)
            )
            court_warmup = executor.submit(
                court_pool_worker_frame_sync,
                (dummy_frame, -1, self.court_model_path, self.device)
            )

            player_warmup.result()
            court_warmup.result()

        warmup_time = time.time() - warmup_start
        print(f"[多進程池] 預熱完成，耗時: {warmup_time:.3f} 秒")
        return warmup_time

    def run_serial(self, video_path, max_frames):
        """序列模式"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        
        if self.player_model is None or self.court_model is None:
            self.load_models()
            
        frame_count = 0
        player_times = []
        court_times = []
        
        start_total = time.time()
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            with torch.no_grad():
                start = time.time()
                player_results = self.player_model.track(source=frame, conf=0.3, persist=True, tracker="bytetrack.yaml")
                player_end = time.time()
                player_time = player_end - start
                player_times.append(player_time)

                court_results = self.court_model.track(source=frame, conf=0.25, persist=True, tracker="bytetrack.yaml")
                court_end = time.time()
                court_time = court_end - player_end
                court_times.append(court_time)
                
                print(f"[序列模式] Frame-{frame_count}: 球員 {player_time:.3f}s, 球場 {court_time:.3f}s")
                    
            frame_count += 1
            
        end_total = time.time()
        total_time = end_total - start_total
        avg_player_time = sum(player_times) / len(player_times) if player_times else 0
        avg_court_time = sum(court_times) / len(court_times) if court_times else 0
        
        cap.release()
        return {
            "total_time": total_time,
            "frames": frame_count,
            "avg_player_time": avg_player_time,
            "avg_court_time": avg_court_time
        }

    def run_threading(self, video_path, max_frames):
        """多執行緒模式"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        if self.player_model is None or self.court_model is None:
            self.load_models()

        frames = []
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.copy())
            frame_count += 1
        cap.release()
        
        player_times = []
        court_times = []
        actual_frame_times = []  # 實際每幀並行處理時間
        total_start = time.time()
        
        for i, frame in enumerate(frames):
            frame_start = time.time()
            p_time = [0]
            c_time = [0]
            
            def _run_player():
                with torch.no_grad():
                    start = time.time()
                    self.player_model.track(source=frame, conf=0.3, persist=True, tracker="bytetrack.yaml")
                    p_time[0] = time.time() - start
            
            def _run_court():
                with torch.no_grad():
                    start = time.time()
                    self.court_model.track(source=frame, conf=0.25, persist=True, tracker="bytetrack.yaml")
                    c_time[0] = time.time() - start
            
            t1 = threading.Thread(target=_run_player)
            t2 = threading.Thread(target=_run_court)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            
            frame_end = time.time()
            actual_frame_time = frame_end - frame_start

            player_times.append(p_time[0])
            court_times.append(c_time[0])
            actual_frame_times.append(actual_frame_time)

            # 計算這一幀的潛在花費（序列模式耗時 - 並行模式耗時）
            serial_time = p_time[0] + c_time[0]
            potential_cost = serial_time - actual_frame_time

            print(f"[多執行緒] Frame-{i}: 球員 {p_time[0]:.3f}s, 球場 {c_time[0]:.3f}s, 實際 {actual_frame_time:.3f}s, 潛在花費 {potential_cost:.3f}s")
        
        total_time = time.time() - total_start
        avg_player_time = sum(player_times) / len(player_times) if player_times else 0
        avg_court_time = sum(court_times) / len(court_times) if court_times else 0
        avg_actual_time = sum(actual_frame_times) / len(actual_frame_times) if actual_frame_times else 0
        
        # 平均潛在花費 = 平均序列耗時 - 平均並行耗時
        avg_serial_time = avg_player_time + avg_court_time
        avg_potential_cost = avg_serial_time - avg_actual_time
        
        return {
            "total_time": total_time,
            "frames": len(frames),
            "avg_player_time": avg_player_time,
            "avg_court_time": avg_court_time,
            "avg_potential_cost": avg_potential_cost
        }

    def run_multiprocess(self, video_path, max_frames):
        """多進程模式"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        frames = []
        count = 0
        while count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.copy())
            count += 1
        cap.release()

        if not frames:
            return

        if sys.platform in ['win32', 'darwin']:
            mp.set_start_method("spawn", force=True)

        player_times = []
        court_times = []
        actual_frame_times = []  # 實際每幀並行處理時間
        total_start = time.time()

        for i, frame in enumerate(frames):
            frame_start = time.time()
            manager = mp.Manager()
            result_dict = manager.dict()

            p_player = mp.Process(
                target=player_process_worker,
                args=([frame], result_dict, self.player_model_path, self.device)
            )
            p_court = mp.Process(
                target=court_process_worker,
                args=([frame], result_dict, self.court_model_path, self.device)
            )

            p_player.start()
            p_court.start()
            p_player.join()
            p_court.join()

            frame_end = time.time()
            actual_frame_time = frame_end - frame_start

            pt = result_dict['player_times'][0]
            ct = result_dict['court_times'][0]
            player_times.append(pt)
            court_times.append(ct)
            actual_frame_times.append(actual_frame_time)

            # 計算這一幀的潛在花費（序列模式耗時 - 並行模式耗時）
            serial_time = pt + ct
            potential_cost = serial_time - actual_frame_time

            print(f"[多進程] Frame-{i}: 球員 {pt:.3f}s, 球場 {ct:.3f}s, 實際 {actual_frame_time:.3f}s, 潛在花費 {potential_cost:.3f}s")

        total_time = time.time() - total_start
        avg_player_time = sum(player_times)/len(player_times) if player_times else 0
        avg_court_time = sum(court_times)/len(court_times) if court_times else 0
        avg_actual_time = sum(actual_frame_times) / len(actual_frame_times) if actual_frame_times else 0

        # 平均潛在花費 = 平均序列耗時 - 平均並行耗時
        avg_serial_time = avg_player_time + avg_court_time
        avg_potential_cost = avg_serial_time - avg_actual_time

        return {
            "total_time": total_time,
            "frames": len(frames),
            "avg_player_time": avg_player_time,
            "avg_court_time": avg_court_time,
            "avg_potential_cost": avg_potential_cost
        }

    def run_queue_multiprocess(self, video_path, max_frames):
        """佇列多進程模式"""
        ctx = mp.get_context("spawn")

        player_task_queue = ctx.Queue(maxsize=10)
        court_task_queue = ctx.Queue(maxsize=10)
        result_queue = ctx.Queue()

        p_player = ctx.Process(
            target=player_queue_worker,
            args=(player_task_queue, result_queue, self.player_model_path, self.device)
        )
        p_court = ctx.Process(
            target=court_queue_worker,
            args=(court_task_queue, result_queue, self.court_model_path, self.device)
        )
        p_player.daemon = True
        p_court.daemon = True
        
        p_player.start()
        p_court.start()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        
        frames = []
        for i in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                if frames:
                    frames.append(frames[-1].copy())
                break
            frames.append(frame.copy())
        cap.release()
        
        start_total = time.time()
        player_times = {}
        court_times = {}
        actual_frame_times = []  # 實際每幀並行處理時間

        for frame_idx, frame in enumerate(frames):
            frame_start = time.time()
            player_task_queue.put((frame_idx, frame.copy()))
            court_task_queue.put((frame_idx, frame.copy()))

            frame_results = {}
            got_results = 0
            while got_results < 2:
                typ, idx, _, t = result_queue.get()
                if idx == frame_idx:
                    frame_results[typ] = t
                    got_results += 1
                    if typ == 'player':
                        player_times[idx] = t
                    else:
                        court_times[idx] = t

            frame_end = time.time()
            actual_frame_time = frame_end - frame_start
            actual_frame_times.append(actual_frame_time)

            # 計算這一幀的潛在花費（序列模式耗時 - 並行模式耗時）
            pt = frame_results.get('player', 0)
            ct = frame_results.get('court', 0)
            serial_time = pt + ct
            potential_cost = serial_time - actual_frame_time

            print(f"[佇列多進程] Frame-{frame_idx}: 球員 {pt:.3f}s, 球場 {ct:.3f}s, 實際 {actual_frame_time:.3f}s, 潛在花費 {potential_cost:.3f}s")

        player_task_queue.put(None)
        court_task_queue.put(None)

        p_player.join(timeout=3)
        p_court.join(timeout=3)

        if p_player.is_alive():
            p_player.terminate()
        if p_court.is_alive():
            p_court.terminate()
            
        end_total = time.time()
        total_time = end_total - start_total
        
        avg_player_time = sum(player_times.values()) / len(player_times) if player_times else 0
        avg_court_time = sum(court_times.values()) / len(court_times) if court_times else 0
        avg_actual_time = sum(actual_frame_times) / len(actual_frame_times) if actual_frame_times else 0
        
        # 平均潛在花費 = 平均序列耗時 - 平均並行耗時
        avg_serial_time = avg_player_time + avg_court_time
        avg_potential_cost = avg_serial_time - avg_actual_time
        
        return {
            "total_time": total_time,
            "frames": len(frames),
            "avg_player_time": avg_player_time,
            "avg_court_time": avg_court_time,
            "avg_potential_cost": avg_potential_cost
        }

    def run_improved_threading(self, video_path, max_frames):
        """改進執行緒池模式"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        if self.player_model is None or self.court_model is None:
            self.load_models()

        frames = []
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.copy())
            frame_count += 1
        cap.release()

        if not frames:
            return

        player_times = []
        court_times = []
        actual_frame_times = []  # 實際每幀並行處理時間

        thread_local = threading.local()

        def get_player_model():
            if not hasattr(thread_local, 'player_model'):
                thread_local.player_model = self.player_model
            return thread_local.player_model

        def get_court_model():
            if not hasattr(thread_local, 'court_model'):
                thread_local.court_model = self.court_model
            return thread_local.court_model

        def process_player(frame, frame_idx):
            model = get_player_model()
            with torch.no_grad():
                start = time.time()
                result = model.track(source=frame, conf=0.3, persist=True, tracker="bytetrack.yaml")
                elapsed = time.time() - start
                print(f"[ThreadPool] Frame-{frame_idx} 球員檢測完成，耗時: {elapsed:.3f} 秒")
                return result, elapsed

        def process_court(frame, frame_idx):
            model = get_court_model()
            with torch.no_grad():
                start = time.time()
                result = model.track(source=frame, conf=0.25, persist=True, tracker="bytetrack.yaml")
                elapsed = time.time() - start
                print(f"[ThreadPool] Frame-{frame_idx} 球場檢測完成，耗時: {elapsed:.3f} 秒")
                return result, elapsed

        total_start = time.time()

        with ThreadPoolExecutor(max_workers=2) as executor:
            for i, frame in enumerate(frames):
                frame_start = time.time()
                player_future = executor.submit(process_player, frame, i)
                court_future = executor.submit(process_court, frame, i)

                player_result, player_time = player_future.result()
                court_result, court_time = court_future.result()

                frame_end = time.time()
                actual_frame_time = frame_end - frame_start

                player_times.append(player_time)
                court_times.append(court_time)
                actual_frame_times.append(actual_frame_time)

                # 計算這一幀的潛在花費（序列模式耗時 - 並行模式耗時）
                serial_time = player_time + court_time
                potential_cost = serial_time - actual_frame_time

                print(f"[ThreadPool] Frame-{i} 完成: 球員 {player_time:.3f}s, 球場 {court_time:.3f}s, 實際 {actual_frame_time:.3f}s, 潛在花費 {potential_cost:.3f}s")

        total_time = time.time() - total_start
        avg_player_time = sum(player_times) / len(player_times) if player_times else 0
        avg_court_time = sum(court_times) / len(court_times) if court_times else 0
        avg_actual_time = sum(actual_frame_times) / len(actual_frame_times) if actual_frame_times else 0

        # 平均潛在花費 = 平均序列耗時 - 平均並行耗時
        avg_serial_time = avg_player_time + avg_court_time
        avg_potential_cost = avg_serial_time - avg_actual_time

        return {
            "total_time": total_time,
            "frames": len(frames),
            "avg_player_time": avg_player_time,
            "avg_court_time": avg_court_time,
            "avg_potential_cost": avg_potential_cost
        }

    def run_mp_pool(self, video_path, max_frames):
        """多進程池模式 - 含預熱機制（修正版）"""
        from concurrent.futures import ProcessPoolExecutor

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        frames = []
        count = 0
        while count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.copy())
            count += 1
        cap.release()

        if not frames:
            return

        print(f"[多進程池] 使用 2 個工作進程 (逐幀同步)")

        # 🔥 關鍵修正：在同一個進程池中進行預熱和正式處理
        with ProcessPoolExecutor(max_workers=2) as executor:
            # 預熱階段
            print("[多進程池] 開始預熱模型...")
            warmup_start = time.time()

            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # 預熱兩個worker
            player_warmup = executor.submit(
                player_pool_worker_frame_sync,
                (dummy_frame, -1, self.player_model_path, self.device)
            )
            court_warmup = executor.submit(
                court_pool_worker_frame_sync,
                (dummy_frame, -1, self.court_model_path, self.device)
            )

            # 等待預熱完成
            player_warmup.result()
            court_warmup.result()

            warmup_time = time.time() - warmup_start
            print(f"[多進程池] 預熱完成，耗時: {warmup_time:.3f} 秒")

            # 🎯 正式測量開始（在同一個進程池中）
            measurement_start = time.time()
            player_times = []
            court_times = []
            actual_frame_times = []  # 實際每幀並行處理時間

            # 正式處理所有幀
            for i, frame in enumerate(frames):
                frame_start = time.time()
                player_future = executor.submit(
                    player_pool_worker_frame_sync,
                    (frame, i, self.player_model_path, self.device)
                )
                court_future = executor.submit(
                    court_pool_worker_frame_sync,
                    (frame, i, self.court_model_path, self.device)
                )

                try:
                    player_result_type, player_idx, player_result, player_time = player_future.result()
                    court_result_type, court_idx, court_result, court_time = court_future.result()

                    frame_end = time.time()
                    actual_frame_time = frame_end - frame_start

                    player_times.append(player_time)
                    court_times.append(court_time)
                    actual_frame_times.append(actual_frame_time)

                    # 計算這一幀的潛在花費（序列模式耗時 - 並行模式耗時）
                    serial_time = player_time + court_time
                    potential_cost = serial_time - actual_frame_time

                    print(f"[多進程池] Frame-{i} 完成: 球員 {player_time:.3f}s, 球場 {court_time:.3f}s, 實際 {actual_frame_time:.3f}s, 潛在花費 {potential_cost:.3f}s")

                except Exception as e:
                    print(f"處理Frame-{i}時發生錯誤: {e}")
                    continue

        total_time = time.time() - measurement_start
        avg_player_time = sum(player_times) / len(player_times) if player_times else 0
        avg_court_time = sum(court_times) / len(court_times) if court_times else 0
        avg_actual_time = sum(actual_frame_times) / len(actual_frame_times) if actual_frame_times else 0

        # 平均潛在花費 = 平均序列耗時 - 平均並行耗時
        avg_serial_time = avg_player_time + avg_court_time
        avg_potential_cost = avg_serial_time - avg_actual_time

        print(f"多進程池模式完成，測量耗時: {total_time:.3f} 秒，預熱耗時: {warmup_time:.3f} 秒")

        return {
            "total_time": total_time,
            "frames": len(frames),
            "avg_player_time": avg_player_time,
            "avg_court_time": avg_court_time,
            "avg_potential_cost": avg_potential_cost,
            "warmup_time": warmup_time
        }

class YOLODetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO 多機制影片辨識")
        self.root.geometry("800x600")
        
        self.detector = None
        self.results = {}
        
        self._create_widgets()

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        settings_frame = ttk.LabelFrame(main_frame, text="設定", padding=10)
        settings_frame.pack(fill=tk.X, pady=5)
        
        # 球員模型路徑
        ttk.Label(settings_frame, text="球員模型路徑:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.player_model_path_var = tk.StringVar(value="best_demo_v2.pt")
        ttk.Entry(settings_frame, textvariable=self.player_model_path_var, width=40).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        ttk.Button(settings_frame, text="瀏覽", command=lambda: self._browse_model("player")).grid(row=0, column=2, padx=5, pady=5)
        
        # 球場模型路徑
        ttk.Label(settings_frame, text="球場模型路徑:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.court_model_path_var = tk.StringVar(value="Court_best.pt")
        ttk.Entry(settings_frame, textvariable=self.court_model_path_var, width=40).grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        ttk.Button(settings_frame, text="瀏覽", command=lambda: self._browse_model("court")).grid(row=1, column=2, padx=5, pady=5)
        
        # 影片路徑
        ttk.Label(settings_frame, text="影片路徑:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.video_path_var = tk.StringVar()
        ttk.Entry(settings_frame, textvariable=self.video_path_var, width=40).grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        ttk.Button(settings_frame, text="瀏覽", command=self._browse_video).grid(row=2, column=2, padx=5, pady=5)
        
        # 影格數量
        ttk.Label(settings_frame, text="處理影格數:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.frames_var = tk.IntVar(value=10)
        ttk.Entry(settings_frame, textvariable=self.frames_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 執行模式
        ttk.Label(settings_frame, text="執行模式:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.mode_var = tk.IntVar(value=MODE_SERIAL)
        modes = [("序列模式", MODE_SERIAL), 
                 ("多執行緒模式", MODE_THREADING), 
                 ("多進程模式", MODE_MULTIPROCESS), 
                 ("佇列多進程模式", MODE_QUEUE),
                 ("改進執行緒池模式", MODE_IMPROVED_THREADING),
                 ("多進程池模式", MODE_MP_POOL)]
                 
        mode_frame = ttk.Frame(settings_frame)
        mode_frame.grid(row=4, column=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        for i, (text, value) in enumerate(modes):
            ttk.Radiobutton(mode_frame, text=text, variable=self.mode_var, value=value).grid(row=0, column=i, padx=5)
            
        # 按鈕區域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        ttk.Button(button_frame, text="預載入模型", command=self._preload_models).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="預熱多進程池", command=self._warmup_mp_pool).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="開始處理", command=self._start_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清除結果", command=self._clear_results).pack(side=tk.LEFT, padx=5)
        
        # 結果顯示區域
        results_frame = ttk.LabelFrame(main_frame, text="結果", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 文字輸出
        self.log_text = tk.Text(results_frame, height=10, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        scrollbar = ttk.Scrollbar(results_frame, command=self.log_text.yview)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # 狀態列
        self.status_var = tk.StringVar(value="就緒")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
    def _browse_model(self, model_type):
        """瀏覽選擇模型檔案"""
        filename = filedialog.askopenfilename(
            title=f"選擇{model_type}模型",
            filetypes=[("YOLO模型", "*.pt"), ("所有檔案", "*.*")]
        )
        if filename:
            if model_type == "player":
                self.player_model_path_var.set(filename)
            elif model_type == "court":
                self.court_model_path_var.set(filename)
            
    def _browse_video(self):
        """瀏覽選擇影片檔案"""
        filename = filedialog.askopenfilename(
            title="選擇影片檔案",
            filetypes=[("影片檔案", "*.mp4 *.avi *.mov"), ("所有檔案", "*.*")]
        )
        if filename:
            self.video_path_var.set(filename)
    
    def _preload_models(self):
        """預載入模型"""
        player_model_path = self.player_model_path_var.get()
        court_model_path = self.court_model_path_var.get()
        
        if not player_model_path or not court_model_path:
            messagebox.showwarning("警告", "請指定兩個模型路徑")
            return
            
        if not os.path.exists(player_model_path):
            messagebox.showerror("錯誤", f"球員模型檔案不存在: {player_model_path}")
            return
            
        if not os.path.exists(court_model_path):
            messagebox.showerror("錯誤", f"球場模型檔案不存在: {court_model_path}")
            return
        
        self.status_var.set("正在預載入模型...")
        self.root.update()

        if self.detector is None:
            self.detector = YOLODetector(player_model_path, court_model_path)
        self.detector.load_models()
        
        self.status_var.set("模型已預載入")
        self._log_message("模型已預載入，現在可以開始測試了。")
    
    def _warmup_mp_pool(self):
        """預熱多進程池模型"""
        player_model_path = self.player_model_path_var.get()
        court_model_path = self.court_model_path_var.get()

        if not player_model_path or not court_model_path:
            messagebox.showwarning("警告", "請指定兩個模型路徑")
            return

        if not os.path.exists(player_model_path):
            messagebox.showerror("錯誤", f"球員模型檔案不存在: {player_model_path}")
            return

        if not os.path.exists(court_model_path):
            messagebox.showerror("錯誤", f"球場模型檔案不存在: {court_model_path}")
            return

        self.status_var.set("正在預熱多進程池...")
        self.root.update()

        if self.detector is None:
            self.detector = YOLODetector(player_model_path, court_model_path)

        try:
            warmup_time = self.detector.warmup_mp_pool_models()
            self.status_var.set("多進程池已預熱")
            self._log_message(f"多進程池預熱完成，耗時: {warmup_time:.3f} 秒")
        except Exception as e:
            self.status_var.set("預熱失敗")
            self._log_message(f"預熱失敗: {str(e)}")
            messagebox.showerror("錯誤", f"預熱失敗:\n{str(e)}")

    def _start_detection(self):
        """開始偵測處理"""
        video_path = self.video_path_var.get()
        if not video_path:
            messagebox.showwarning("警告", "請選擇影片檔案")
            return
            
        if not os.path.exists(video_path):
            messagebox.showerror("錯誤", f"影片檔案不存在: {video_path}")
            return
            
        player_model_path = self.player_model_path_var.get()
        court_model_path = self.court_model_path_var.get()
        
        if not player_model_path or not court_model_path:
            messagebox.showwarning("警告", "請指定兩個模型路徑")
            return
            
        if not os.path.exists(player_model_path):
            messagebox.showerror("錯誤", f"球員模型檔案不存在: {player_model_path}")
            return
            
        if not os.path.exists(court_model_path):
            messagebox.showerror("錯誤", f"球場模型檔案不存在: {court_model_path}")
            return

        self.status_var.set("處理中...")
        self.root.update()

        if self.detector is None:
            self.detector = YOLODetector(player_model_path, court_model_path)
            self.detector.load_models()

        mode = self.mode_var.get()
        max_frames = self.frames_var.get()

        threading.Thread(target=self._run_detection, args=(mode, video_path, max_frames), daemon=True).start()
        
    def _run_detection(self, mode, video_path, max_frames):
        """執行選定的偵測模式"""
        try:
            self.results = {}

            self._log_message(f"開始執行，模式: {self._get_mode_name(mode)}, 影格數: {max_frames}")
            
            if mode == MODE_SERIAL:
                self._log_message("執行序列模式...")
                self.results = self.detector.run_serial(video_path, max_frames)
            elif mode == MODE_THREADING:
                self._log_message("執行多執行緒模式...")
                self.results = self.detector.run_threading(video_path, max_frames)
            elif mode == MODE_MULTIPROCESS:
                self._log_message("執行多進程模式...")
                self.results = self.detector.run_multiprocess(video_path, max_frames)
            elif mode == MODE_QUEUE:
                self._log_message("執行佇列多進程模式...")
                self.results = self.detector.run_queue_multiprocess(video_path, max_frames)
            elif mode == MODE_IMPROVED_THREADING:
                self._log_message("執行改進的多執行緒模式...")
                self.results = self.detector.run_improved_threading(video_path, max_frames)
            elif mode == MODE_MP_POOL:
                self._log_message("執行多進程池模式...")
                self.results = self.detector.run_mp_pool(video_path, max_frames)
            else:
                messagebox.showerror("錯誤", f"未知的執行模式: {mode}")
                return

            if "total_time" in self.results:
                total_time = self.results["total_time"]
                frames = self.results.get("frames", 0)
                avg_player_time = self.results.get("avg_player_time", 0)
                avg_court_time = self.results.get("avg_court_time", 0)
                warmup_time = self.results.get("warmup_time", 0)
                avg_potential_cost = self.results.get("avg_potential_cost", 0)

                self._log_message(f"\n===== {self._get_mode_name(mode)} 執行結果 =====")
                self._log_message(f"總處理時間: {total_time:.3f} 秒")
                if warmup_time > 0:
                    self._log_message(f"預熱時間: {warmup_time:.3f} 秒 (不計入測量)")
                self._log_message(f"處理總幀數: {frames} 幀")
                self._log_message(f"平均每幀耗時: {total_time/frames:.3f} 秒 (約 {frames/total_time:.2f} FPS)")
                self._log_message(f"平均球員偵測耗時: {avg_player_time:.3f} 秒")
                self._log_message(f"平均球場偵測耗時: {avg_court_time:.3f} 秒")

                # 顯示平均潛在花費 (僅非序列模式)
                if mode != MODE_SERIAL and avg_potential_cost > 0:
                    self._log_message(f"平均潛在花費 (並行節省時間): {avg_potential_cost:.3f} 秒")

            self.root.after(0, lambda: self.status_var.set("已完成"))

        except Exception as e:
            self._log_message(f"錯誤: {str(e)}")
            import traceback
            self._log_message(traceback.format_exc())
            self.root.after(0, lambda: self.status_var.set("發生錯誤"))
            
    def _get_mode_name(self, mode):
        """根據模式ID獲取模式名稱"""
        mode_names = {
            MODE_SERIAL: "序列模式",
            MODE_THREADING: "多執行緒模式",
            MODE_MULTIPROCESS: "多進程模式",
            MODE_QUEUE: "佇列多進程模式",
            MODE_IMPROVED_THREADING: "改進的多執行緒模式",
            MODE_MP_POOL: "多進程池模式"
        }
        return mode_names.get(mode, "未知模式")
    
    def _log_message(self, message):
        """記錄訊息到文字框"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        print(message)
        
    def _clear_results(self):
        """清除結果顯示"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.results = {}
        self.status_var.set("就緒")

if __name__ == "__main__":
    if sys.platform in ['win32', 'darwin']:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
    
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    parser = argparse.ArgumentParser(description='YOLO六種檢測機制')
    parser.add_argument('--mode', type=int, choices=[0, 1, 2, 3, 4, 5],
                      help='偵測模式: 0=序列, 1=多執行緒, 2=多進程, 3=佇列多進程, 4=改進執行緒池, 5=多進程池')
    parser.add_argument('--frames', type=int, default=10, help='要處理的影格數')
    parser.add_argument('--video', type=str, help='影片檔案路徑')
    parser.add_argument('--player_model', type=str, default='best_demo_v2.pt', help='球員模型路徑')
    parser.add_argument('--court_model', type=str, default='Court_best.pt', help='球場模型路徑')
    
    args = parser.parse_args()

    root = tk.Tk()
    app = YOLODetectionApp(root)

    if args.video and os.path.exists(args.video):
        app.video_path_var.set(args.video)
        
    if args.player_model and os.path.exists(args.player_model):
        app.player_model_path_var.set(args.player_model)
        
    if args.court_model and os.path.exists(args.court_model):
        app.court_model_path_var.set(args.court_model)
        
    if args.frames > 0:
        app.frames_var.set(args.frames)
        
    if args.mode is not None:
        app.mode_var.set(args.mode)
        if args.video and os.path.exists(args.video):
            root.after(1000, app._preload_models)
            root.after(2000, app._start_detection)
    
    root.mainloop()