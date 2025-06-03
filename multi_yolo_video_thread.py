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
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']  # 中文字型優先順序
matplotlib.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# ===== 修改開始: 將多進程工作函式移至模組頂層，避免 pickle 非可序列化的 self =====
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

# ===== 新增頂層佇列工作函式，以避免 pickle 自身鎖 =====
def player_queue_worker(task_queue, result_queue, player_model_path, device):
    """球員偵測佇列工作函式"""
    print("[球員佇列工作進程] 啟動，載入模型中...")
    model = YOLO(player_model_path).to(device)
    with torch.no_grad():
        while True:
            task = task_queue.get()
            if task is None:
                print("[球員佇列工作進程] 收到結束訊號，退出。"); break
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
                print("[球場佇列工作進程] 收到結束訊號，退出。"); break
            idx, frame = task
            start = time.time()
            res = model.track(source=frame, conf=0.25, persist=True, tracker="bytetrack.yaml")
            t = time.time() - start
            print(f"[球場佇列工作進程] Frame-{idx} 推理完成，耗時: {t:.3f} 秒")
            result_queue.put(('court', idx, res, t))
# ===== 修改結束 =====

# 功能選擇宏定義
MODE_SERIAL = 0              # 序列模式
MODE_THREADING = 1           # 多執行緒模式
MODE_MULTIPROCESS = 2        # 多進程模式
MODE_QUEUE = 3               # 佇列多進程模式
MODE_IMPROVED_THREADING = 4  # 改進的多執行緒模式（使用線程池）
MODE_MP_POOL = 5             # 多進程池模式

# ===== 新增多進程池工作函式 =====
def player_pool_worker(args):
    """球員偵測池工作函式"""
    frame, idx, model_path, device = args
    model = YOLO(model_path).to(device)
    with torch.no_grad():
        start = time.time()
        result = model.track(source=frame, conf=0.3, persist=True, tracker="bytetrack.yaml")
        inference_time = time.time() - start
    print(f"[多進程池] 球員偵測 Frame-{idx} 完成，耗時: {inference_time:.3f} 秒")
    return ('player', idx, result, inference_time)

def court_pool_worker(args):
    """球場偵測池工作函式"""
    frame, idx, model_path, device = args
    model = YOLO(model_path).to(device)
    with torch.no_grad():
        start = time.time()
        result = model.track(source=frame, conf=0.25, persist=True, tracker="bytetrack.yaml")
        inference_time = time.time() - start
    print(f"[多進程池] 球場偵測 Frame-{idx} 完成，耗時: {inference_time:.3f} 秒")
    return ('court', idx, result, inference_time)
# ...existing code...

class YOLODetector:
    def __init__(self, player_model_path, court_model_path):
        self.player_model_path = player_model_path
        self.court_model_path = court_model_path
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"使用裝置: {self.device}")
        
        # 先預載入模型，排除初始載入時間的影響
        self.player_model = None
        self.court_model = None
        
    def load_models(self):
        """預先載入兩個模型，以排除載入時間的影響"""
        print("預先載入模型中...")
        start = time.time()
        self.player_model = YOLO(self.player_model_path).to(self.device)
        self.court_model = YOLO(self.court_model_path).to(self.device)
        
        # 預先進行模型融合，避免多執行緒中重複融合導致錯誤
        with torch.no_grad():
            # 執行一次推理，確保模型已完全初始化和融合
            dummy_img = torch.zeros((1, 3, 640, 640), device=self.device)
            self.player_model.predict(source=dummy_img)
            self.court_model.predict(source=dummy_img)
        
        end = time.time()
        print(f"模型載入完成，耗時: {end - start:.3f} 秒")
    
    def run_serial(self, video_path, max_frames):
        """序列模式：單線程執行所有影格辨識"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"無法開啟影片: {video_path}")
            return
        
        # 確保模型已載入
        if self.player_model is None or self.court_model is None:
            self.load_models()
            
        frame_count = 0
        player_times = []
        court_times = []
        
        start_total = time.time()
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                print("讀取影片結束或失敗。")
                break
                
            # 執行球員推理
            with torch.no_grad():
                start = time.time()
                player_results = self.player_model.track(source=frame, conf=0.3, persist=True, tracker="bytetrack.yaml")
                player_end = time.time()
                player_time = player_end - start
                player_times.append(player_time)
                
                # 執行球場推理
                court_results = self.court_model.track(source=frame, conf=0.25, persist=True, tracker="bytetrack.yaml")
                court_end = time.time()
                court_time = court_end - player_end
                court_times.append(court_time)
                
                print(f"[序列模式] Frame-{frame_count}:")
                print(f"  球員偵測耗時: {player_time:.3f} 秒")
                print(f"  球場偵測耗時: {court_time:.3f} 秒")
                    
            frame_count += 1
            
        end_total = time.time()
        total_time = end_total - start_total
        avg_player_time = sum(player_times) / len(player_times) if player_times else 0
        avg_court_time = sum(court_times) / len(court_times) if court_times else 0
        
        print(f"序列模式完成，總耗時: {total_time:.3f} 秒，平均每幀: {total_time/frame_count:.3f} 秒")
        print(f"球員偵測平均耗時: {avg_player_time:.3f} 秒")
        print(f"球場偵測平均耗時: {avg_court_time:.3f} 秒")
        cap.release()
        
        return {
            "total_time": total_time,
            "frames": frame_count,
            "avg_player_time": avg_player_time,
            "avg_court_time": avg_court_time
        }
        
    # ==== 執行緒任務函式 ====
    def _player_detection_thread(self, frames, results, times):
        """球員偵測執行緒工作函數"""
        for i, frame in enumerate(frames):
            with torch.no_grad():
                start = time.time()
                result = self.player_model.track(source=frame, conf=0.3, persist=True, tracker="bytetrack.yaml")
                end = time.time()
                
                inference_time = end - start
                times[i] = inference_time
                results[i] = result
                print(f"[多執行緒] Frame-{i} 球員偵測完成，耗時: {inference_time:.3f} 秒")
                
    def _court_detection_thread(self, frames, results, times):
        """球場偵測執行緒工作函數"""
        for i, frame in enumerate(frames):
            with torch.no_grad():
                start = time.time()
                result = self.court_model.track(source=frame, conf=0.25, persist=True, tracker="bytetrack.yaml")
                end = time.time()
                
                inference_time = end - start
                times[i] = inference_time
                results[i] = result
                print(f"[多執行緒] Frame-{i} 球場偵測完成，耗時: {inference_time:.3f} 秒")
                
    def run_threading(self, video_path, max_frames):
        """多執行緒模式：一個執行緒處理球員偵測，一個處理球場偵測"""
        # 讀取影片幀
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"無法開啟影片: {video_path}")
            return
            
        # 確保模型已載入
        if self.player_model is None or self.court_model is None:
            self.load_models()
        
        # 讀取影片幀到記憶體
        frames = []
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.copy())
            frame_count += 1
        cap.release()
          # 同一張 frame 並行兩個 thread，完成後才進下一張
        player_results = []
        court_results  = []
        player_times   = []
        court_times    = []
        total_start = time.time()
        for i, frame in enumerate(frames):
            p_time = [0]
            c_time = [0]
            p_res  = [None]
            c_res  = [None]
            
            def _run_player():
                with torch.no_grad():
                    start = time.time()
                    p_res[0] = self.player_model.track(
                        source=frame, conf=0.3, persist=True, tracker="bytetrack.yaml"
                    )
                    p_time[0] = time.time() - start
            
            def _run_court():
                with torch.no_grad():
                    start = time.time()
                    c_res[0] = self.court_model.track(
                        source=frame, conf=0.25, persist=True, tracker="bytetrack.yaml"
                    )
                    c_time[0] = time.time() - start
            
            t1 = threading.Thread(target=_run_player)
            t2 = threading.Thread(target=_run_court)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            
            player_results.append(p_res[0])
            court_results.append(c_res[0])
            player_times.append(p_time[0])
            court_times.append(c_time[0])
            
            print(f"[多執行緒] Frame-{i}: 球員 {p_time[0]:.3f}s, 球場 {c_time[0]:.3f}s")
        
        total_time = time.time() - total_start
        avg_player_time = sum(player_times) / len(player_times) if player_times else 0
        avg_court_time  = sum(court_times)  / len(court_times)  if court_times  else 0
        
        print(f"多執行緒模式完成，總耗時: {total_time:.3f} 秒，平均每幀: {total_time/len(frames):.3f} 秒")
        print(f"球員偵測平均耗時: {avg_player_time:.3f} 秒")
        print(f"球場偵測平均耗時: {avg_court_time:.3f} 秒")
        
        return {
            "total_time": total_time,
            "frames": len(frames),
            "avg_player_time": avg_player_time,
            "avg_court_time": avg_court_time
        }

    # ==== 多進程模式 ====    
    def run_multiprocess(self, video_path, max_frames):
        """多進程模式：對每張影格啟動兩個 process（player、court）並行，完成後才處理下一張"""
        # 讀取影片幀
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"無法開啟影片: {video_path}")
            return

        # 讀取到記憶體
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
            print("沒有讀取到有效的影片幀")
            return

        # Windows/macOS 必要
        if sys.platform in ['win32', 'darwin']:
            mp.set_start_method("spawn", force=True)

        player_times = []
        court_times  = []
        total_start = time.time()

        for i, frame in enumerate(frames):
            # 每張影格都用新的 Manager 與 dict
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

            # 啟動並等待
            p_start = time.time()
            p_player.start()
            p_court.start()
            p_player.join()
            p_court.join()
            p_end = time.time()

            # 從 result_dict 取出 index 0 的時間
            pt = result_dict['player_times'][0]
            ct = result_dict['court_times'][0]
            player_times.append(pt)
            court_times.append(ct)

            print(f"[多進程] Frame-{i}: 球員 {pt:.3f}s, 球場 {ct:.3f}s, 含啟動/同步 {p_end-p_start:.3f}s")

        total_time = time.time() - total_start
        avg_player_time = sum(player_times)/len(player_times) if player_times else 0
        avg_court_time  = sum(court_times) /len(court_times)  if court_times  else 0

        print(f"多進程模式完成，總耗時: {total_time:.3f} 秒，平均每幀: {total_time/len(frames):.3f} 秒")
        print(f"球員偵測平均耗時: {avg_player_time:.3f} 秒")
        print(f"球場偵測平均耗時: {avg_court_time:.3f} 秒")

        return {
            "total_time": total_time,
            "frames": len(frames),
            "avg_player_time": avg_player_time,
            "avg_court_time": avg_court_time
        }

    def run_queue_multiprocess(self, video_path, max_frames):
        """佇列多進程模式：使用任務佇列的生產者消費者模式"""
        # 確保使用spawn方式
        ctx = mp.get_context("spawn")
        
        # 建立佇列
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
        
        # 讀取影片幀并分配任務
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"無法開啟影片: {video_path}")
            return
        
        start_total = time.time()
        
        # 讀取和分派任務
        frames_sent = 0
        while frames_sent < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 將同一幀分配給兩個不同的任務佇列
            player_task_queue.put((frames_sent, frame.copy()))
            court_task_queue.put((frames_sent, frame.copy()))
            
            frames_sent += 1
        
        cap.release()
        
        # 收集結果
        expected = frames_sent * 2
        got = 0
        player_times, court_times = {}, {}
        while got < expected:
            typ, idx, _, t = result_queue.get()
            if typ == 'player': player_times[idx] = t
            else: court_times[idx] = t
            got += 1

        player_task_queue.put(None)
        court_task_queue.put(None)
        
        # 等待工作進程結束
        p_player.join(timeout=3)
        p_court.join(timeout=3)
        
        # 如果進程沒有正常結束，強制終止
        if p_player.is_alive():
            p_player.terminate()
        if p_court.is_alive():
            p_court.terminate()
            
        end_total = time.time()
        total_time = end_total - start_total
        
        # 計算平均時間
        avg_player_time = sum(player_times.values()) / len(player_times) if player_times else 0
        avg_court_time = sum(court_times.values()) / len(court_times) if court_times else 0
        
        print(f"佇列多進程模式完成，總耗時: {total_time:.3f} 秒，平均每幀: {total_time/frames_sent:.3f} 秒")
        print(f"球員偵測平均耗時: {avg_player_time:.3f} 秒")
        print(f"球場偵測平均耗時: {avg_court_time:.3f} 秒")
        
        return {
            "total_time": total_time,
            "frames": frames_sent,
            "avg_player_time": avg_player_time,
            "avg_court_time": avg_court_time
        }
        
    def run_improved_threading(self, video_path, max_frames):
        """改進的多執行緒模式：使用線程池並行處理所有影格
        
        這種模式利用 ThreadPoolExecutor 來管理線程，避免每次都創建和銷毀線程的開銷。
        同時，它一次性處理所有影格，而不是一次處理一個影格。
        """
        # 讀取影片幀
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"無法開啟影片: {video_path}")
            return
            
        # 確保模型已載入
        if self.player_model is None or self.court_model is None:
            self.load_models()
        
        # 讀取影片幀到記憶體
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
            print("沒有讀取到有效的影片幀")
            return
        
        player_results = [None] * len(frames)
        court_results = [None] * len(frames)
        player_times = [0] * len(frames)
        court_times = [0] * len(frames)
        
        # 創建ThreadLocal存儲，確保每個執行緒有自己的模型副本
        thread_local = threading.local()
        
        def get_player_model():
            if not hasattr(thread_local, 'player_model'):
                # 如果執行緒還沒有自己的模型，複製主模型的權重但不進行融合
                print(f"執行緒 {threading.current_thread().name} 初始化球員模型")
                thread_local.player_model = self.player_model
            return thread_local.player_model
            
        def get_court_model():
            if not hasattr(thread_local, 'court_model'):
                # 如果執行緒還沒有自己的模型，複製主模型的權重但不進行融合
                print(f"執行緒 {threading.current_thread().name} 初始化球場模型")
                thread_local.court_model = self.court_model
            return thread_local.court_model
        
        def process_player(idx):
            """球員檢測任務函數"""
            model = get_player_model()
            with torch.no_grad():
                start = time.time()
                # 使用 predict 而非 track 來避免重複融合模型
                result = model.predict(
                    source=frames[idx], conf=0.3, tracker="bytetrack.yaml"
                )
                elapsed = time.time() - start
                player_results[idx] = result
                player_times[idx] = elapsed
                print(f"[改進執行緒池] Frame-{idx} 球員檢測完成，耗時: {elapsed:.3f} 秒")
                return idx
        
        def process_court(idx):
            """球場檢測任務函數"""
            model = get_court_model()
            with torch.no_grad():
                start = time.time()
                # 使用 predict 而非 track 來避免重複融合模型
                result = model.predict(
                    source=frames[idx], conf=0.25, tracker="bytetrack.yaml"
                )
                elapsed = time.time() - start
                court_results[idx] = result
                court_times[idx] = elapsed
                print(f"[改進執行緒池] Frame-{idx} 球場檢測完成，耗時: {elapsed:.3f} 秒")
                return idx
        
        total_start = time.time()
        
        # 使用執行緒池並行處理所有幀
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交所有球員檢測任務
            player_futures = [executor.submit(process_player, i) for i in range(len(frames))]
            # 提交所有球場檢測任務
            court_futures = [executor.submit(process_court, i) for i in range(len(frames))]
            
            # 等待所有任務完成
            for future in player_futures:
                future.result()
            for future in court_futures:
                future.result()
        
        total_time = time.time() - total_start
        avg_player_time = sum(player_times) / len(player_times) if player_times else 0
        avg_court_time = sum(court_times) / len(court_times) if court_times else 0
        
        print(f"改進執行緒模式完成，總耗時: {total_time:.3f} 秒，平均每幀: {total_time/len(frames):.3f} 秒")
        print(f"球員檢測平均耗時: {avg_player_time:.3f} 秒")
        print(f"球場檢測平均耗時: {avg_court_time:.3f} 秒")
        
        return {
            "total_time": total_time,
            "frames": len(frames),
            "avg_player_time": avg_player_time,
            "avg_court_time": avg_court_time
        }
        
    # ==== 多進程池模式 ====    
    def run_mp_pool(self, video_path, max_frames):
        """多進程池模式：使用 concurrent.futures.ProcessPoolExecutor 進行並行處理"""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        # 讀取影片幀
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"無法開啟影片: {video_path}")
            return

        # 讀取到記憶體
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
            print("沒有讀取到有效的影片幀")
            return

        # 計算最佳進程數
        num_processes = max(2, min(mp.cpu_count(), 8))
        print(f"[多進程池] 使用 {num_processes} 個工作進程 (concurrent.futures)")

        total_start = time.time()
        
        # 準備參數
        player_args = [(frames[i], i, self.player_model_path, self.device) for i in range(len(frames))]
        court_args = [(frames[i], i, self.court_model_path, self.device) for i in range(len(frames))]
        
        # 使用 ProcessPoolExecutor 進行處理
        results = []
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # 提交所有球員偵測任務
            player_futures = [executor.submit(player_pool_worker, arg) for arg in player_args]
            # 提交所有球場偵測任務
            court_futures = [executor.submit(court_pool_worker, arg) for arg in court_args]
            
            # 收集所有結果
            for future in as_completed(player_futures + court_futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"處理任務時發生錯誤: {e}")
        
        # 處理結果
        player_times = {}
        court_times = {}
        
        for res_type, idx, _, t in results:
            if res_type == 'player':
                player_times[idx] = t
            else:
                court_times[idx] = t
        
        total_time = time.time() - total_start
        
        # 計算平均時間
        avg_player_time = sum(player_times.values()) / len(player_times) if player_times else 0
        avg_court_time = sum(court_times.values()) / len(court_times) if court_times else 0
        
        print(f"多進程池模式完成，總耗時: {total_time:.3f} 秒，平均每幀: {total_time/len(frames):.3f} 秒")
        print(f"球員偵測平均耗時: {avg_player_time:.3f} 秒")
        print(f"球場偵測平均耗時: {avg_court_time:.3f} 秒")
        
        return {
            "total_time": total_time,
            "frames": len(frames),
            "avg_player_time": avg_player_time,
            "avg_court_time": avg_court_time
        }
        
class YOLODetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO 多機制影片辨識")
        self.root.geometry("800x600")
        
        self.detector = None
        self.video_path = ""
        self.current_mode = MODE_SERIAL
        self.results = {}
        
        self._create_widgets()
        
    def analyze_performance(self, video_path, max_frames=100, step=10):
        """分析不同模式和不同影格數的性能表現
        
        Args:
            video_path: 影片路徑
            max_frames: 最大分析影格數
            step: 影格數的間隔
        
        Returns:
            pd.DataFrame: 包含性能數據的資料框
        """
        if not os.path.exists(video_path):
            messagebox.showerror("錯誤", f"影片檔案不存在: {video_path}")
            return None
            
        # 確保偵測器和模型已載入
        if self.detector is None:
            player_model_path = self.player_model_path_var.get()
            court_model_path = self.court_model_path_var.get()
            if not os.path.exists(player_model_path) or not os.path.exists(court_model_path):
                messagebox.showerror("錯誤", "模型檔案不存在")
                return None
            
            self.detector = YOLODetector(player_model_path, court_model_path)
            self.detector.load_models()
        
        # 準備資料框結構
        data = []
        modes = [
            (MODE_SERIAL, "序列模式"),
            # (MODE_THREADING, "多執行緒模式"),
            (MODE_IMPROVED_THREADING, "改進執行緒池模式"),
            # (MODE_MULTIPROCESS, "多進程模式"),
            (MODE_QUEUE, "佇列多進程模式"),
            (MODE_MP_POOL, "多進程池模式")
        ]
        
        # 更新狀態
        self.status_var.set("性能分析中...")
        self.root.update()
        
        # 執行分析前先進行一次模型預熱
        self._log_message("執行全局模型預熱...")
        # 先讀取一幀影像用於預熱
        cap = cv2.VideoCapture(video_path)
        ret, warmup_frame = cap.read()
        cap.release()
        
        if ret:
            # 預熱所有模型
            with torch.no_grad():
                for _ in range(5):  # 多做幾次確保完全預熱
                    self.detector.player_model.track(source=warmup_frame, conf=0.3, persist=True, tracker="bytetrack.yaml")
                    self.detector.court_model.track(source=warmup_frame, conf=0.25, persist=True, tracker="bytetrack.yaml")
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
            self._log_message("模型預熱完成")
        
        # 執行分析
        for mode_id, mode_name in modes:
            self._log_message(f"\n開始分析 {mode_name}")
            
            # 針對不同影格數進行測試
            for frames in range(step, max_frames + 1, step):
                self._log_message(f"測試 {mode_name} 處理 {frames} 幀...")
                  # 執行相應模式的檢測
                if mode_id == MODE_SERIAL:
                    result = self.detector.run_serial(video_path, frames)
                elif mode_id == MODE_THREADING:
                    result = self.detector.run_threading(video_path, frames)
                elif mode_id == MODE_MULTIPROCESS:
                    result = self.detector.run_multiprocess(video_path, frames)
                elif mode_id == MODE_QUEUE:
                    result = self.detector.run_queue_multiprocess(video_path, frames)
                elif mode_id == MODE_IMPROVED_THREADING:
                    result = self.detector.run_improved_threading(video_path, frames)
                elif mode_id == MODE_MP_POOL:
                    result = self.detector.run_mp_pool(video_path, frames)
                
                # 收集結果數據
                if result and "total_time" in result:
                    total_time = result["total_time"]
                    actual_frames = result.get("frames", 0)
                    avg_frame_time = total_time / actual_frames if actual_frames > 0 else 0
                    avg_player_time = result.get("avg_player_time", 0)
                    avg_court_time = result.get("avg_court_time", 0)
                    
                    # 將結果添加到數據列表
                    data.append({
                        "模式ID": mode_id,
                        "模式名稱": mode_name,
                        "影格數": actual_frames,
                        "總時間(秒)": total_time,
                        "平均每幀時間(秒)": avg_frame_time,
                        "平均球員偵測時間(秒)": avg_player_time,
                        "平均球場偵測時間(秒)": avg_court_time,
                        "FPS": 1 / avg_frame_time if avg_frame_time > 0 else 0
                    })
                    
                    self._log_message(f"  結果: 平均每幀 {avg_frame_time:.3f} 秒, FPS: {1/avg_frame_time:.2f}")
          # 創建 DataFrame
        df = pd.DataFrame(data)
        
        # 更新狀態
        self.status_var.set("性能分析完成")
        
        # 顯示分析結果圖表
        self._plot_performance_results(df)
        
        return df
    
    def _plot_performance_results(self, df):
        """繪製性能分析結果圖表
        
        Args:
            df: 包含性能數據的DataFrame
        """
        if df is None or df.empty:
            messagebox.showwarning("警告", "沒有可用的分析數據")
            return
              # 創建新視窗顯示圖表
        plot_window = tk.Toplevel(self.root)
        plot_window.title("性能分析結果")
        plot_window.geometry("1000x800")
          # 使用matplotlib繪製圖表
        
        # 確保在此處也設置中文字型
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 創建筆記本控制項顯示多個圖表
        notebook = ttk.Notebook(plot_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
          # 平均每幀處理時間對比
        frame_time_tab = ttk.Frame(notebook)
        notebook.add(frame_time_tab, text="平均每幀處理時間")
        
        fig1 = Figure(figsize=(10, 6), dpi=100)
        # 設置字型屬性
        fig1.patch.set_facecolor('#F0F0F0')  # 設置圖表背景色
        ax1 = fig1.add_subplot(111)
        
        for mode_id in df["模式ID"].unique():
            mode_data = df[df["模式ID"] == mode_id]
            mode_name = mode_data["模式名稱"].iloc[0]
            ax1.plot(mode_data["影格數"], mode_data["平均每幀時間(秒)"], marker='o', linewidth=2, label=mode_name)
            
        ax1.set_xlabel("影格數")
        ax1.set_ylabel("平均每幀處理時間 (秒)")
        ax1.set_title("不同模式下平均每幀處理時間對比")
        ax1.legend()
        ax1.grid(True)
        
        canvas1 = FigureCanvasTkAgg(fig1, frame_time_tab)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
          # FPS對比
        fps_tab = ttk.Frame(notebook)
        notebook.add(fps_tab, text="FPS對比")
        
        fig2 = Figure(figsize=(10, 6), dpi=100)
        fig2.patch.set_facecolor('#F0F0F0')  # 設置圖表背景色
        ax2 = fig2.add_subplot(111)
        
        for mode_id in df["模式ID"].unique():
            mode_data = df[df["模式ID"] == mode_id]
            mode_name = mode_data["模式名稱"].iloc[0]
            ax2.plot(mode_data["影格數"], mode_data["FPS"], marker='o', linewidth=2, label=mode_name)
            
        ax2.set_xlabel("影格數")
        ax2.set_ylabel("FPS")
        ax2.set_title("不同模式下FPS對比")
        ax2.legend()
        ax2.grid(True)
        
        canvas2 = FigureCanvasTkAgg(fig2, fps_tab)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
          # 球員與球場偵測時間對比
        detection_tab = ttk.Frame(notebook)
        notebook.add(detection_tab, text="偵測時間對比")
        
        fig3 = Figure(figsize=(10, 6), dpi=100)
        fig3.patch.set_facecolor('#F0F0F0')  # 設置圖表背景色
        ax3 = fig3.add_subplot(111)
        
        # 繪製按模式分組的長條圖
        mode_names = [name for _, name in df.groupby("模式ID")["模式名稱"].first().items()]
        x = np.arange(len(mode_names))
        width = 0.35
        
        # 計算每種模式的平均值
        avg_player_times = [group["平均球員偵測時間(秒)"].mean() for _, group in df.groupby("模式ID")]
        avg_court_times = [group["平均球場偵測時間(秒)"].mean() for _, group in df.groupby("模式ID")]
        
        rects1 = ax3.bar(x - width/2, avg_player_times, width, label='球員偵測')
        rects2 = ax3.bar(x + width/2, avg_court_times, width, label='球場偵測')
        
        ax3.set_xlabel('執行模式')
        ax3.set_ylabel('平均時間 (秒)')
        ax3.set_title('不同模式下球員和球場偵測平均時間')
        ax3.set_xticks(x)
        ax3.set_xticklabels(mode_names)
        ax3.legend()
        
        canvas3 = FigureCanvasTkAgg(fig3, detection_tab)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 數據表格顯示
        table_tab = ttk.Frame(notebook)
        notebook.add(table_tab, text="數據表格")
        
        # 創建捲動區域以顯示表格
        table_frame = ttk.Frame(table_tab)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 建立表格顯示
        table = ttk.Treeview(table_frame)
        
        # 定義列
        table['columns'] = ('模式名稱', '影格數', '總時間', '平均每幀時間', '平均球員時間', '平均球場時間', 'FPS')
        
        # 格式化列
        table.column('#0', width=0, stretch=tk.NO)
        table.column('模式名稱', anchor=tk.W, width=120)
        table.column('影格數', anchor=tk.CENTER, width=80)
        table.column('總時間', anchor=tk.CENTER, width=80)
        table.column('平均每幀時間', anchor=tk.CENTER, width=120)
        table.column('平均球員時間', anchor=tk.CENTER, width=120)
        table.column('平均球場時間', anchor=tk.CENTER, width=120)
        table.column('FPS', anchor=tk.CENTER, width=80)
        
        # 建立標題
        table.heading('#0', text='', anchor=tk.CENTER)
        table.heading('模式名稱', text='模式名稱', anchor=tk.CENTER)
        table.heading('影格數', text='影格數', anchor=tk.CENTER)
        table.heading('總時間', text='總時間(秒)', anchor=tk.CENTER)
        table.heading('平均每幀時間', text='平均每幀(秒)', anchor=tk.CENTER)
        table.heading('平均球員時間', text='球員偵測(秒)', anchor=tk.CENTER)
        table.heading('平均球場時間', text='球場偵測(秒)', anchor=tk.CENTER)
        table.heading('FPS', text='FPS', anchor=tk.CENTER)
        
        # 插入數據
        for i, row in df.iterrows():
            table.insert(parent='', index='end', iid=i, text='',
                       values=(row['模式名稱'], int(row['影格數']), f"{row['總時間(秒)']:.3f}", 
                               f"{row['平均每幀時間(秒)']:.3f}", f"{row['平均球員偵測時間(秒)']:.3f}", 
                               f"{row['平均球場偵測時間(秒)']:.3f}", f"{row['FPS']:.2f}"))
        
        # 加入捲動條
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=table.yview)
        table.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        table.pack(fill=tk.BOTH, expand=True)
        
        # 保存數據按鈕
        save_button = ttk.Button(plot_window, text="保存數據", 
                               command=lambda: self._save_performance_data(df))
        save_button.pack(pady=10)
    
    def _save_performance_data(self, df):
        """保存性能數據到CSV文件
        
        Args:
            df: 包含性能數據的DataFrame
        """
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="保存性能數據"
        )
        
        if filename:
            try:
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                messagebox.showinfo("成功", f"數據已保存至 {filename}")
            except Exception as e:
                messagebox.showerror("錯誤", f"保存數據時發生錯誤:\n{str(e)}")
                
    def _create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 設定區域
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
        ttk.Button(button_frame, text="開始處理", command=self._start_detection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清除結果", command=self._clear_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="性能分析", command=self._start_performance_analysis).pack(side=tk.LEFT, padx=5)
        
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
        """預載入模型，避免計時時包含載入時間"""
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
        
        # 初始化偵測器並預載入模型
        if self.detector is None:
            self.detector = YOLODetector(player_model_path, court_model_path)
        self.detector.load_models()
        
        self.status_var.set("模型已預載入")
        self._log_message("模型已預載入，現在可以開始測試了。")
    
    def _start_detection(self):
        """開始偵測處理"""
        # 檢查輸入
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
            
        # 更新狀態
        self.status_var.set("處理中...")
        self.root.update()
        
        # 設置偵測器
        if self.detector is None:
            self.detector = YOLODetector(player_model_path, court_model_path)
            # 先預載入模型
            self.detector.load_models()
        
        # 執行模式選擇
        mode = self.mode_var.get()
        max_frames = self.frames_var.get()
        
        # 開始非同步執行
        threading.Thread(target=self._run_detection, args=(mode, video_path, max_frames), daemon=True).start()
        
    def _run_detection(self, mode, video_path, max_frames):
        """執行選定的偵測模式"""
        try:
            # 清除之前的結果
            self.results = {}
            
            # 根據選擇的模式執行不同方法
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
                
            # 顯示完成訊息
            if "total_time" in self.results:
                total_time = self.results["total_time"]
                frames = self.results.get("frames", 0)
                avg_player_time = self.results.get("avg_player_time", 0)
                avg_court_time = self.results.get("avg_court_time", 0)
                
                self._log_message(f"\n===== {self._get_mode_name(mode)} 執行結果 =====")
                self._log_message(f"總處理時間: {total_time:.3f} 秒")
                self._log_message(f"處理總幀數: {frames} 幀")
                self._log_message(f"平均每幀耗時: {total_time/frames:.3f} 秒 (約 {frames/total_time:.2f} FPS)")
                self._log_message(f"平均球員偵測耗時: {avg_player_time:.3f} 秒")
                self._log_message(f"平均球場偵測耗時: {avg_court_time:.3f} 秒")
                
                # 計算並行效率
                serial_time = avg_player_time + avg_court_time
                parallel_time = total_time / frames
                speedup = serial_time / parallel_time if parallel_time > 0 else 0
            self.root.after(0, lambda: self.status_var.set("已完成"))
            # 增加：偵測完成後自動退出 mainloop，讓 VizTracer 能正常寫入 result.json
            #self.root.after(0, lambda: self.root.quit())

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
        print(message)  # 同時輸出到控制台
        
    def _clear_results(self):
        """清除結果顯示"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.results = {}
        self.status_var.set("就緒")
        
    def _start_performance_analysis(self):
        """開始性能分析"""
        # 檢查影片是否已選擇
        video_path = self.video_path_var.get()
        if not video_path:
            messagebox.showwarning("警告", "請選擇影片檔案")
            return
            
        if not os.path.exists(video_path):
            messagebox.showerror("錯誤", f"影片檔案不存在: {video_path}")
            return
            
        # 確認是否要開始分析 (這可能需要較長時間)
        confirm = messagebox.askyesno(
            "確認", 
            "性能分析將對所有模式進行測試，可能需要較長時間完成。\n是否繼續？"
        )
        if not confirm:
            return
            
        # 設定分析參數
        analysis_dialog = tk.Toplevel(self.root)
        analysis_dialog.title("性能分析設定")
        analysis_dialog.geometry("400x200")
        analysis_dialog.transient(self.root)
        analysis_dialog.grab_set()
        
        ttk.Label(analysis_dialog, text="最大影格數:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        max_frames_var = tk.IntVar(value=100)
        ttk.Entry(analysis_dialog, textvariable=max_frames_var, width=10).grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(analysis_dialog, text="影格增加步進:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        step_var = tk.IntVar(value=10)
        ttk.Entry(analysis_dialog, textvariable=step_var, width=10).grid(row=1, column=1, padx=10, pady=10)
        
        def start_analysis():
            analysis_dialog.destroy()
            max_frames = max_frames_var.get()
            step = step_var.get()
            
            if max_frames <= 0 or step <= 0:
                messagebox.showwarning("警告", "請輸入有效的最大影格數和步進值")
                return
                
            # 在新線程中啟動分析，避免UI凍結
            threading.Thread(
                target=lambda: self.analyze_performance(video_path, max_frames, step),
                daemon=True
            ).start()
            
        ttk.Button(analysis_dialog, text="開始分析", command=start_analysis).grid(row=2, column=0, padx=10, pady=20)
        ttk.Button(analysis_dialog, text="取消", command=analysis_dialog.destroy).grid(row=2, column=1, padx=10, pady=20)

if __name__ == "__main__":
    # 設定多處理程序啟動方法
    if sys.platform in ['win32', 'darwin']:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            # 若已經設定過，忽略錯誤
            pass
      # 設置 matplotlib 中文字型
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
      # 解析命令行參數，允許直接從命令行啟動特定模式
    parser = argparse.ArgumentParser(description='YOLO四種檢測機制')
    parser.add_argument('--mode', type=int, choices=[0, 1, 2, 3, 4, 5], 
                      help='偵測模式: 0=序列, 1=多執行緒, 2=多進程, 3=佇列多進程, 4=改進執行緒池, 5=多進程池')
    parser.add_argument('--frames', type=int, default=10, help='要處理的影格數')
    parser.add_argument('--video', type=str, help='影片檔案路徑')
    parser.add_argument('--player_model', type=str, default='best_demo_v2.pt', help='球員模型路徑')
    parser.add_argument('--court_model', type=str, default='Court_best.pt', help='球場模型路徑')
    
    args = parser.parse_args()
    
    # 啟動GUI
    root = tk.Tk()
    app = YOLODetectionApp(root)
    
    # 如果有命令行參數，自動填入GUI並啟動
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
        # 如果所有需要的參數都有，自動啟動
        if args.video and os.path.exists(args.video):
            root.after(1000, app._preload_models)  # 先預載入模型
            root.after(2000, app._start_detection)  # 然後開始偵測
    
    root.mainloop()
