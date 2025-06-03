import os
import sys
import threading
import multiprocessing as mp
import time
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import torch
from ultralytics import YOLO

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
MODE_SERIAL = 0       # 序列模式
MODE_THREADING = 1    # 多執行緒模式
MODE_MULTIPROCESS = 2 # 多進程模式
MODE_QUEUE = 3        # 佇列多進程模式

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
        end = time.time()
        print(f"模型載入完成，耗時: {end - start:.3f} 秒")
    
    def run_serial(self, video_path, max_frames):
        """序列模式：單線程執行所有影格辨識"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"無法開啟影片: {video_path}")
            return
        
        # 確保模型已載入
        #if self.player_model is None or self.court_model is None:
        self.load_models()
        
        # 只讀取第一幀
        # ret, frame = cap.read()
        # if not ret:
        #     print("無法讀取影片第一幀。")
        #     cap.release()
        #     return
        
        # cap.release()
        # 讀取多個不同的frame
        frames = []
        for i in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"影片只有 {i} 幀，重複最後一幀")
                frames.append(frames[-1].copy() if frames else None)
                break
            frames.append(frame.copy())
        cap.release()        
        
        frame_count = 0
        player_times = []
        court_times = []
        
        start_total = time.time()
        #while frame_count < max_frames:
        for frame_count, frame in enumerate(frames):
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
        
        # 計算包含第一筆的平均時間
        avg_player_time = sum(player_times) / len(player_times) if player_times else 0
        avg_court_time = sum(court_times) / len(court_times) if court_times else 0
        
        # 計算排除第一筆的平均時間
        avg_player_time_exclude_first = sum(player_times[1:]) / len(player_times[1:]) if len(player_times) > 1 else 0
        avg_court_time_exclude_first = sum(court_times[1:]) / len(court_times[1:]) if len(court_times) > 1 else 0
        
        # 第一筆單獨時間
        first_frame_player_time = player_times[0] if player_times else 0
        first_frame_court_time = court_times[0] if court_times else 0
        first_frame_total_time = first_frame_player_time + first_frame_court_time
        
        print(f"序列模式完成，總耗時: {total_time:.3f} 秒，平均每幀: {total_time/frame_count:.3f} 秒")
        print(f"球員偵測平均耗時: {avg_player_time:.3f} 秒")
        print(f"球場偵測平均耗時: {avg_court_time:.3f} 秒")
        
        return {
            "total_time": total_time,
            "frames": frame_count,
            "avg_player_time": avg_player_time,
            "avg_court_time": avg_court_time,
            "avg_player_time_exclude_first": avg_player_time_exclude_first,
            "avg_court_time_exclude_first": avg_court_time_exclude_first,
            "first_frame_player_time": first_frame_player_time,
            "first_frame_court_time": first_frame_court_time,
            "first_frame_total_time": first_frame_total_time
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
        # 讀取影片第一幀
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"無法開啟影片: {video_path}")
            return
            
        # 確保模型已載入
        if self.player_model is None or self.court_model is None:
            self.load_models()
        
        # 讀取第一幀
        # ret, first_frame = cap.read()
        # if not ret:
        #     print("無法讀取影片第一幀。")
        #     cap.release()
        #     return
            
        # cap.release()
        
        # # 將第一幀複製多次，模擬處理多幀
        # frames = [first_frame.copy() for _ in range(max_frames)]
        frames = []
        for i in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"影片只有 {i} 幀，重複最後一幀")
                frames.append(frames[-1].copy() if frames else None)
                break
            frames.append(frame.copy())
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
                start = time.time()
                p_res[0] = self.player_model.track(
                    source=frame, conf=0.3, persist=True, tracker="bytetrack.yaml"
                )
                p_time[0] = time.time() - start
            
            def _run_court():
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
        
        # 計算包含第一筆的平均時間
        avg_player_time = sum(player_times) / len(player_times) if player_times else 0
        avg_court_time  = sum(court_times)  / len(court_times)  if court_times  else 0
        
        # 計算排除第一筆的平均時間
        avg_player_time_exclude_first = sum(player_times[1:]) / len(player_times[1:]) if len(player_times) > 1 else 0
        avg_court_time_exclude_first = sum(court_times[1:]) / len(court_times[1:]) if len(court_times) > 1 else 0
        
        # 第一筆單獨時間
        first_frame_player_time = player_times[0] if player_times else 0
        first_frame_court_time = court_times[0] if court_times else 0
        first_frame_total_time = max(first_frame_player_time, first_frame_court_time)
        
        print(f"多執行緒模式完成，總耗時: {total_time:.3f} 秒，平均每幀: {total_time/len(frames):.3f} 秒")
        print(f"球員偵測平均耗時: {avg_player_time:.3f} 秒")
        print(f"球場偵測平均耗時: {avg_court_time:.3f} 秒")
        
        return {
            "total_time": total_time,
            "frames": len(frames),
            "avg_player_time": avg_player_time,
            "avg_court_time": avg_court_time,
            "avg_player_time_exclude_first": avg_player_time_exclude_first,
            "avg_court_time_exclude_first": avg_court_time_exclude_first,
            "first_frame_player_time": first_frame_player_time,
            "first_frame_court_time": first_frame_court_time,
            "first_frame_total_time": first_frame_total_time
        }

    # ==== 多進程模式 ====    
    def run_multiprocess(self, video_path, max_frames):
        """多進程模式：對每張影格啟動兩個 process（player、court）並行，完成後才處理下一張"""
        # 讀取影片第一幀
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"無法開啟影片: {video_path}")
            return

        # # 讀取第一幀
        # ret, first_frame = cap.read()
        # if not ret:
        #     print("無法讀取影片第一幀。")
        #     cap.release()
        #     return
            
        # cap.release()

        # # 將第一幀複製多次，模擬處理多幀
        # frames = [first_frame.copy() for _ in range(max_frames)]
        frames = []
        for i in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"影片只有 {i} 幀，重複最後一幀")
                frames.append(frames[-1].copy() if frames else None)
                break
            frames.append(frame.copy())
        cap.release()     
        # Windows/macOS 必要
        if sys.platform in ['win32', 'darwin']:
            mp.set_start_method("spawn", force=True)

        player_times = []
        court_times  = []
        process_overhead_times = []
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
            process_total_time = p_end - p_start
            
            player_times.append(pt)
            court_times.append(ct)
            process_overhead_times.append(process_total_time)

            print(f"[多進程] Frame-{i}: 球員 {pt:.3f}s, 球場 {ct:.3f}s, 含啟動/同步 {process_total_time:.3f}s")

        total_time = time.time() - total_start
        
        # 計算包含第一筆的平均時間
        avg_player_time = sum(player_times)/len(player_times) if player_times else 0
        avg_court_time  = sum(court_times) /len(court_times)  if court_times  else 0
        avg_process_overhead = sum(process_overhead_times) / len(process_overhead_times) if process_overhead_times else 0
        
        # 計算排除第一筆的平均時間
        avg_player_time_exclude_first = sum(player_times[1:]) / len(player_times[1:]) if len(player_times) > 1 else 0
        avg_court_time_exclude_first = sum(court_times[1:]) / len(court_times[1:]) if len(court_times) > 1 else 0
        
        # 第一筆單獨時間
        first_frame_player_time = player_times[0] if player_times else 0
        first_frame_court_time = court_times[0] if court_times else 0
        first_frame_total_time = process_overhead_times[0] if process_overhead_times else 0
        
        # 計算開銷相關指標
        avg_pure_inference = max(avg_player_time, avg_court_time)
        overhead_time = avg_process_overhead - avg_pure_inference
        overhead_ratio = (overhead_time / avg_process_overhead * 100) if avg_process_overhead > 0 else 0

        print(f"多進程模式完成，總耗時: {total_time:.3f} 秒，平均每幀: {total_time/len(frames):.3f} 秒")
        print(f"球員偵測平均耗時: {avg_player_time:.3f} 秒")
        print(f"球場偵測平均耗時: {avg_court_time:.3f} 秒")

        return {
            "total_time": total_time,
            "frames": len(frames),
            "avg_player_time": avg_player_time,
            "avg_court_time": avg_court_time,
            "avg_player_time_exclude_first": avg_player_time_exclude_first,
            "avg_court_time_exclude_first": avg_court_time_exclude_first,
            "first_frame_player_time": first_frame_player_time,
            "first_frame_court_time": first_frame_court_time,
            "first_frame_total_time": first_frame_total_time,
            "avg_process_overhead": avg_process_overhead,
            "overhead_ratio": overhead_ratio
        }

    def run_queue_multiprocess(self, video_path, max_frames):
        """佇列多進程模式：使用任務佇列的生產者消費者模式，同步處理同一frame - 完整開銷測量版本"""
        
        # 1. 進程啟動時間測量
        total_start_time = time.time()
        process_start_time = time.time()
        
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
        
        process_startup_time = time.time() - process_start_time
        print(f"進程啟動時間: {process_startup_time:.3f} 秒")
        
        # 2. 讀取影片幀
        file_read_start = time.time()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"無法開啟影片: {video_path}")
            return
        
        frames = []
        for i in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                print(f"影片只有 {i} 幀，重複最後一幀")
                frames.append(frames[-1].copy() if frames else None)
                break
            frames.append(frame.copy())
        cap.release()
        
        file_read_time = time.time() - file_read_start
        print(f"影片讀取時間: {file_read_time:.3f} 秒")
        
        # 3. 逐幀處理和詳細開銷測量
        player_times = []
        court_times = []
        data_transfer_times = []
        wait_times = []
        frame_overhead_times = []
        frame_total_times = []
        
        processing_start_time = time.time()
        
        for frame_idx, frame in enumerate(frames):
            frame_start = time.time()
            
            # 3.1 數據傳輸時間
            data_transfer_start = time.time()
            player_task_queue.put((frame_idx, frame.copy()))
            court_task_queue.put((frame_idx, frame.copy()))
            data_transfer_time = time.time() - data_transfer_start
            data_transfer_times.append(data_transfer_time)
            
            # 3.2 等待結果時間
            wait_start = time.time()
            frame_results = {}
            got_results = 0
            while got_results < 2:
                typ, idx, _, t = result_queue.get()
                if idx == frame_idx:
                    frame_results[typ] = t
                    got_results += 1
            wait_time = time.time() - wait_start
            wait_times.append(wait_time)
            
            frame_total_time = time.time() - frame_start
            frame_total_times.append(frame_total_time)
            
            # 3.3 計算各種時間
            player_inference_time = frame_results['player']
            court_inference_time = frame_results['court']
            pure_inference_time = max(player_inference_time, court_inference_time)
            frame_overhead = frame_total_time - pure_inference_time
            frame_overhead_times.append(frame_overhead)
            
            player_times.append(player_inference_time)
            court_times.append(court_inference_time)
            
            print(f"[佇列多進程] Frame-{frame_idx} 時間軸:")
            print(f"  1. 數據傳輸: {data_transfer_time:.3f}s")
            print(f"  2. 並行推理: max({player_inference_time:.3f}s球員, {court_inference_time:.3f}s球場) = {pure_inference_time:.3f}s")
            print(f"  3. 等待同步: {wait_time:.3f}s")
            print(f"  → 幀總時間: {frame_total_time:.3f}s")
            print(f"  → 理論時間: {data_transfer_time + pure_inference_time:.3f}s")
            print(f"  → 額外開銷: {frame_overhead:.3f}s ({frame_overhead/frame_total_time*100:.1f}%)")
        
        processing_time = time.time() - processing_start_time
        
        # 4. 進程清理時間
        cleanup_start = time.time()
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
        
        cleanup_time = time.time() - cleanup_start
        print(f"進程清理時間: {cleanup_time:.3f} 秒")
        
        total_time = time.time() - total_start_time
        
        # 5. 計算各種統計數據
        # 5.1 推理時間統計
        avg_player_time = sum(player_times) / len(player_times) if player_times else 0
        avg_court_time = sum(court_times) / len(court_times) if court_times else 0
        avg_player_time_exclude_first = sum(player_times[1:]) / len(player_times[1:]) if len(player_times) > 1 else 0
        avg_court_time_exclude_first = sum(court_times[1:]) / len(court_times[1:]) if len(court_times) > 1 else 0
        
        # 5.2 開銷時間統計
        avg_data_transfer_time = sum(data_transfer_times) / len(data_transfer_times) if data_transfer_times else 0
        avg_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0
        avg_frame_overhead = sum(frame_overhead_times) / len(frame_overhead_times) if frame_overhead_times else 0
        avg_frame_total_time = sum(frame_total_times) / len(frame_total_times) if frame_total_times else 0
        
        # 5.3 第一筆時間
        first_frame_player_time = player_times[0] if player_times else 0
        first_frame_court_time = court_times[0] if court_times else 0
        first_frame_total_time = frame_total_times[0] if frame_total_times else 0
        first_frame_overhead = frame_overhead_times[0] if frame_overhead_times else 0
        
        # 5.4 總開銷分析
        total_pure_inference_time = sum(max(player_times[i], court_times[i]) for i in range(len(player_times)))
        total_overhead_time = total_time - total_pure_inference_time
        total_data_transfer_time = sum(data_transfer_times)
        total_wait_time = sum(wait_times)
        
        # 6. 詳細結果輸出
        print(f"\n===== 佇列多進程模式 - 詳細開銷分析 =====")
        print(f"總處理時間: {total_time:.3f} 秒")
        print(f"處理總幀數: {max_frames} 幀")
        
        print(f"\n時間分解:")
        print(f"  進程啟動時間: {process_startup_time:.3f} 秒 ({process_startup_time/total_time*100:.1f}%)")
        print(f"  影片讀取時間: {file_read_time:.3f} 秒 ({file_read_time/total_time*100:.1f}%)")
        print(f"  實際處理時間: {processing_time:.3f} 秒 ({processing_time/total_time*100:.1f}%)")
        print(f"  進程清理時間: {cleanup_time:.3f} 秒 ({cleanup_time/total_time*100:.1f}%)")
        
        print(f"\n推理時間統計:")
        print(f"  總純推理時間: {total_pure_inference_time:.3f} 秒")
        print(f"  球員偵測平均: {avg_player_time:.3f} 秒")
        print(f"  球場偵測平均: {avg_court_time:.3f} 秒")
        
        print(f"\n開銷時間統計:")
        print(f"  總開銷時間: {total_overhead_time:.3f} 秒 ({total_overhead_time/total_time*100:.1f}%)")
        print(f"  總數據傳輸時間: {total_data_transfer_time:.3f} 秒 ({total_data_transfer_time/total_time*100:.1f}%)")
        print(f"  總等待同步時間: {total_wait_time:.3f} 秒 ({total_wait_time/total_time*100:.1f}%)")
        print(f"  平均每幀開銷: {avg_frame_overhead:.3f} 秒")
        print(f"  平均每幀開銷占比: {avg_frame_overhead/avg_frame_total_time*100:.1f}%")
        
        print(f"\n第一筆vs後續比較:")
        if len(player_times) > 1:
            print(f"  第一筆總開銷: {first_frame_overhead:.3f} 秒")
            avg_overhead_exclude_first = sum(frame_overhead_times[1:]) / len(frame_overhead_times[1:]) if len(frame_overhead_times) > 1 else 0
            print(f"  後續平均開銷: {avg_overhead_exclude_first:.3f} 秒")
            if avg_overhead_exclude_first > 0:
                overhead_ratio = first_frame_overhead / avg_overhead_exclude_first
                print(f"  第一筆開銷倍數: {overhead_ratio:.2f}x")
        
        # 7. 返回完整結果
        return {
            "total_time": total_time,
            "frames": max_frames,
            
            # 推理時間
            "avg_player_time": avg_player_time,
            "avg_court_time": avg_court_time,
            "avg_player_time_exclude_first": avg_player_time_exclude_first,
            "avg_court_time_exclude_first": avg_court_time_exclude_first,
            "first_frame_player_time": first_frame_player_time,
            "first_frame_court_time": first_frame_court_time,
            "first_frame_total_time": first_frame_total_time,
            "total_pure_inference_time": total_pure_inference_time,
            
            # 開銷分析
            "process_startup_time": process_startup_time,
            "file_read_time": file_read_time,
            "processing_time": processing_time,
            "cleanup_time": cleanup_time,
            "total_overhead_time": total_overhead_time,
            "total_data_transfer_time": total_data_transfer_time,
            "total_wait_time": total_wait_time,
            "avg_data_transfer_time": avg_data_transfer_time,
            "avg_wait_time": avg_wait_time,
            "avg_frame_overhead": avg_frame_overhead,
            "avg_frame_total_time": avg_frame_total_time,
            "first_frame_overhead": first_frame_overhead,
            
            # 百分比
            "overhead_percentage": total_overhead_time/total_time*100,
            "startup_percentage": process_startup_time/total_time*100,
            "cleanup_percentage": cleanup_time/total_time*100,
            "data_transfer_percentage": total_data_transfer_time/total_time*100,
            "wait_time_percentage": total_wait_time/total_time*100,
            
            # 原有數據保持兼容性
            "player_times": player_times,
            "court_times": court_times,
            "data_transfer_times": data_transfer_times,
            "wait_times": wait_times,
            "frame_overhead_times": frame_overhead_times,
            "frame_total_times": frame_total_times
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
                 ("佇列多進程模式", MODE_QUEUE)]
                 
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
            else:
                messagebox.showerror("錯誤", f"未知的執行模式: {mode}")
                return
                
            # 顯示完成訊息
            if "total_time" in self.results:
                self._display_results(mode)
                
            self.root.after(0, lambda: self.status_var.set("已完成"))

        except Exception as e:
            self._log_message(f"錯誤: {str(e)}")
            import traceback
            self._log_message(traceback.format_exc())
            self.root.after(0, lambda: self.status_var.set("發生錯誤"))
    
    def _display_results(self, mode):
        """顯示詳細的執行結果"""
        total_time = self.results["total_time"]
        frames = self.results.get("frames", 0)
        avg_player_time = self.results.get("avg_player_time", 0)
        avg_court_time = self.results.get("avg_court_time", 0)
        avg_player_time_exclude_first = self.results.get("avg_player_time_exclude_first", 0)
        avg_court_time_exclude_first = self.results.get("avg_court_time_exclude_first", 0)
        first_frame_player_time = self.results.get("first_frame_player_time", 0)
        first_frame_court_time = self.results.get("first_frame_court_time", 0)
        first_frame_total_time = self.results.get("first_frame_total_time", 0)
        
        self._log_message(f"\n===== {self._get_mode_name(mode)} 執行結果 =====")
        self._log_message(f"總處理時間: {total_time:.3f} 秒")
        self._log_message(f"處理總幀數: {frames} 幀")
        
        # 第一筆時間記錄
        self._log_message(f"\n第一筆時間記錄:")
        self._log_message(f"  球員偵測: {first_frame_player_time:.6f} 秒")
        self._log_message(f"  球場偵測: {first_frame_court_time:.6f} 秒")
        
        # 包含第一筆的平均時間
        self._log_message(f"\n包含第一筆的平均時間:")
        self._log_message(f"  球員偵測平均: {avg_player_time:.6f} 秒")
        self._log_message(f"  球場偵測平均: {avg_court_time:.6f} 秒")
        
        # 排除第一筆的平均時間
        if frames > 1:
            self._log_message(f"\n排除第一筆的平均時間:")
            self._log_message(f"  球員偵測平均: {avg_player_time_exclude_first:.6f} 秒")
            self._log_message(f"  球場偵測平均: {avg_court_time_exclude_first:.6f} 秒")
            
            # 第一筆影響倍數
            if avg_player_time_exclude_first > 0:
                player_ratio = first_frame_player_time / avg_player_time_exclude_first
                self._log_message(f"  第一筆球員偵測倍數: {player_ratio:.2f}x")
            if avg_court_time_exclude_first > 0:
                court_ratio = first_frame_court_time / avg_court_time_exclude_first
                self._log_message(f"  第一筆球場偵測倍數: {court_ratio:.2f}x")
        
        # Mode 2 特殊的開銷分析
        if mode == MODE_MULTIPROCESS and "avg_process_overhead" in self.results:
            avg_process_overhead = self.results["avg_process_overhead"]
            overhead_ratio = self.results["overhead_ratio"]
            self._log_message(f"\n多進程開銷分析:")
            self._log_message(f"  平均進程總開銷時間: {avg_process_overhead:.6f} 秒")
            self._log_message(f"  開銷占比: {overhead_ratio:.1f}%")
        
        # FPS 計算
        self._log_message(f"\n===== FPS 分析 =====")
        
        # 包含第一筆的FPS
        if mode == MODE_SERIAL:
            theoretical_fps_with_first = 1 / (avg_player_time + avg_court_time) if (avg_player_time + avg_court_time) > 0 else 0
            self._log_message(f"理論最大 FPS (序列，包含第一筆): {theoretical_fps_with_first:.2f}")
        else:
            theoretical_fps_with_first = 1 / max(avg_player_time, avg_court_time) if max(avg_player_time, avg_court_time) > 0 else 0
            self._log_message(f"理論最大 FPS (並行，包含第一筆): {theoretical_fps_with_first:.2f}")
        
        # 排除第一筆的FPS
        if frames > 1:
            if mode == MODE_SERIAL:
                theoretical_fps_exclude_first = 1 / (avg_player_time_exclude_first + avg_court_time_exclude_first) if (avg_player_time_exclude_first + avg_court_time_exclude_first) > 0 else 0
                self._log_message(f"理論最大 FPS (序列，排除第一筆): {theoretical_fps_exclude_first:.2f}")
            else:
                theoretical_fps_exclude_first = 1 / max(avg_player_time_exclude_first, avg_court_time_exclude_first) if max(avg_player_time_exclude_first, avg_court_time_exclude_first) > 0 else 0
                self._log_message(f"理論最大 FPS (並行，排除第一筆): {theoretical_fps_exclude_first:.2f}")
        
        # 實際處理頻率
        processing_rate_with_first = frames / total_time if total_time > 0 else 0
        self._log_message(f"實際處理頻率 (包含第一筆): {processing_rate_with_first:.2f} 次/秒")
        
        if frames > 1:
            exclude_first_total_time = total_time - first_frame_total_time
            processing_rate_exclude_first = (frames - 1) / exclude_first_total_time if exclude_first_total_time > 0 else 0
            self._log_message(f"實際處理頻率 (排除第一筆): {processing_rate_exclude_first:.2f} 次/秒")
        
        # 計算並行效率
        if mode != MODE_SERIAL and frames > 1:
            serial_time = avg_player_time_exclude_first + avg_court_time_exclude_first
            parallel_time = max(avg_player_time_exclude_first, avg_court_time_exclude_first)
            if parallel_time > 0:
                speedup = serial_time / parallel_time
                efficiency = speedup / 2.0 * 100  # 兩個模型的並行效率
                self._log_message(f"並行加速比 (排除第一筆): {speedup:.2f}x")
                self._log_message(f"並行效率 (排除第一筆): {efficiency:.1f}%")
            
    def _get_mode_name(self, mode):
        """根據模式ID獲取模式名稱"""
        mode_names = {
            MODE_SERIAL: "序列模式",
            MODE_THREADING: "多執行緒模式",
            MODE_MULTIPROCESS: "多進程模式",
            MODE_QUEUE: "佇列多進程模式"
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

if __name__ == "__main__":
    # 設定多處理程序啟動方法
    if sys.platform in ['win32', 'darwin']:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            # 若已經設定過，忽略錯誤
            pass
    
    # 解析命令行參數，允許直接從命令行啟動特定模式
    import argparse
    parser = argparse.ArgumentParser(description='YOLO四種檢測機制')
    parser.add_argument('--mode', type=int, choices=[0, 1, 2, 3], help='偵測模式: 0=序列, 1=多執行緒, 2=多進程, 3=佇列多進程')
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