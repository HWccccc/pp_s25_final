import multiprocessing
import time
import os # For VizTracer process naming

# --- Helper function for VizTracer process naming ---
def set_process_name(name):
    if hasattr(os, 'sched_setaffinity'): # Check for Linux-like system for better naming
        try:
            from ctypes import cdll, byref, create_string_buffer
            libc = cdll.LoadLibrary('libc.so.6')
            buff = create_string_buffer(len(name) + 1)
            buff.value = name.encode('utf-8')
            libc.prctl(15, byref(buff), 0, 0, 0)
        except Exception:
            pass # Silently ignore if prctl is not available or fails
    # For other systems or as a fallback, multiprocessing.current_process().name works
    # but VizTracer might group them less intuitively without explicit naming if possible.

# --- 任務處理函式 ---
def yolo_detection_task(item_id, data):
    # set_process_name(f"YOLO_{item_id}") # VizTracer might benefit
    print(f"[PID:{os.getpid()}] YOLO (T1) starting for {item_id}")
    time.sleep(2) # 模擬 YOLO 處理時間
    result = f"yolo_result_for_{item_id}"
    print(f"[PID:{os.getpid()}] YOLO (T1) finished for {item_id}")
    return item_id, result

def ocr_detection_task(item_id, yolo_data):
    # set_process_name(f"OCR_{item_id}")
    print(f"[PID:{os.getpid()}] OCR (T2) starting for {item_id} with {yolo_data}")
    time.sleep(3) # 模擬 OCR 處理時間
    result = f"ocr_result_for_{item_id}"
    print(f"[PID:{os.getpid()}] OCR (T2) finished for {item_id}")
    return item_id, result

def data_process_task(item_id, ocr_data, previous_item_done_event=None):
    # set_process_name(f"DataProc_{item_id}")
    if previous_item_done_event:
        print(f"[PID:{os.getpid()}] DataProcess (T3) for {item_id} WAITING for previous item's T3...")
        previous_item_done_event.wait() # 等待前一個物件的 T3 完成
        print(f"[PID:{os.getpid()}] DataProcess (T3) for {item_id} RESUMING.")

    print(f"[PID:{os.getpid()}] DataProcess (T3) starting for {item_id} with {ocr_data}")
    time.sleep(1) # 模擬 Data Process 處理時間
    result = f"final_data_for_{item_id}"
    print(f"[PID:{os.getpid()}] DataProcess (T3) finished for {item_id}")
    return item_id, result

# --- Worker Process Functions ---
def yolo_worker(input_q, output_q):
    set_process_name("YOLO_Worker")
    print(f"[PID:{os.getpid()}] YOLO_Worker starting...")
    for item_id, data in iter(input_q.get, (None, None)): # (None,None) is a sentinel
        item_id_res, yolo_res = yolo_detection_task(item_id, data)
        output_q.put((item_id_res, yolo_res))
    output_q.put((None, None)) # Propagate sentinel
    print(f"[PID:{os.getpid()}] YOLO_Worker shutting down.")

def ocr_worker(input_q, output_q):
    set_process_name("OCR_Worker")
    print(f"[PID:{os.getpid()}] OCR_Worker starting...")
    for item_id, yolo_data in iter(input_q.get, (None, None)):
        item_id_res, ocr_res = ocr_detection_task(item_id, yolo_data)
        output_q.put((item_id_res, ocr_res))
    output_q.put((None, None)) # Propagate sentinel
    print(f"[PID:{os.getpid()}] OCR_Worker shutting down.")

def data_process_worker(input_q, output_q, item_completion_events):
    set_process_name("DataProcess_Worker")
    print(f"[PID:{os.getpid()}] DataProcess_Worker starting...")
    # item_completion_events is a dictionary like:
    # { 'item_A_id': (None, current_item_event_to_set),
    #   'item_B_id': (event_for_A_to_wait_on, current_item_event_to_set),
    #   'item_C_id': (event_for_B_to_wait_on, current_item_event_to_set) }

    for item_id, ocr_data in iter(input_q.get, (None, None)):
        previous_event_to_wait_for, current_event_to_set = item_completion_events.get(item_id, (None, None))

        item_id_res, final_res = data_process_task(item_id, ocr_data, previous_event_to_wait_for)
        output_q.put((item_id_res, final_res))

        if current_event_to_set:
            current_event_to_set.set() # 通知下一個物件的 T3 可以開始了 (如果它正在等待)
            print(f"[PID:{os.getpid()}] DataProcess_Worker set event for {item_id}")

    # No sentinel propagation for output_q needed from here if it's the final output
    print(f"[PID:{os.getpid()}] DataProcess_Worker shutting down.")

# --- 主流程 ---
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True) # Recommended for consistency across platforms

    # 1. 建立佇列
    ui_to_yolo_q = multiprocessing.Queue()
    yolo_to_ocr_q = multiprocessing.Queue()
    ocr_to_data_q = multiprocessing.Queue()
    final_output_q = multiprocessing.Queue() # 從 DataProcess 輸出結果

    # 2. 準備任務資料 (模擬 UI 輸入) - 修正為與 sequential 版本一致
    items_to_process = [
        ("frame_1", "raw_data_1"),
        ("frame_2", "raw_data_2"),
        ("frame_3", "raw_data_3"),
    ]

    all_final_results = []
    print("Starting pipeline processing...\n")
    
    # 記錄開始時間
    start_time = time.time()

    # 3. 建立同步事件 (用於 T3 的相依性)
    # T3 for frame_2 waits for T3 for frame_1 to complete.
    # T3 for frame_3 waits for T3 for frame_2 to complete.
    # `item_completion_events` maps an item_id to a tuple:
    # (event_this_item_waits_for, event_this_item_will_set_upon_completion)
    item_s3_events = {}
    previous_item_event = None
    for item_id, _ in items_to_process:
        current_item_event = multiprocessing.Event()
        item_s3_events[item_id] = (previous_item_event, current_item_event)
        previous_item_event = current_item_event

    # 4. 建立並啟動 Worker Processes
    yolo_proc = multiprocessing.Process(target=yolo_worker, args=(ui_to_yolo_q, yolo_to_ocr_q))
    ocr_proc = multiprocessing.Process(target=ocr_worker, args=(yolo_to_ocr_q, ocr_to_data_q))
    data_proc = multiprocessing.Process(target=data_process_worker, args=(ocr_to_data_q, final_output_q, item_s3_events))

    workers = [yolo_proc, ocr_proc, data_proc]
    for w in workers:
        w.start()

    # 5. 從 UI (模擬) 將任務放入第一個佇列
    for item_id, data in items_to_process:
        print(f"--- Processing {item_id} ---")
        print(f"[MainProc] Sending {item_id} to pipeline.")
        ui_to_yolo_q.put((item_id, data))
        time.sleep(0.5) # 稍微錯開輸入，以便觀察 pipeline 效果

    # 6. 發送 sentinel (結束信號) 到第一個 worker，使其可以逐級傳遞
    ui_to_yolo_q.put((None, None))

    # 7. 收集最終結果 (可選，依 UI 需求)
    results_count = 0
    while results_count < len(items_to_process):
        try:
            item_id, final_result = final_output_q.get(timeout=20) # 增加 timeout 時間
            if item_id is None: # Should not happen here if data_proc doesn't send sentinel
                break
            print(f"[MainProc] Received final result for {item_id}: {final_result}")
            print(f"--- Finished processing {item_id} ---\n")
            all_final_results.append((item_id, final_result))
            results_count += 1
        except multiprocessing.queues.Empty:
            print("[MainProc] Timeout waiting for results. Processes might be stuck or finished.")
            break

    # 8. 等待所有 worker process 結束
    for w in workers:
        w.join(timeout=15) # 增加 timeout 時間
        if w.is_alive():
            print(f"[MainProc] Worker {w.name} (PID:{w.pid}) did not terminate gracefully. Terminating.")
            w.terminate() # Force terminate if join timed out
            w.join()

    # 9. 顯示所有結果 - 與 sequential 版本一致的格式
    print("\nAll items processed in pipeline. Final results:")
    for item_id, result in all_final_results:
        print(f"  {item_id}: {result}")
    
    # 計算實際處理時間
    end_time = time.time()
    actual_time = end_time - start_time
    
    print(f"\nActual pipeline processing time: {actual_time:.2f} seconds.")
    print(f"Estimated pipeline processing benefits: Overlapped execution of T1, T2, T3 stages.")
    print("[MainProc] All tasks completed and processes joined.")