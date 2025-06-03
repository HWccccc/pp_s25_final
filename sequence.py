import time
import os

# --- 任務處理函式 (與之前相同，但 data_process_task 不再需要 event) ---
def yolo_detection_task(item_id, data):
    print(f"[PID:{os.getpid()}] YOLO (T1) starting for {item_id}")
    time.sleep(2) # 模擬 YOLO 處理時間
    result = f"yolo_result_for_{item_id}"
    print(f"[PID:{os.getpid()}] YOLO (T1) finished for {item_id}")
    return item_id, result

def ocr_detection_task(item_id, yolo_data):
    print(f"[PID:{os.getpid()}] OCR (T2) starting for {item_id} with {yolo_data}")
    time.sleep(3) # 模擬 OCR 處理時間
    result = f"ocr_result_for_{item_id}"
    print(f"[PID:{os.getpid()}] OCR (T2) finished for {item_id}")
    return item_id, result

def data_process_task(item_id, ocr_data):
    # 注意：循序版本中，B3 等待 A3 的條件是隱含滿足的，
    # 因為 B 的所有處理 (B1, B2, B3) 都會在 A 的所有處理 (A1, A2, A3) 完成後才開始。
    print(f"[PID:{os.getpid()}] DataProcess (T3) starting for {item_id} with {ocr_data}")
    time.sleep(1) # 模擬 Data Process 處理時間
    result = f"final_data_for_{item_id}"
    print(f"[PID:{os.getpid()}] DataProcess (T3) finished for {item_id}")
    return item_id, result

# --- 主流程 (循序執行) ---
if __name__ == "__main__":
    # 1. 準備任務資料 (模擬 UI 輸入)
    items_to_process = [
        ("frame_1", "raw_data_1"),
        ("frame_2", "raw_data_2"),
        ("frame_3", "raw_data_3"),
    ]

    all_final_results = []

    print("Starting sequential processing...\n")

    # 2. 循序處理每個物件
    for item_id, data in items_to_process:
        print(f"--- Processing {item_id} ---")

        # 階段 1: YOLO Detection
        _, yolo_result = yolo_detection_task(item_id, data)

        # 階段 2: OCR Detection
        _, ocr_result = ocr_detection_task(item_id, yolo_result)

        # 階段 3: Data Process
        _, final_result = data_process_task(item_id, ocr_result)

        all_final_results.append((item_id, final_result))
        print(f"--- Finished processing {item_id} ---\n")

    # 3. 顯示所有結果
    print("\nAll items processed sequentially. Final results:")
    for item_id, result in all_final_results:
        print(f"  {item_id}: {result}")

    total_time_estimate = len(items_to_process) * (2 + 3 + 1) # 每個物件的處理時間總和
    print(f"\nEstimated total sequential processing time: {total_time_estimate} seconds.")