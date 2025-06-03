import os
import sys
from models.basketball_tracker import BasketballTracker
from ui.gradio_interface import GradioInterface
import multiprocessing as mp

def main():
    # 確保數據文件夾存在
    data_folder = 'big3_data'
    if not os.path.exists(data_folder):
        print(f"警告: 數據文件夾 '{data_folder}' 不存在。請確保數據文件夾已正確設置。")
        os.makedirs(data_folder)

    try:
        # 初始化追蹤器（新增 max_parallel_frames 參數）
        tracker = BasketballTracker(
            player_model_path='best_demo_v2.pt',
            court_model_path='Court_best.pt',
            data_folder=data_folder,
            max_parallel_frames=3  # 新增：可調整的並行幀數（建議 2-4）
        )
        
        # 設置球場參考圖
        tracker.set_court_reference('court_pic.jpg')

        # 創建並啟動 Gradio 介面
        interface = GradioInterface(tracker)
        
        # 顯示 pipeline 狀態資訊
        print(f"Pipeline 並行設定：最大 {tracker.max_parallel_frames} 個幀同時處理")
        print("啟動 Gradio 介面...")
        
        interface.launch()

    except Exception as e:
        print(f"程序運行時發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 設置多進程啟動方法（重要：避免衝突）
    if sys.platform.startswith('win') or sys.platform.startswith('darwin'):
        try: 
            mp.set_start_method('spawn', force=True)
            print("設置多進程模式：spawn")
        except RuntimeError: 
            print("警告: MP 'spawn' 設置失敗，使用預設模式")
    
    main()