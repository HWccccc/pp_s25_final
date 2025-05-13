import os
from models.basketball_tracker import BasketballTracker
from ui.gradio_interface import GradioInterface

def main():
    # 確保數據文件夾存在
    data_folder = 'big3_data'
    if not os.path.exists(data_folder):
        print(f"警告: 數據文件夾 '{data_folder}' 不存在。請確保數據文件夾已正確設置。")
        os.makedirs(data_folder)

    try:
        # 初始化追蹤器
        tracker = BasketballTracker(
            player_model_path='best_demo_v2.pt',
            court_model_path='Court_best.pt',
            data_folder=data_folder
        )
        
        # 設置球場參考圖
        tracker.set_court_reference('court_pic.jpg')

        # 創建並啟動 Gradio 介面
        interface = GradioInterface(tracker)
        interface.launch()

    except Exception as e:
        print(f"程序運行時發生錯誤: {e}")

if __name__ == "__main__":
    main()