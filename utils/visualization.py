import cv2
import numpy as np
import pandas as pd

class Visualizer:
    @staticmethod
    def create_player_table(player_data):
        """創建球員資訊HTML表格"""
        if not player_data:
            return "未檢測到球員"

        html_template = """
        <meta charset="UTF-8">
        <style>
            table {
                width: auto;
                border-collapse: collapse;
            }
            th, td {
                padding: 5px;
                text-align: center;
                border: 1px solid #ddd;
            }
            .stamina-bar {
                width: 100px;
                background-color: #ddd;
            }
            .stamina-fill {
                height: 20px;
                background-color: green;
                text-align: center;
                color: white;
            }
        </style>
        """

        df = pd.DataFrame(player_data)
        
        # 處理圖片顯示
        if 'Pic' in df.columns:
            df['Pic'] = df['Pic'].apply(
                lambda x: f'<img src="{x}" style="width:50px; height:50px;">' if x else '')
        
        # 處理體力條顯示
        if '體力' in df.columns:
            df['體力'] = df['體力'].apply(
                lambda x: f'<div class="stamina-bar"><div class="stamina-fill" style="width:{x}%">{x}%</div></div>')

        # 產生HTML表格
        table_html = df.to_html(escape=False, index=False)
        return html_template + table_html

    @staticmethod
    def draw_tracking_info(image, detections, show_ids=True, show_boxes=True):
        """在圖像上繪製追蹤資訊"""
        display_image = image.copy()
        
        for det in detections:
            if 'box' not in det:
                continue
                
            x1, y1, x2, y2 = map(int, det['box'])
            color = det.get('color', (0, 255, 0))
            
            if show_boxes:
                cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
            
            if show_ids and 'id' in det:
                label = f"ID: {det['id']}"
                if 'class' in det:
                    label = f"{det['class']} {label}"
                
                cv2.putText(display_image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return display_image

    @staticmethod
    def draw_court_analysis(image, court_data):
        """在圖像上繪製場地分析資訊"""
        display_image = image.copy()
        
        # 繪製禁區
        if 'RA' in court_data:
            points = np.array(court_data['RA']['points'], dtype=np.int32)
            cv2.polylines(display_image, [points], True, (255, 0, 0), 2)
        
        # 繪製罰球區
        if 'FTA' in court_data:
            points = np.array(court_data['FTA']['points'], dtype=np.int32)
            cv2.polylines(display_image, [points], True, (0, 255, 0), 2)
        
        return display_image

    @staticmethod
    def create_status_display(frame_count, fps, detections_count):
        """創建狀態顯示"""
        status_text = f"""
        幀數: {frame_count}
        FPS: {fps:.2f}
        檢測物件數: {detections_count}
        """
        return status_text

    @staticmethod
    def draw_path_history(image, paths, max_history=50):
        """繪製移動路徑歷史"""
        display_image = image.copy()
        
        for track_id, path in paths.items():
            # 只保留最近的 max_history 個點
            points = path[-max_history:]
            if len(points) < 2:
                continue
            
            # 生成該 track_id 的固定顏色
            np.random.seed(int(track_id))
            color = tuple(map(int, np.random.randint(0, 255, size=3)))
            
            # 繪製路徑
            points_array = np.array(points, dtype=np.int32)
            cv2.polylines(display_image, [points_array], False, color, 2)
            
            # 在路徑起點和終點標記
            if points:
                cv2.circle(display_image, tuple(map(int, points[0])), 5, color, -1)
                cv2.circle(display_image, tuple(map(int, points[-1])), 5, color, -1)
        
        return display_image

    @staticmethod
    def create_heatmap(image, positions, sigma=50):
        """創建移動熱圖"""
        heatmap = np.zeros(image.shape[:2], dtype=np.float32)
        
        for pos in positions:
            x, y = map(int, pos)
            # 使用高斯核生成熱圖
            y_grid, x_grid = np.ogrid[-y:heatmap.shape[0]-y, -x:heatmap.shape[1]-x]
            mask = np.exp(-(x_grid*x_grid + y_grid*y_grid) / (2*sigma*sigma))
            heatmap += mask
        
        # 正規化熱圖
        heatmap = np.uint8(255 * heatmap / np.max(heatmap))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 與原圖混合
        output = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
        
        return output
