import cv2
import numpy as np


class CoordinateMapper:
    def __init__(self, reference_image=None, reference_points=None):
        self.court_image = None
        self.reference_points = None
        self.tracking_data = {}  # 新增追蹤數據儲存
        self.max_allowed_distance = 2.5  # 設定最大允許距離為1.5公尺
        if reference_image is not None:
            self.set_reference(reference_image, reference_points)

    def set_reference(self, court_image_path, reference_points=None):
        """設置參考圖片和參考點"""
        self.court_image = cv2.imread(court_image_path)
        if reference_points is None:
            self.reference_points = np.array([
                [497, 347],  # top_left
                [497, 567],  # bottom_left
                [853, 347],  # top_right
                [853, 567]  # bottom_right
            ], dtype=np.float32)
        else:
            self.reference_points = np.array(reference_points, dtype=np.float32)

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

    def calculate_distance(self, point1, point2):
        """計算兩點間的實際距離(公尺)"""
        try:
            # 定義比例尺 (pixel to meters)
            distance_scale_horizontal = 5.77 / (853 - 497)  # 5.77M = 356pixels
            distance_scale_vertical = 3.66 / (567 - 347)  # 3.66M = 220pixels

            # 計算水平和垂直距離
            horizontal_pixel_dist = abs(point1[0] - point2[0])
            horizontal_real_dist = horizontal_pixel_dist * distance_scale_horizontal

            vertical_pixel_dist = abs(point1[1] - point2[1])
            vertical_real_dist = vertical_pixel_dist * distance_scale_vertical

            # 計算實際距離(畢氏定理)
            total_distance = round((horizontal_real_dist ** 2 + vertical_real_dist ** 2) ** 0.5, 2)
            return total_distance
        except Exception as e:
            print(f"計算距離錯誤: {e}")
            return float('inf')  # 返回無限大表示計算失敗

    def track_movement(self, track_id, position):
        """追蹤並記錄移動軌跡"""
        try:
            # 檢查位置是否在合理範圍內
            if not (0 <= position[0] <= 1920 and 0 <= position[1] <= 1080):  # 假設影像解析度為 1920x1080
                print(f"位置超出範圍: {position}")
                return self.tracking_data.get(track_id, {}).get('total_distance', 0)

            if track_id not in self.tracking_data:
                self.tracking_data[track_id] = {
                    'positions': [],
                    'total_distance': 0,
                    'color': self.get_color_from_id(track_id)
                }
                self.tracking_data[track_id]['positions'].append(position)
                return 0

            last_pos = self.tracking_data[track_id]['positions'][-1]
            distance = self.calculate_distance(last_pos, position)

            # 如果距離大於閾值，記錄錯誤並返回當前總距離
            if distance > self.max_allowed_distance:
                print(f"警告: 軌跡 {track_id} 的移動距離 ({distance}m) 超過閾值 ({self.max_allowed_distance}m)")
                return self.tracking_data[track_id]['total_distance']

            # 距離合理，更新軌跡
            self.tracking_data[track_id]['positions'].append(position)
            self.tracking_data[track_id]['total_distance'] += distance

            return self.tracking_data[track_id]['total_distance']

        except Exception as e:
            print(f"追蹤移動錯誤: {e}")
            return self.tracking_data.get(track_id, {}).get('total_distance', 0)

    def compute_homography(self, video_points):
        """計算單應性矩陣"""
        if self.reference_points is None:
            raise ValueError("Reference points not set")
        return cv2.findHomography(video_points, self.reference_points)

    def project_point(self, point, H):
        """投影單個點"""
        point_reshaped = np.array([point], dtype=np.float32).reshape(1, 1, 2)
        transformed_point = cv2.perspectiveTransform(point_reshaped, H)
        return transformed_point[0][0]

    def draw_trajectory(self, court_image, track_id, max_points=50):
        """在球場圖上繪製球員軌跡"""
        if track_id not in self.tracking_data:
            return court_image

        color = self.tracking_data[track_id]['color']
        positions = self.tracking_data[track_id]['positions'][-max_points:]

        if len(positions) > 1:
            points_array = np.array(positions, dtype=np.int32)
            cv2.polylines(court_image, [points_array], False, color, 2)

            # 繪製最新位置
            latest_pos = positions[-1]
            cv2.circle(court_image, (int(latest_pos[0]), int(latest_pos[1])), 5, color, -1)
            cv2.putText(court_image,
                        f"ID: {track_id} Dist: {self.tracking_data[track_id]['total_distance']}m",
                        (int(latest_pos[0]) + 5, int(latest_pos[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return court_image

    def clear_trajectories(self):
        """清除所有軌跡記錄"""
        self.tracking_data.clear()

    def remove_trajectory(self, track_id):
        """移除指定ID的軌跡記錄"""
        if track_id in self.tracking_data:
            del self.tracking_data[track_id]