import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import cv2
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 嘗試導入 CUDA 加速庫
try:
    import cupy as cp
    # 測試基本操作避免運行時錯誤
    test_array = cp.array([1, 2, 3])
    result = cp.sum(test_array).get()
    CUPY_AVAILABLE = True
except:
    CUPY_AVAILABLE = False

# Method A: Original coordinate_mapper.py (完全相同)
class CoordinateMapperOriginal:
    """完全按照原始 coordinate_mapper.py"""
    
    def __init__(self, reference_image=None, reference_points=None):
        self.court_image = None
        self.reference_points = None
        self.tracking_data = {}
        self.max_allowed_distance = 2.5
        
        if reference_image is not None:
            self.set_reference(reference_image, reference_points)

    def set_reference(self, court_image_path, reference_points=None):
        """設置參考圖片和參考點 - 原始邏輯"""
        self.court_image = cv2.imread(court_image_path) if isinstance(court_image_path, str) else court_image_path
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
        """根據 track_id 生成固定顏色 - 原始邏輯"""
        try:
            if track_id is None:
                return (255, 255, 255)
            seed_value = abs(hash(str(track_id))) % (2 ** 32 - 1)
            np.random.seed(seed_value)
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            return color
        except Exception as e:
            print(f"生成顏色錯誤: {e}")
            return (255, 255, 255)

    def calculate_distance(self, point1, point2):
        """計算兩點間的實際距離(公尺) - 原始邏輯"""
        try:
            distance_scale_horizontal = 5.77 / (853 - 497)
            distance_scale_vertical = 3.66 / (567 - 347)

            horizontal_pixel_dist = abs(point1[0] - point2[0])
            horizontal_real_dist = horizontal_pixel_dist * distance_scale_horizontal

            vertical_pixel_dist = abs(point1[1] - point2[1])
            vertical_real_dist = vertical_pixel_dist * distance_scale_vertical

            total_distance = round((horizontal_real_dist ** 2 + vertical_real_dist ** 2) ** 0.5, 2)
            return total_distance
        except Exception as e:
            print(f"計算距離錯誤: {e}")
            return float('inf')

    def track_movement(self, track_id, position):
        """追蹤並記錄移動軌跡 - 原始邏輯"""
        try:
            if not (0 <= position[0] <= 1920 and 0 <= position[1] <= 1080):
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

            if distance > self.max_allowed_distance:
                return self.tracking_data[track_id]['total_distance']

            self.tracking_data[track_id]['positions'].append(position)
            self.tracking_data[track_id]['total_distance'] += distance
            return self.tracking_data[track_id]['total_distance']

        except Exception as e:
            print(f"追蹤移動錯誤: {e}")
            return self.tracking_data.get(track_id, {}).get('total_distance', 0)

    def draw_trajectory_on_court(self, max_points=50):
        """在球場上繪製軌跡 - 原始邏輯"""
        if self.court_image is None:
            return None
            
        display_image = self.court_image.copy()
        
        for track_id, data in self.tracking_data.items():
            color = data['color']
            positions = data['positions'][-max_points:]
            
            if len(positions) > 1:
                points_array = np.array(positions, dtype=np.int32)
                cv2.polylines(display_image, [points_array], False, color, 2)
                
                latest_pos = positions[-1]
                cv2.circle(display_image, (int(latest_pos[0]), int(latest_pos[1])), 5, color, -1)
                cv2.putText(display_image,
                            f"ID: {track_id} Dist: {data['total_distance']:.1f}m",
                            (int(latest_pos[0]) + 5, int(latest_pos[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return display_image

# Method B: NumPy 向量化處理 (原本的C)
class CoordinateMapperMethodB:
    """方案B: NumPy 向量化處理"""
    
    def __init__(self, reference_image=None, reference_points=None):
        self.court_image = None
        self.reference_points = None
        self.tracking_data = {}
        self.max_allowed_distance = 2.5
        self.distance_scale_horizontal = 5.77 / (853 - 497)
        self.distance_scale_vertical = 3.66 / (567 - 347)
        
        if reference_image is not None:
            self.set_reference(reference_image, reference_points)

    def set_reference(self, court_image_path, reference_points=None):
        """設置參考圖片和參考點"""
        self.court_image = cv2.imread(court_image_path) if isinstance(court_image_path, str) else court_image_path
        if reference_points is None:
            self.reference_points = np.array([
                [497, 347], [497, 567], [853, 347], [853, 567]
            ], dtype=np.float32)
        else:
            self.reference_points = np.array(reference_points, dtype=np.float32)

    def get_color_from_id(self, track_id):
        """根據 track_id 生成固定顏色"""
        if track_id is None:
            return (255, 255, 255)
        seed_value = abs(hash(str(track_id))) % (2 ** 32 - 1)
        np.random.seed(seed_value)
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        return color

    def calculate_distances_vectorized_batch(self, points1_list, points2_list):
        """NumPy 向量化批次距離計算"""
        if not points1_list or not points2_list:
            return []
        
        points1_array = np.array(points1_list)
        points2_array = np.array(points2_list)
        
        # 向量化計算所有距離
        horizontal_diffs = np.abs(points1_array[:, 0] - points2_array[:, 0]) * self.distance_scale_horizontal
        vertical_diffs = np.abs(points1_array[:, 1] - points2_array[:, 1]) * self.distance_scale_vertical
        
        distances = np.sqrt(horizontal_diffs**2 + vertical_diffs**2)
        return np.round(distances, 2)

    def track_movement(self, track_id, position):
        """追蹤移動 - 但實際計算會在 update_all_players 中進行"""
        try:
            if not (0 <= position[0] <= 1920 and 0 <= position[1] <= 1080):
                return self.tracking_data.get(track_id, {}).get('total_distance', 0)

            if track_id not in self.tracking_data:
                self.tracking_data[track_id] = {
                    'positions': [],
                    'total_distance': 0,
                    'color': self.get_color_from_id(track_id)
                }
                self.tracking_data[track_id]['positions'].append(position)
                return 0

            # 暫時加入新位置，等待向量化計算
            self.tracking_data[track_id]['positions'].append(position)
            return self.tracking_data[track_id]['total_distance']

        except Exception as e:
            print(f"追蹤移動錯誤: {e}")
            return self.tracking_data.get(track_id, {}).get('total_distance', 0)

    def update_all_players_vectorized(self, player_positions):
        """NumPy 向量化更新所有球員位置 - 即時處理一個 frame"""
        if not player_positions:
            return
        
        # 收集需要計算距離的球員
        points1_list = []
        points2_list = []
        track_ids = []
        
        for track_id, new_position in player_positions.items():
            if track_id in self.tracking_data and len(self.tracking_data[track_id]['positions']) > 1:
                last_pos = self.tracking_data[track_id]['positions'][-2]
                current_pos = self.tracking_data[track_id]['positions'][-1]
                
                points1_list.append(last_pos)
                points2_list.append(current_pos)
                track_ids.append(track_id)
        
        # NumPy 向量化計算所有距離
        if points1_list:
            distances = self.calculate_distances_vectorized_batch(points1_list, points2_list)
            
            # 更新距離
            for i, track_id in enumerate(track_ids):
                distance = distances[i]
                if distance <= self.max_allowed_distance:
                    self.tracking_data[track_id]['total_distance'] += distance
                else:
                    # 移除不合理的位置
                    self.tracking_data[track_id]['positions'].pop()

    def draw_trajectory_on_court(self, max_points=50):
        """繪製軌跡"""
        if self.court_image is None:
            return None
            
        display_image = self.court_image.copy()
        
        for track_id, data in self.tracking_data.items():
            color = data['color']
            positions = data['positions'][-max_points:]
            
            if len(positions) > 1:
                points_array = np.array(positions, dtype=np.int32)
                cv2.polylines(display_image, [points_array], False, color, 2)
                
                latest_pos = positions[-1]
                cv2.circle(display_image, (int(latest_pos[0]), int(latest_pos[1])), 5, color, -1)
                cv2.putText(display_image,
                            f"ID: {track_id} Dist: {data['total_distance']:.1f}m (NumPy)",
                            (int(latest_pos[0]) + 5, int(latest_pos[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return display_image

# Method C: CUDA/GPU 並行處理 (原本的B)
if CUPY_AVAILABLE:
    class CoordinateMapperMethodC:
        """方案C: CUDA/GPU 並行處理"""
        
        def __init__(self, reference_image=None, reference_points=None):
            self.court_image = None
            self.reference_points = None
            self.tracking_data = {}
            self.max_allowed_distance = 2.5
            self.distance_scale_horizontal = 5.77 / (853 - 497)
            self.distance_scale_vertical = 3.66 / (567 - 347)
            
            if reference_image is not None:
                self.set_reference(reference_image, reference_points)

        def set_reference(self, court_image_path, reference_points=None):
            """設置參考圖片和參考點"""
            self.court_image = cv2.imread(court_image_path) if isinstance(court_image_path, str) else court_image_path
            if reference_points is None:
                self.reference_points = np.array([
                    [497, 347], [497, 567], [853, 347], [853, 567]
                ], dtype=np.float32)
            else:
                self.reference_points = np.array(reference_points, dtype=np.float32)

        def get_color_from_id(self, track_id):
            """根據 track_id 生成固定顏色"""
            if track_id is None:
                return (255, 255, 255)
            seed_value = abs(hash(str(track_id))) % (2 ** 32 - 1)
            np.random.seed(seed_value)
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            return color

        def calculate_distances_gpu_batch(self, points1_list, points2_list):
            """GPU 批次距離計算"""
            if not points1_list or not points2_list:
                return []
            
            try:
                # 轉移到 GPU
                points1_gpu = cp.array(points1_list)
                points2_gpu = cp.array(points2_list)
                
                # GPU 向量化計算
                horizontal_diffs = cp.abs(points1_gpu[:, 0] - points2_gpu[:, 0]) * self.distance_scale_horizontal
                vertical_diffs = cp.abs(points1_gpu[:, 1] - points2_gpu[:, 1]) * self.distance_scale_vertical
                
                distances_gpu = cp.sqrt(horizontal_diffs**2 + vertical_diffs**2)
                
                # 轉回 CPU
                return cp.asnumpy(cp.round(distances_gpu, 2))
            except:
                # GPU失敗時回退到numpy
                points1_array = np.array(points1_list)
                points2_array = np.array(points2_list)
                horizontal_diffs = np.abs(points1_array[:, 0] - points2_array[:, 0]) * self.distance_scale_horizontal
                vertical_diffs = np.abs(points1_array[:, 1] - points2_array[:, 1]) * self.distance_scale_vertical
                distances = np.sqrt(horizontal_diffs**2 + vertical_diffs**2)
                return np.round(distances, 2)

        def track_movement(self, track_id, position):
            """追蹤移動 - 但實際計算會在 update_all_players 中批次進行"""
            try:
                if not (0 <= position[0] <= 1920 and 0 <= position[1] <= 1080):
                    return self.tracking_data.get(track_id, {}).get('total_distance', 0)

                if track_id not in self.tracking_data:
                    self.tracking_data[track_id] = {
                        'positions': [],
                        'total_distance': 0,
                        'color': self.get_color_from_id(track_id)
                    }
                    self.tracking_data[track_id]['positions'].append(position)
                    return 0

                # 暫時加入新位置，等待批次計算
                self.tracking_data[track_id]['positions'].append(position)
                return self.tracking_data[track_id]['total_distance']

            except Exception as e:
                print(f"追蹤移動錯誤: {e}")
                return self.tracking_data.get(track_id, {}).get('total_distance', 0)

        def update_all_players_gpu(self, player_positions):
            """GPU 批次更新所有球員位置 - 即時處理一個 frame"""
            if not player_positions:
                return
            
            # 收集需要計算距離的球員
            points1_list = []
            points2_list = []
            track_ids = []
            
            for track_id, new_position in player_positions.items():
                if track_id in self.tracking_data and len(self.tracking_data[track_id]['positions']) > 1:
                    last_pos = self.tracking_data[track_id]['positions'][-2]  # 倒數第二個位置
                    current_pos = self.tracking_data[track_id]['positions'][-1]  # 最新位置
                    
                    points1_list.append(last_pos)
                    points2_list.append(current_pos)
                    track_ids.append(track_id)
            
            # GPU 批次計算所有距離
            if points1_list:
                distances = self.calculate_distances_gpu_batch(points1_list, points2_list)
                
                # 更新距離
                for i, track_id in enumerate(track_ids):
                    distance = distances[i]
                    if distance <= self.max_allowed_distance:
                        self.tracking_data[track_id]['total_distance'] += distance
                    else:
                        # 移除不合理的位置
                        self.tracking_data[track_id]['positions'].pop()

        def draw_trajectory_on_court(self, max_points=50):
            """繪製軌跡"""
            if self.court_image is None:
                return None
                
            display_image = self.court_image.copy()
            
            for track_id, data in self.tracking_data.items():
                color = data['color']
                positions = data['positions'][-max_points:]
                
                if len(positions) > 1:
                    points_array = np.array(positions, dtype=np.int32)
                    cv2.polylines(display_image, [points_array], False, color, 2)
                    
                    latest_pos = positions[-1]
                    cv2.circle(display_image, (int(latest_pos[0]), int(latest_pos[1])), 5, color, -1)
                    cv2.putText(display_image,
                                f"ID: {track_id} Dist: {data['total_distance']:.1f}m",
                                (int(latest_pos[0]) + 5, int(latest_pos[1]) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            return display_image
else:
    # 如果沒有 CuPy，創建一個假的類別
    class CoordinateMapperMethodC:
        def __init__(self, *args, **kwargs):
            raise ImportError("CuPy not available for GPU acceleration")

class SimplePerformanceTestUI:
    """改進版效能測試UI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Improved Coordinate Mapper Performance Test")
        self.root.geometry("1400x800")
        
        # 測試參數
        self.num_players = tk.StringVar(value="5")
        self.test_frames = tk.StringVar(value="100")
        self.selected_method = tk.StringVar(value="A")
        self.court_image_path = tk.StringVar(value="")
        
        # 測試狀態
        self.current_mapper = None
        self.running = False
        self.auto_testing = False
        self.player_positions = {}
        self.update_times = []
        
        # Auto Test 相關
        self.auto_test_results = {}
        self.auto_test_thread = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """設置UI"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 左側控制面板
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        
        # 球場圖片選擇
        court_frame = ttk.LabelFrame(control_frame, text="Court Image", padding="5")
        court_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(court_frame, text="Load Court Image", 
                  command=self.load_court_image).grid(row=0, column=0, pady=5)
        ttk.Label(court_frame, textvariable=self.court_image_path, 
                 width=40, foreground="blue").grid(row=1, column=0, pady=5)
        
        # 方案選擇
        method_frame = ttk.LabelFrame(control_frame, text="Method Selection", padding="5")
        method_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Radiobutton(method_frame, text="Method A: Original CPU (Baseline)", 
                       variable=self.selected_method, value="A").grid(row=0, column=0, sticky=tk.W)
        
        ttk.Radiobutton(method_frame, text="Method B: NumPy Vectorized", 
                       variable=self.selected_method, value="B").grid(row=1, column=0, sticky=tk.W)
        
        if CUPY_AVAILABLE:
            ttk.Radiobutton(method_frame, text="Method C: CUDA/GPU Parallel", 
                           variable=self.selected_method, value="C").grid(row=2, column=0, sticky=tk.W)
        else:
            ttk.Label(method_frame, text="Method C: CUDA/GPU (Not Available - install CuPy)", 
                     foreground="red").grid(row=2, column=0, sticky=tk.W)
        
        # 測試參數設定
        params_frame = ttk.LabelFrame(control_frame, text="Test Parameters", padding="5")
        params_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 球員數量輸入
        ttk.Label(params_frame, text="Number of Players:").grid(row=0, column=0, sticky=tk.W, pady=2)
        player_entry = ttk.Entry(params_frame, textvariable=self.num_players, width=10)
        player_entry.grid(row=0, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        
        # 測試幀數輸入
        ttk.Label(params_frame, text="Test Frames:").grid(row=1, column=0, sticky=tk.W, pady=2)
        frames_entry = ttk.Entry(params_frame, textvariable=self.test_frames, width=10)
        frames_entry.grid(row=1, column=1, sticky=tk.W, pady=2, padx=(5, 0))
        
        # 手動測試按鈕
        manual_button_frame = ttk.LabelFrame(control_frame, text="Manual Test", padding="5")
        manual_button_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        button_row1 = ttk.Frame(manual_button_frame)
        button_row1.grid(row=0, column=0, pady=2)
        
        self.start_button = ttk.Button(button_row1, text="Start Test", command=self.start_test)
        self.start_button.grid(row=0, column=0, padx=(0, 5))
        
        self.stop_button = ttk.Button(button_row1, text="Stop Test", 
                                     command=self.stop_test, state='disabled')
        self.stop_button.grid(row=0, column=1, padx=(0, 5))
        
        ttk.Button(button_row1, text="Reset", command=self.reset_test).grid(row=0, column=2)
        
        # Auto Test 按鈕
        auto_button_frame = ttk.LabelFrame(control_frame, text="Auto Test", padding="5")
        auto_button_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.auto_test_button = ttk.Button(auto_button_frame, text="Start Auto Test", 
                                          command=self.start_auto_test)
        self.auto_test_button.grid(row=0, column=0, pady=5)
        
        self.auto_stop_button = ttk.Button(auto_button_frame, text="Stop Auto Test", 
                                          command=self.stop_auto_test, state='disabled')
        self.auto_stop_button.grid(row=0, column=1, padx=(5, 0), pady=5)
        
        # 效能顯示
        perf_frame = ttk.LabelFrame(control_frame, text="Performance", padding="5")
        perf_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.perf_text = tk.Text(perf_frame, height=12, width=50)
        self.perf_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # 球場顯示區域
        court_display_frame = ttk.LabelFrame(main_frame, text="Court Visualization", padding="10")
        court_display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 創建 matplotlib 圖表
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, court_display_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 配置網格權重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
    
    def load_court_image(self):
        """載入球場圖片"""
        file_path = filedialog.askopenfilename(
            title="Select Court Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.court_image_path.set(file_path)
            print(f"Court image loaded: {file_path}")
    
    def init_mapper(self, method):
        """初始化選定的方案"""
        court_path = self.court_image_path.get()
        
        if not court_path:
            messagebox.showwarning("Warning", "Please load a court image first!")
            return None
        
        try:
            if method == "A":
                return CoordinateMapperOriginal(court_path)
            elif method == "B":
                return CoordinateMapperMethodB(court_path)
            elif method == "C":
                if not CUPY_AVAILABLE:
                    messagebox.showerror("Error", "CuPy not available for GPU acceleration!")
                    return None
                return CoordinateMapperMethodC(court_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize mapper: {e}")
            return None
    
    def validate_inputs(self):
        """驗證輸入參數"""
        try:
            num_players = int(self.num_players.get())
            test_frames = int(self.test_frames.get())
            
            if num_players <= 0 or test_frames <= 0:
                raise ValueError("Numbers must be positive")
                
            return num_players, test_frames
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
            return None, None
    
    def generate_player_positions(self, num_players):
        """生成球員位置"""
        for player_id in range(num_players):
            if player_id not in self.player_positions:
                # 初始位置
                self.player_positions[player_id] = [
                    np.random.uniform(100, 800),
                    np.random.uniform(100, 500)
                ]
            else:
                # 更新位置 (隨機移動)
                current_pos = self.player_positions[player_id]
                new_pos = [
                    current_pos[0] + np.random.uniform(-20, 20),
                    current_pos[1] + np.random.uniform(-20, 20)
                ]
                # 邊界檢查
                new_pos[0] = max(50, min(850, new_pos[0]))
                new_pos[1] = max(50, min(550, new_pos[1]))
                self.player_positions[player_id] = new_pos
    
    def run_single_test(self, method, num_players, test_frames):
        """執行單一測試"""
        print(f"開始測試 Method {method}, {num_players} 球員, {test_frames} 幀")
        
        mapper = self.init_mapper(method)
        if not mapper:
            return None
        
        times = []
        self.player_positions.clear()
        
        # 記錄總開始時間
        total_start_time = time.time()
        
        for frame in range(test_frames + 1):  # +1 因為要跳過第一幀
            start_time = time.time()
            
            # 生成球員位置
            self.generate_player_positions(num_players)
            
            # 根據方案執行計算
            if method == "A":
                for player_id, position in self.player_positions.items():
                    mapper.track_movement(player_id, position)
            elif method == "B":
                for player_id, position in self.player_positions.items():
                    mapper.track_movement(player_id, position)
                mapper.update_all_players_vectorized(self.player_positions)
            elif method == "C":
                for player_id, position in self.player_positions.items():
                    mapper.track_movement(player_id, position)
                mapper.update_all_players_gpu(self.player_positions)
            
            frame_time = (time.time() - start_time) * 1000  # ms
            
            # 跳過第一幀（預熱）
            if frame > 0:
                times.append(frame_time)
            
            # 每100幀顯示進度
            if frame % 100 == 0:
                print(f"  進度: {frame}/{test_frames + 1} 幀")
        
        total_time = time.time() - total_start_time
        print(f"完成測試 Method {method}: 總時間 {total_time:.2f}s, 有效幀數 {len(times)}")
        
        if times:
            return {
                'method': method,
                'num_players': num_players,
                'frames': len(times),
                'total_time': total_time,
                'avg_time': np.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_time': np.std(times),
                'all_times': times
            }
        return None
    
    def start_test(self):
        """開始手動測試"""
        num_players, test_frames = self.validate_inputs()
        if num_players is None:
            return
        
        method = self.selected_method.get()
        self.current_mapper = self.init_mapper(method)
        if not self.current_mapper:
            return
        
        self.running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        
        # 重置數據
        self.update_times.clear()
        self.player_positions.clear()
        self.frame_count = 0
        self.target_frames = test_frames
        self.test_start_time = time.time()  # 記錄測試開始時間
        
        # 開始模擬
        self.update_simulation()
        print(f"Started Method {method} with {num_players} players for {test_frames} frames")
    
    def update_simulation(self):
        """更新模擬"""
        if not self.running or self.frame_count >= self.target_frames:
            self.stop_test()
            return
        
        num_players, _ = self.validate_inputs()
        if num_players is None:
            self.stop_test()
            return
        
        # 記錄開始時間
        start_time = time.time()
        
        # 生成新的球員位置
        self.generate_player_positions(num_players)
        
        # 根據方案類型選擇更新方式
        method = self.selected_method.get()
        
        if method == "A":
            for player_id, position in self.player_positions.items():
                self.current_mapper.track_movement(player_id, position)
        elif method == "B":
            for player_id, position in self.player_positions.items():
                self.current_mapper.track_movement(player_id, position)
            self.current_mapper.update_all_players_vectorized(self.player_positions)
        elif method == "C" and CUPY_AVAILABLE:
            for player_id, position in self.player_positions.items():
                self.current_mapper.track_movement(player_id, position)
            self.current_mapper.update_all_players_gpu(self.player_positions)
        
        # 繪製軌跡（每10幀更新一次顯示以提高效能）
        if self.frame_count % 10 == 0:
            trajectory_image = self.current_mapper.draw_trajectory_on_court()
            self.update_court_display(trajectory_image)
        
        # 記錄更新時間（跳過第一幀）
        update_time = (time.time() - start_time) * 1000
        if self.frame_count > 0:  # 跳過第一幀
            self.update_times.append(update_time)
        
        self.frame_count += 1
        
        # 更新效能顯示
        self.update_performance_display()
        
        # 繼續下一次更新
        if self.running:
            self.root.after(1, self.update_simulation)  # 儘快執行下一幀
    
    def stop_test(self):
        """停止測試"""
        self.running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        print("Test stopped")
    
    def reset_test(self):
        """重置測試"""
        self.stop_test()
        self.update_times.clear()
        self.player_positions.clear()
        
        if self.current_mapper:
            self.current_mapper.tracking_data.clear()
        
        # 清除顯示
        self.ax.clear()
        self.ax.set_title('No Test Running')
        self.canvas.draw()
        
        self.perf_text.delete(1.0, tk.END)
        self.perf_text.insert(1.0, "Test Reset - Ready to Start")
        
        print("Test reset")
    
    def start_auto_test(self):
        """開始自動測試"""
        if not self.court_image_path.get():
            messagebox.showwarning("Warning", "Please load a court image first!")
            return
        
        self.auto_testing = True
        self.auto_test_button.config(state='disabled')
        self.auto_stop_button.config(state='normal')
        self.start_button.config(state='disabled')
        
        # 清空之前的結果
        self.auto_test_results.clear()
        
        # 在新線程中運行自動測試
        self.auto_test_thread = threading.Thread(target=self.run_auto_test)
        self.auto_test_thread.daemon = True
        self.auto_test_thread.start()
    
    def run_auto_test(self):
        """執行自動測試"""
        player_counts = [6, 20, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
        methods = ['A', 'B']
        if CUPY_AVAILABLE:
            methods.append('C')
        
        total_tests = len(player_counts) * len(methods)
        current_test = 0
        
        try:
            for num_players in player_counts:
                if not self.auto_testing:
                    break
                    
                for method in methods:
                    if not self.auto_testing:
                        break
                    
                    current_test += 1
                    
                    # 更新進度顯示
                    self.root.after(0, self.update_auto_test_progress, 
                                   current_test, total_tests, method, num_players)
                    
                    # 執行測試
                    result = self.run_single_test(method, num_players, 5000)
                    
                    if result:
                        key = f"{method}_{num_players}"
                        self.auto_test_results[key] = result
                        
                        # 更新結果顯示
                        self.root.after(0, self.update_auto_test_results)
                    
                    # 短暫休息避免系統過載
                    time.sleep(0.1)
        
        except Exception as e:
            print(f"Auto test error: {e}")
        
        finally:
            # 測試完成，恢復UI狀態
            self.root.after(0, self.finish_auto_test)
    
    def update_auto_test_progress(self, current, total, method, num_players):
        """更新自動測試進度"""
        progress_text = f"Auto Test Progress: {current}/{total}\n"
        progress_text += f"Current: Method {method}, {num_players} players\n"
        progress_text += f"Progress: {current/total*100:.1f}%\n\n"
        
        if self.auto_test_results:
            progress_text += "Completed Results:\n"
            for key, result in self.auto_test_results.items():
                method, players = key.split('_')
                progress_text += f"Method {method} ({players} players): {result['avg_time']:.2f}ms\n"
        
        self.perf_text.delete(1.0, tk.END)
        self.perf_text.insert(1.0, progress_text)
    
    def update_auto_test_results(self):
        """更新自動測試結果顯示"""
        if not self.auto_test_results:
            return
        
        results_text = "Auto Test Results:\n"
        results_text += "=" * 50 + "\n\n"
        
        # 按玩家數量分組顯示
        player_counts =  [6, 20, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
        methods = ['A', 'B', 'C'] if CUPY_AVAILABLE else ['A', 'B']
        
        for num_players in player_counts:
            results_text += f"Players: {num_players}\n"
            
            method_results = []
            for method in methods:
                key = f"{method}_{num_players}"
                if key in self.auto_test_results:
                    result = self.auto_test_results[key]
                    method_results.append((method, result['avg_time']))
                    results_text += f"  Method {method}: {result['avg_time']:.2f}ms "
                    results_text += f"(±{result['std_time']:.2f}) "
                    results_text += f"總時間: {result['total_time']:.2f}s "
                    results_text += f"幀數: {result['frames']}\n"
            
            # 計算加速比
            if len(method_results) > 1:
                baseline = method_results[0][1]  # Method A as baseline
                results_text += "  Speedup vs Method A:\n"
                for method, avg_time in method_results[1:]:
                    speedup = baseline / avg_time
                    results_text += f"    Method {method}: {speedup:.2f}x\n"
            
            results_text += "\n"
        
        self.perf_text.delete(1.0, tk.END)
        self.perf_text.insert(1.0, results_text)
    
    def stop_auto_test(self):
        """停止自動測試"""
        self.auto_testing = False
        print("Auto test stopping...")
    
    def finish_auto_test(self):
        """完成自動測試"""
        self.auto_testing = False
        self.auto_test_button.config(state='normal')
        self.auto_stop_button.config(state='disabled')
        self.start_button.config(state='normal')
        
        # 顯示最終結果
        self.update_auto_test_results()
        print("Auto test completed!")
    
    def update_court_display(self, trajectory_image):
        """更新球場顯示"""
        if trajectory_image is not None:
            # 轉換顏色空間
            rgb_image = cv2.cvtColor(trajectory_image, cv2.COLOR_BGR2RGB)
            
            # 清除並顯示新圖像
            self.ax.clear()
            self.ax.imshow(rgb_image)
            self.ax.set_title(f'Method {self.selected_method.get()} - {self.num_players.get()} Players')
            self.ax.axis('off')
            
            self.canvas.draw()
    
    def update_performance_display(self):
        """更新效能顯示"""
        if not self.running or not self.update_times:
            return
        
        # 計算統計數據
        avg_time = np.mean(self.update_times)
        min_time = min(self.update_times)
        max_time = max(self.update_times)
        std_time = np.std(self.update_times)
        current_frames = len(self.update_times)
        
        # 計算已花費時間
        if hasattr(self, 'test_start_time'):
            elapsed_time = time.time() - self.test_start_time
        else:
            elapsed_time = 0
        
        perf_info = f"""Method {self.selected_method.get()} Performance:

Players: {self.num_players.get()}
Progress: {self.frame_count}/{self.target_frames} frames
Completed: {current_frames} frames (excluding warmup)

已花費時間: {elapsed_time:.2f}s
Average Time: {avg_time:.2f} ms
Min Time: {min_time:.2f} ms
Max Time: {max_time:.2f} ms
Std Dev: {std_time:.2f} ms

Estimated FPS: {1000/avg_time:.1f}
Estimated Total Time: {(self.target_frames * avg_time / 1000):.2f}s
"""
        
        self.perf_text.delete(1.0, tk.END)
        self.perf_text.insert(1.0, perf_info)

def main():
    """主函數"""
    print("🎯 Improved Coordinate Mapper Performance Test")
    print("=" * 50)
    print(f"\n🔧 Available Methods:")
    print(f"   A: Original CPU - ✅ Available")
    print(f"   B: NumPy Vectorized - ✅ Available") 
    print(f"   C: CUDA/GPU Parallel - {'✅ Available' if CUPY_AVAILABLE else '❌ Not Available (install CuPy)'}")
    
    if not CUPY_AVAILABLE:
        print(f"\n💡 To enable GPU acceleration:")
        print(f"   pip install cupy-cuda11x  # for CUDA 11.x")
        print(f"   pip install cupy-cuda12x  # for CUDA 12.x")
    
    root = tk.Tk()
    app = SimplePerformanceTestUI(root)
    
    def on_closing():
        if app.running or app.auto_testing:
            if messagebox.askokcancel("Confirm", "Test is running. Close anyway?"):
                app.stop_test()
                app.stop_auto_test()
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()