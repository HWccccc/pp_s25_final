from collections import deque
import numpy as np
from typing import Tuple, List, Dict, Optional, Any


class ScoringAnalyzer:
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        # 初始化得分相關變數
        self.team_scores = {
            '3-s': 0,
            'biv': 0,
            'unknown': 0
        }
        self.cooldown = 0
        self.cooldown_frames = 20
        self.last_touch_team = None

        # 初始化解析度相關參數
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.touch_distance_ratio = 0.08  # 觸球判定範圍佔畫面寬度的比例
        self.horizontal_threshold_ratio = 0.3  # 水平判定閾值佔籃框寬度的比例

        # 初始化追蹤用的隊列
        self.basketball_center_positions = deque(maxlen=20)
        self.basketball_left_positions = deque(maxlen=20)
        self.basketball_right_positions = deque(maxlen=20)
        self.hoop_center_positions = deque(maxlen=20)
        self.hoop_left_positions = deque(maxlen=20)
        self.hoop_right_positions = deque(maxlen=20)

    def update_positions(self, basketball_data: Optional[Tuple], hoop_data: Optional[Tuple]) -> None:
        """
        更新球和籃框的位置資料

        Args:
            basketball_data: tuple (center, left, right) of basketball positions
            hoop_data: tuple (center, left, right) of hoop positions
        """
        if not basketball_data or not hoop_data:
            #print("[Debug] Missing basketball_data or hoop_data")
            return

        basketball_center, basketball_left, basketball_right = basketball_data
        hoop_center, hoop_left, hoop_right = hoop_data

        if all([basketball_center, hoop_center, hoop_left, hoop_right]):
            # 座標調整為新解析度
            def scale_position(pos):
                if pos:
                    x, y = pos
                    scaled_x = (x / 1440) * self.frame_width
                    scaled_y = (y / 1080) * self.frame_height
                    return (scaled_x, scaled_y)
                return None

            scaled_basketball_center = scale_position(basketball_center)
            scaled_basketball_left = scale_position(basketball_left)
            scaled_basketball_right = scale_position(basketball_right)
            scaled_hoop_center = scale_position(hoop_center)
            scaled_hoop_left = scale_position(hoop_left)
            scaled_hoop_right = scale_position(hoop_right)

            self.basketball_center_positions.append(scaled_basketball_center)
            if scaled_basketball_left and scaled_basketball_right:
                self.basketball_left_positions.append(scaled_basketball_left)
                self.basketball_right_positions.append(scaled_basketball_right)
            self.hoop_center_positions.append(scaled_hoop_center)
            self.hoop_left_positions.append(scaled_hoop_left)
            self.hoop_right_positions.append(scaled_hoop_right)

            #print(f"[Debug] Updated positions:")
            #print(f"- Original basketball center: {basketball_center}")
            #print(f"- Scaled basketball center: {scaled_basketball_center}")
            #print(f"- Original hoop center: {hoop_center}")
            #print(f"- Scaled hoop center: {scaled_hoop_center}")
        else:
            print("[Debug] Some position data is missing")

    def predict_basketball_position(self) -> Optional[Tuple[int, int]]:
        """使用拋物線預測籃球位置"""
        if len(self.basketball_center_positions) >= 3:
            prev_positions = np.array(list(self.basketball_center_positions))
            xs = prev_positions[:, 0]
            ys = prev_positions[:, 1]

            try:
                coeff = np.polyfit(xs, ys, 2)
                a, b, c = coeff
                #print(f"[Debug] Trajectory coefficients: a={a:.4f}, b={b:.4f}, c={c:.4f}")

                last_x = xs[-1]
                predicted_x = last_x + (last_x - xs[-2])
                predicted_y = a * (predicted_x ** 2) + b * predicted_x + c

                #print(f"[Debug] Predicted position: ({predicted_x:.2f}, {predicted_y:.2f})")
                return (int(predicted_x), int(predicted_y))
            except np.RankWarning:
                #print("[Debug] Failed to fit trajectory - possibly collinear points")
                return None
        #print("[Debug] Not enough position data for prediction")
        return None

    def update_last_touch(self, basketball_center: Tuple[float, float],
                          player_data: Dict[str, Tuple[float, float, float, float, str]]) -> None:
        """
        更新最後觸球球員的隊伍

        Args:
            basketball_center: 籃球中心座標 (x, y)
            player_data: 球員資料字典，包含位置和隊伍資訊
        """
        if not basketball_center or not player_data:
            #print("[Debug] Missing basketball_center or player_data")
            return

        min_distance = float('inf')
        closest_team = None
        touch_threshold = self.frame_width * self.touch_distance_ratio

        # 調整籃球座標至新解析度
        scaled_bx = (basketball_center[0] / 1440) * self.frame_width
        scaled_by = (basketball_center[1] / 1080) * self.frame_height

        for track_id, (x1, y1, x2, y2, team) in player_data.items():
            # 調整球員座標至新解析度
            scaled_x1 = (x1 / 1440) * self.frame_width
            scaled_y1 = (y1 / 1080) * self.frame_height
            scaled_x2 = (x2 / 1440) * self.frame_width
            scaled_y2 = (y2 / 1080) * self.frame_height

            player_center = ((scaled_x1 + scaled_x2) / 2, (scaled_y1 + scaled_y2) / 2)
            distance = np.sqrt((scaled_bx - player_center[0]) ** 2 +
                               (scaled_by - player_center[1]) ** 2)

            #print(f"[Debug] Player {track_id} (Team {team}):")
            #print(f"- Distance to ball: {distance:.2f}")
            #print(f"- Touch threshold: {touch_threshold:.2f}")

            if distance < min_distance and distance < touch_threshold:
                min_distance = distance
                closest_team = team

        if closest_team:
            self.last_touch_team = closest_team
            print(f"[Debug] Updated last touch team: {closest_team} (distance: {min_distance:.2f})")
        else:
            print("[Debug] No team close enough to ball")

    def check_scoring(self, basketball_center: Optional[Tuple[float, float]],
                      hoop_center: Optional[Tuple[float, float]]) -> Tuple[bool, Optional[str]]:
        """
        檢查是否進籃

        Returns:
            tuple: (是否進籃, 進球隊伍)
        """
        if self.cooldown > 0:
            #print(f"[Debug] Scoring cooldown active: {self.cooldown} frames remaining")
            self.cooldown -= 1
            return False, None

        if (len(self.basketball_center_positions) <= 2 or
                not self.hoop_left_positions or
                not self.hoop_right_positions or
                not basketball_center or
                not hoop_center):
            #print("[Debug] Insufficient position data for scoring check")
            return False, None

        height_check = False
        downward_motion = False

        #print("\n[Debug] Starting scoring check...")

        # 調整座標至新解析度
        scaled_basketball = (
            (basketball_center[0] / 1440) * self.frame_width,
            (basketball_center[1] / 1080) * self.frame_height
        )
        scaled_hoop = (
            (hoop_center[0] / 1440) * self.frame_width,
            (hoop_center[1] / 1080) * self.frame_height
        )

        #print(f"[Debug] Scaled positions:")
        #print(f"- Basketball: {scaled_basketball}")
        #print(f"- Hoop: {scaled_hoop}")

        for i in range(1, len(self.basketball_center_positions)):
            prev_bx, prev_by = self.basketball_center_positions[i - 1]
            curr_bx, curr_by = self.basketball_center_positions[i]
            curr_hlx, curr_hly = self.hoop_left_positions[i]
            curr_hrx, curr_hry = self.hoop_right_positions[i]

            #print(f"\n[Debug] Frame {i} analysis:")
            #print(f"- Previous ball position: ({prev_bx:.2f}, {prev_by:.2f})")
            #print(f"- Current ball position: ({curr_bx:.2f}, {curr_by:.2f})")
            #print(f"- Hoop left: ({curr_hlx:.2f}, {curr_hly:.2f})")
            #print(f"- Hoop right: ({curr_hrx:.2f}, {curr_hry:.2f})")

            if curr_hlx < curr_bx < curr_hrx:
                #print("[Debug] Ball is within hoop's horizontal range")
                if prev_by < curr_hly:
                    height_check = True
                    #print("[Debug] Ball was above the hoop")
                if curr_by > prev_by and height_check:
                    downward_motion = True
                    #print("[Debug] Ball is moving downward")
            else:
                height_check = False
                #print("[Debug] Ball is outside hoop's horizontal range")

        if height_check and downward_motion and scaled_basketball[1] > scaled_hoop[1]:
            #print("\n[Debug] Initial scoring conditions met:")
            #print("- Ball was above hoop: Yes")
            #print("- Downward motion detected: Yes")
            #print("- Ball is below hoop center: Yes")

            curr_hlx, curr_hly = self.hoop_left_positions[-1]
            curr_hrx, curr_hry = self.hoop_right_positions[-1]

            hoop_cx = (curr_hlx + curr_hrx) / 2.0
            hoop_width = curr_hrx - curr_hlx
            curr_bx = scaled_basketball[0]

            horizontal_threshold = hoop_width * self.horizontal_threshold_ratio
            horizontal_condition = abs(curr_bx - hoop_cx) < horizontal_threshold

            #print(f"[Debug] Horizontal check:")
            #print(f"- Ball distance from hoop center: {abs(curr_bx - hoop_cx):.2f}")
            #print(f"- Hoop width: {hoop_width:.2f}")
            #print(f"- Threshold: {horizontal_threshold:.2f}")
            #print(f"- Horizontal condition met: {horizontal_condition}")

            hoop_vertical_mid = (curr_hly + curr_hry) / 2.0
            vertical_condition = scaled_basketball[1] > hoop_vertical_mid

            #print(f"[Debug] Vertical check:")
            #print(f"- Ball vertical position: {scaled_basketball[1]:.2f}")
            #print(f"- Hoop vertical midpoint: {hoop_vertical_mid:.2f}")
            #print(f"- Vertical condition met: {vertical_condition}")

            if horizontal_condition and vertical_condition:
                #print("\n[Debug] SCORE DETECTED!")
                #print(f"[Debug] Last touch team: {self.last_touch_team}")

                self.cooldown = self.cooldown_frames
                self.basketball_center_positions.clear()

                if self.last_touch_team:
                    self.team_scores[self.last_touch_team] += 2
                    return True, self.last_touch_team
                else:
                    self.team_scores['unknown'] += 2
                    return True, 'unknown'
            else:
                print("[Debug] Final scoring conditions not met")

        return False, None

    def get_scores(self) -> Dict[str, int]:
        """獲取當前比分"""
        #print(f"[Debug] Current scores: {self.team_scores}")
        return self.team_scores.copy()

    def reset_scores(self) -> None:
        """重置比分"""
        #("[Debug] Resetting scores")
        self.team_scores = {
            '3-s': 0,
            'biv': 0,
            'unknown': 0
        }