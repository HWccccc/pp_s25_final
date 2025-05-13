import os
import pandas as pd
import base64
from datetime import datetime

class DataManager:
    def __init__(self, data_folder):
        """初始化數據管理器"""
        self.data_folder = data_folder
        self.cache = {}
        self.cache_timeout = 300  # 快取超時時間（秒）
        self.last_cache_time = {}

    def get_player_data(self, team, number):
        """獲取球員資料"""
        try:
            csv_path = os.path.join(self.data_folder, team, f'{team}.csv')
            if not os.path.exists(csv_path):
                return None, f"找不到球隊 {team} 的資料檔案"

            cache_key = f"{team}_{number}"
            current_time = datetime.now().timestamp()

            # 檢查快取是否有效
            if (cache_key in self.cache and 
                current_time - self.last_cache_time.get(cache_key, 0) < self.cache_timeout):
                return self.cache[cache_key], None

            # 讀取 CSV 檔案
            df = pd.read_csv(csv_path)
            if 'Number' not in df.columns:
                return None, f"球隊 {team} 的資料檔案格式錯誤"

            player_data = df[df['Number'] == int(number)].to_dict('records')
            if not player_data:
                return None, f"找不到球隊 {team} 中號碼為 {number} 的球員"

            # 更新快取
            self.cache[cache_key] = player_data[0]
            self.last_cache_time[cache_key] = current_time

            return player_data[0], None

        except Exception as e:
            return None, f"讀取球員資料時發生錯誤: {str(e)}"

    def get_player_image(self, team, player_name):
        """獲取球員圖片"""
        try:
            image_path = os.path.join(self.data_folder, team, f'{player_name}.png')
            if not os.path.exists(image_path):
                return None, f"找不到球員 {player_name} 的圖片"

            cache_key = f"img_{team}_{player_name}"
            current_time = datetime.now().timestamp()

            # 檢查快取是否有效
            if (cache_key in self.cache and 
                current_time - self.last_cache_time.get(cache_key, 0) < self.cache_timeout):
                return self.cache[cache_key], None

            # 讀取並編碼圖片
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()
                image_data = f'data:image/png;base64,{encoded_image}'

                # 更新快取
                self.cache[cache_key] = image_data
                self.last_cache_time[cache_key] = current_time

                return image_data, None

        except Exception as e:
            return None, f"讀取球員圖片時發生錯誤: {str(e)}"

    def update_player_stats(self, player_key, stats):
        """更新球員統計資料"""
        try:
            team, number = player_key.split('_')
            csv_path = os.path.join(self.data_folder, team, f'{team}_stats.csv')

            # 讀取現有統計資料或創建新的
            if os.path.exists(csv_path):
                stats_df = pd.read_csv(csv_path)
            else:
                stats_df = pd.DataFrame(columns=['Number', 'Games', 'Points', 'Stamina'])

            # 更新統計資料
            player_stats = stats_df[stats_df['Number'] == int(number)]
            if player_stats.empty:
                stats_df = pd.concat([stats_df, pd.DataFrame([{
                    'Number': int(number),
                    **stats
                }])], ignore_index=True)
            else:
                idx = player_stats.index[0]
                for key, value in stats.items():
                    stats_df.at[idx, key] = value

            # 保存更新後的統計資料
            stats_df.to_csv(csv_path, index=False)
            return True, None

        except Exception as e:
            return False, f"更新球員統計資料時發生錯誤: {str(e)}"

    def get_team_stats(self, team):
        """獲取球隊統計資料"""
        try:
            csv_path = os.path.join(self.data_folder, team, f'{team}_stats.csv')
            if not os.path.exists(csv_path):
                return None, f"找不到球隊 {team} 的統計資料"

            stats_df = pd.read_csv(csv_path)
            return stats_df.to_dict('records'), None

        except Exception as e:
            return None, f"讀取球隊統計資料時發生錯誤: {str(e)}"

    def clear_cache(self):
        """清除快取"""
        self.cache.clear()
        self.last_cache_time.clear()
