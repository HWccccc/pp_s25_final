import gradio as gr
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


class GradioInterface:
    def __init__(self, tracker):
        self.tracker = tracker
        self.interface = None
        self.is_playing = False
        self.scores_html = None
        self.session_player_history = {}  # 新增：儲存本次播放的歷史記錄

    def update_scores(self, scores):
        """更新得分顯示"""
        try:
            html = """
            <style>
                .scoreboard {
                    display: flex;
                    justify-content: space-around;
                    align-items: center;
                    background-color: #f5f5f5;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 10px 0;
                }
                .team-container {
                    text-align: center;
                }
                .team-logo {
                    width: 80px;
                    height: 80px;
                    object-fit: contain;
                }
                .team-score {
                    font-size: 32px;
                    font-weight: bold;
                    margin-top: 10px;
                }
                .team-3s { color: #2196F3; }
                .team-biv { color: #f44336; }
            </style>
            <div class="scoreboard">
            """

            # 3-s team
            team_img = self.tracker.get_player_image("3-s", "3-s")
            html += f"""
                <div class="team-container">
                    <img src="{team_img}" class="team-logo" onerror="this.style.display='none'">
                    <div class="team-score team-3s">{scores.get('3-s', 0)}</div>
                </div>
            """

            # BIV team
            team_img = self.tracker.get_player_image("biv", "biv")
            html += f"""
                <div class="team-container">
                    <img src="{team_img}" class="team-logo" onerror="this.style.display='none'">
                    <div class="team-score team-biv">{scores.get('biv', 0)}</div>
                </div>
            """

            html += "</div>"
            return html

        except Exception as e:
            print(f"更新得分顯示錯誤: {e}")
            return "得分顯示錯誤"

    def process_video_realtime(self, video_path):
        """即時處理影片"""
        if not self.tracker.load_video(video_path):
            return None, None, None, None, None, "無法開啟影片", "播放", None

        self.is_playing = True
        while True:  # 改成無限循環
            if not self.is_playing:  # 檢查是否暫停
                yield display_pil, processed_pil, court_pil, current_status, history_html, message, "播放", scores_html
                continue  # 如果暫停，繼續等待

            processed_frame, court_frame, detected_players = self.tracker.next_frame()
            if processed_frame is None:
                break

            # 更新球員狀態
            current_status = self.update_current_status(detected_players)
            history_html, message = self.update_player_status(detected_players)

            # 獲取當前得分
            scores = self.tracker.get_current_scores()
            scores_html = self.update_scores(scores)

            # 轉換顏色空間
            display_frame = self.tracker.get_current_frame()
            if display_frame is not None:
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            if processed_frame is not None:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            if court_frame is not None:
                court_frame = cv2.cvtColor(court_frame, cv2.COLOR_BGR2RGB)

            # 轉換為PIL Image
            display_pil = Image.fromarray(display_frame) if display_frame is not None else None
            processed_pil = Image.fromarray(processed_frame) if processed_frame is not None else None
            court_pil = Image.fromarray(court_frame) if court_frame is not None else None

            yield display_pil, processed_pil, court_pil, current_status, history_html, message, "暫停", scores_html

    def handle_play_pause(self):
        """處理播放/暫停"""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.tracker.play()
            return "暫停"
        else:
            self.tracker.pause()
            return "播放"

    def handle_next_frame(self):
        """處理下一幀"""
        processed_frame, court_frame, detected_players = self.tracker.next_frame()
        scores = self.tracker.get_current_scores()
        scores_html = self.update_scores(scores)
        return *self.update_frame_display(processed_frame, court_frame, detected_players), scores_html

    def handle_prev_frame(self):
        """處理上一幀"""
        processed_frame, court_frame, detected_players = self.tracker.prev_frame()
        scores = self.tracker.get_current_scores()
        scores_html = self.update_scores(scores)
        return *self.update_frame_display(processed_frame, court_frame, detected_players), scores_html

    def update_frame_display(self, processed_frame, court_frame, detected_players):
        """更新幀顯示"""
        if processed_frame is None:
            return None, None, None, "", "", "無法獲取幀", "播放"

        current_status = self.update_current_status(detected_players)
        history_html, message = self.update_player_status(detected_players)

        display_frame = self.tracker.get_current_frame()
        if display_frame is not None:
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        if processed_frame is not None:
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        if court_frame is not None:
            court_frame = cv2.cvtColor(court_frame, cv2.COLOR_BGR2RGB)

        display_pil = Image.fromarray(display_frame) if display_frame is not None else None
        processed_pil = Image.fromarray(processed_frame) if processed_frame is not None else None
        court_pil = Image.fromarray(court_frame) if court_frame is not None else None

        return display_pil, processed_pil, court_pil, current_status, history_html, message, "播放"

    def get_stamina_color(x):
        """根據體力值返回對應的顏色"""
        if x <= 25:
            return '#FF0000'  # 紅色 (25%以下)
        elif x <= 50:
            return '#FFD700'  # 黃色 (50%以下)
        elif x <= 75:
            return '#00FF00'  # 綠色 (75%以下)
        else:
            return '#0000FF'  # 藍色 (75%以上)

    def update_current_status(self, detected_players):
        """更新當前球員狀態"""
        current_info = pd.DataFrame()
        processed_players = set()

        for player in detected_players:
            team = player['team']
            number = player['number']
            player_info = player['info']

            player_key = f"{team}_{number}"
            if player_key in processed_players:
                continue

            processed_players.add(player_key)

            if not player_info.empty:
                row_data = {
                    'Team': [team],
                    'Pic': [self.tracker.get_player_image(team, player_info['Player'].iloc[0])],
                    'Player': [player_info['Player'].iloc[0]],
                    'Number': [player_info['Number'].iloc[0]]
                }

                state = self.tracker.player_states.get(player_key, {'score': 0, 'stamina': 100})
                row_data['體力'] = [state['stamina']]

                row_df = pd.DataFrame(row_data)
                current_info = pd.concat([current_info, row_df])

        if not current_info.empty:
            current_info['Pic'] = current_info['Pic'].apply(
                lambda x: f'<img src="{x}" style="width:50px; height:50px;">' if x else '')

            # 直接在 lambda 函數中處理顏色邏輯
            current_info['體力'] = current_info['體力'].apply(
                lambda
                    x: f'<div style="width:100px;background-color:#ddd;"><div style="width:{x}%;background-color:{("#FF0000" if x <= 25 else "#FFD700" if x <= 50 else "#00FF00" if x <= 75 else "#0000FF")};height:20px;text-align:center;color:white;">{x:.3f}%</div></div>')

            current_info = current_info.sort_values(by=['Team', 'Number']).reset_index(drop=True)

        html_table = f'''
        <meta charset="UTF-8">
        <style>
            table {{
                width: auto;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 5px;
                text-align: center;
                border: 1px solid #ddd;
            }}
        </style>
        {current_info.to_html(escape=False, index=False) if not current_info.empty else "未檢測到球員"}
        '''

        return html_table

    def update_player_status(self, detected_players):
        """更新球員歷史狀態"""
        messages = []

        # 處理檢測到的球員
        for player in detected_players:
            team = player['team']
            number = player['number']
            player_info = player['info']

            player_key = f"{team}_{number}"

            # 只在首次檢測到球員時記錄
            if player_key not in self.session_player_history and not player_info.empty:
                # 獲取該球員的所有統計數據
                player_data = {
                    'Team': team,
                    'Pic': f'<img src="{self.tracker.get_player_image(team, player_info["Player"].iloc[0])}" style="width:50px; height:50px;">',
                    'Player': player_info['Player'].iloc[0],
                    'Number': int(player_info['Number'].iloc[0]),
                    'PTS': int(player_info['PTS'].iloc[0]),
                    'REB': int(player_info['REB'].iloc[0]),
                    'AST': int(player_info['AST'].iloc[0]),
                    'STL': int(player_info['STL'].iloc[0]),
                    'BLK': int(player_info['BLK'].iloc[0]),
                    'FGM': int(player_info['FGM'].iloc[0]),
                    'FGA': int(player_info['FGA'].iloc[0]),
                    'FG%': float(player_info['FG%'].iloc[0]),
                    '3PM': int(player_info['3PM'].iloc[0]),
                    '3PA': int(player_info['3PA'].iloc[0]),
                    '3PT%': float(player_info['3PT%'].iloc[0]),
                    '4PM': int(player_info['4PM'].iloc[0]),
                    '4PA': int(player_info['4PA'].iloc[0]),
                    '4PT%': float(player_info['4PT%'].iloc[0])
                }
                self.session_player_history[player_key] = player_data
                messages.append(f"識別到 {team} 的背號為 {number} 的球員")

        # 生成排序後的列表
        sorted_players = sorted(
            self.session_player_history.values(),
            key=lambda x: (
                0 if x['Team'] == '3-s' else 1,  # 3-s 隊排在前面
                x['Number']  # 按號碼排序
            )
        )

        # 生成 HTML 表格
        if sorted_players:
            html = '''
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
            </style>
            <table>
                <thead>
                    <tr>
                        <th>Team</th>
                        <th>Pic</th>
                        <th>Player</th>
                        <th>Number</th>
                        <th>PTS</th>
                        <th>REB</th>
                        <th>AST</th>
                        <th>STL</th>
                        <th>BLK</th>
                        <th>FGM</th>
                        <th>FGA</th>
                        <th>FG%</th>
                        <th>3PM</th>
                        <th>3PA</th>
                        <th>3PT%</th>
                        <th>4PM</th>
                        <th>4PA</th>
                        <th>4PT%</th>
                    </tr>
                </thead>
                <tbody>
            '''

            for player in sorted_players:
                html += f'''
                    <tr>
                        <td>{player['Team']}</td>
                        <td>{player['Pic']}</td>
                        <td>{player['Player']}</td>
                        <td>{player['Number']}</td>
                        <td>{player['PTS']}</td>
                        <td>{player['REB']}</td>
                        <td>{player['AST']}</td>
                        <td>{player['STL']}</td>
                        <td>{player['BLK']}</td>
                        <td>{player['FGM']}</td>
                        <td>{player['FGA']}</td>
                        <td>{player['FG%']:.1f}</td>
                        <td>{player['3PM']}</td>
                        <td>{player['3PA']}</td>
                        <td>{player['3PT%']:.1f}</td>
                        <td>{player['4PM']}</td>
                        <td>{player['4PA']}</td>
                        <td>{player['4PT%']:.1f}</td>
                    </tr>
                '''

            html += '''
                </tbody>
            </table>
            '''
        else:
            html = "無歷史資料"

        return html, "; ".join(messages)

    def create_interface(self):
        """創建 Gradio 介面"""
        with gr.Blocks(title="智慧籃球分析平台") as self.interface:
            gr.Markdown("# 智慧籃球分析平台")

            with gr.Row():
                with gr.Column(scale=2):
                    video_input = gr.Video(label="選擇影片", height=300)

                with gr.Column(scale=1):
                    gr.Markdown("### 追蹤選項")
                    player_check = gr.Checkbox(label="Player", value=True)
                    ball_check = gr.Checkbox(label="Ball", value=True)
                    team_check = gr.Checkbox(label="Team", value=True)
                    number_check = gr.Checkbox(label="Number", value=True)

            with gr.Row():
                start_btn = gr.Button("開始播放")
                play_pause_btn = gr.Button("暫停", interactive=True)
                prev_frame_btn = gr.Button("上一幀", interactive=True)
                next_frame_btn = gr.Button("下一幀", interactive=True)
                # 添加性能報告按鈕
                perf_report_btn = gr.Button("顯示性能報告")

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Tabs() as tabs:
                        with gr.Tab("原始影片"):
                            original_output = gr.Image(label="原始影片", streaming=True)
                with gr.Column(scale=1):
                    with gr.Tabs() as tabs:
                        with gr.Tab("追蹤預測"):
                            processed_output = gr.Image(label="預測結果", streaming=True)
                        with gr.Tab("球場視圖"):
                            court_view = gr.Image(label="球場平面圖", streaming=True)

            # 添加性能報告文本框
            performance_textbox = gr.Textbox(label="性能報告", lines=10)

            self.scores_html = gr.HTML()

            def score_callback(scores):
                return self.update_scores(scores)

            self.tracker.set_score_callback(score_callback)

            gr.Markdown("### 球員當前狀態")
            player_current_status = gr.HTML(label="球員當前狀態")

            gr.Markdown("### 球員歷史狀態")
            player_history_status = gr.HTML(label="球員歷史狀態")

            gr.Markdown("### 識別結果")
            detection_message = gr.Textbox(label="識別結果")

            def enable_buttons():
                return {
                    play_pause_btn: gr.Button(interactive=True),
                    prev_frame_btn: gr.Button(interactive=True),
                    next_frame_btn: gr.Button(interactive=True)
                }

            # 性能報告按鈕的點擊事件
            def get_performance_report():
                self.tracker.print_performance_report()  # 在控制台打印
                stats = self.tracker.get_performance_stats()

                # 格式化報告文本
                report = "=== Performance Report ===\n"
                report += f"Total Frames Processed: {stats.get('total_frames', 0)}\n"
                report += f"Total Processing Time: {stats.get('total_time', 0):.2f} seconds\n"
                report += f"Average Time per Frame: {stats.get('avg_time_per_frame', 0) * 1000:.2f} ms\n"
                report += f"Average FPS: {stats.get('fps', 0):.2f}\n\n"

                report += "Breakdown by Steps:\n"
                for key, value in stats.items():
                    if key.endswith('_avg'):
                        step = key.replace('_avg', '')
                        report += f"\n{step}:\n"
                        report += f"  Average: {value * 1000:.2f} ms\n"
                        report += f"  Maximum: {stats.get(f'{step}_max', 0) * 1000:.2f} ms\n"
                        report += f"  Minimum: {stats.get(f'{step}_min', 0) * 1000:.2f} ms\n"

                return report

            # 開始按鈕事件
            start_btn.click(
                fn=self.process_video_realtime,
                inputs=[video_input],
                outputs=[
                    original_output, processed_output, court_view,
                    player_current_status, player_history_status,
                    detection_message, play_pause_btn, self.scores_html
                ]
            ).then(
                fn=enable_buttons,
                outputs=[play_pause_btn, prev_frame_btn, next_frame_btn]
            )

            # 播放/暫停按鈕事件
            play_pause_btn.click(
                fn=self.handle_play_pause,
                outputs=[play_pause_btn]
            )

            # 上一幀按鈕事件
            prev_frame_btn.click(
                fn=self.handle_prev_frame,
                outputs=[
                    original_output, processed_output, court_view,
                    player_current_status, player_history_status,
                    detection_message, play_pause_btn, self.scores_html
                ]
            )

            # 下一幀按鈕事件
            next_frame_btn.click(
                fn=self.handle_next_frame,
                outputs=[
                    original_output, processed_output, court_view,
                    player_current_status, player_history_status,
                    detection_message, play_pause_btn, self.scores_html
                ]
            )

            # # 性能報告按鈕事件
            perf_report_btn.click(
                fn=get_performance_report,
                outputs=[performance_textbox]
            )

            # 追蹤選項變更事件
            for check in [player_check, ball_check, team_check, number_check]:
                check.change(
                    fn=self.tracker.update_display_options,
                    inputs=[player_check, ball_check, team_check, number_check],
                    outputs=None
                )

        return self.interface

    def launch(self):
        """啟動介面"""
        if self.interface is None:
            self.create_interface()
        self.interface.launch(share=True)