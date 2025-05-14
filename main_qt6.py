import os
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                            QWidget, QPushButton, QCheckBox, QLabel, QTabWidget,
                            QTextEdit, QFileDialog, QGroupBox, QGridLayout,
                            QScrollArea, QSizePolicy, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMutex
from PyQt6.QtGui import QPixmap, QImage, QFont

import cv2
import numpy as np
import pandas as pd
import multiprocessing as mp
from models.basketball_tracker import BasketballTracker

# --- VideoThread class remains the same ---
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(object, object, object, list)
    status_update_signal = pyqtSignal(dict)

    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker
        self.is_running = False
        self.is_paused = False
        self._mutex = QMutex()

    def run(self):
        self.is_running = True
        while self.is_running:
            if not self.is_paused:
                processed_frame, court_frame, detected_players = self.tracker.next_frame()
                if processed_frame is None:
                    self.change_pixmap_signal.emit(None, None, None, [])
                    break
                display_frame = self.tracker.get_current_frame()
                scores = self.tracker.get_current_scores()
                self.change_pixmap_signal.emit(display_frame, processed_frame, court_frame, detected_players)
                self.status_update_signal.emit(scores)
            self.msleep(30)

    def stop(self):
        self.is_running = False
        self.wait()
    def pause(self): self.is_paused = True
    def play(self): self.is_paused = False
    def toggle_play_pause(self):
        self.is_paused = not self.is_paused
        return "暫停" if not self.is_paused else "播放"

class PyQtInterface(QMainWindow):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker
        self.video_thread = None
        self.session_player_history = {}

        self.setWindowTitle("智慧籃球分析平台")
        self.setGeometry(50, 50, 1400, 800)

        # Initialize UI elements
        self.video_path_label = QLabel("未選擇影片")
        self.select_video_btn = QPushButton("選擇影片檔案")
        self.player_check = QCheckBox("Player")
        self.ball_check = QCheckBox("Ball")
        self.team_check = QCheckBox("Team")
        self.number_check = QCheckBox("Number")
        self.start_btn = QPushButton("開始播放")
        self.play_pause_btn = QPushButton("暫停")
        self.prev_frame_btn = QPushButton("上一幀")
        self.next_frame_btn = QPushButton("下一幀")
        self.perf_report_btn = QPushButton("顯示性能報告")

        self.original_label = QLabel()
        self.processed_label = QLabel()
        self.court_label = QLabel()
        
        self.scores_label = QLabel("比分板")

        self.player_current_status_title = QLabel("球員當前狀態")
        self.player_current_status_content = QTextEdit()
        self.player_history_status_title = QLabel("球員歷史狀態")
        self.player_history_status_content = QTextEdit()
        self.detection_message_title = QLabel("識別結果")
        self.detection_message_content = QTextEdit()
        self.performance_textbox_title = QLabel("性能報告")
        self.performance_textbox_content = QTextEdit()

        self.init_ui()

    def create_section_title_label(self, text):
        label = QLabel(text)
        label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #ddd;
                padding: 8px 5px;
                margin-top: 10px;
                border-top: 1px solid #4a4a4f;
                background-color: transparent;
            }
        """)
        return label

    def init_ui(self):
        main_content_widget = QWidget()
        main_layout = QVBoxLayout(main_content_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(10,10,10,10)

        # --- 1. Top Area: Video Selection (Left) & Tracking Options (Right) ---
        top_area_layout = QHBoxLayout()
        top_area_layout.setSpacing(10)

        video_group = QGroupBox("影片選擇")
        video_layout = QVBoxLayout(video_group)
        self.video_path_label.setWordWrap(True)
        self.video_path_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_path_label.setMinimumHeight(40)
        video_layout.addWidget(self.video_path_label)
        self.select_video_btn.clicked.connect(self.select_video)
        video_layout.addWidget(self.select_video_btn)
        top_area_layout.addWidget(video_group, 7)

        tracking_group = QGroupBox("追蹤選項")
        tracking_grid_layout = QGridLayout(tracking_group)
        self.player_check.setChecked(True)
        self.ball_check.setChecked(True)
        self.team_check.setChecked(True)
        self.number_check.setChecked(True)
        tracking_grid_layout.addWidget(self.player_check, 0, 0)
        tracking_grid_layout.addWidget(self.ball_check, 0, 1)
        tracking_grid_layout.addWidget(self.team_check, 1, 0)
        tracking_grid_layout.addWidget(self.number_check, 1, 1)
        for check in [self.player_check, self.ball_check, self.team_check, self.number_check]:
            check.stateChanged.connect(self.update_display_options)
        top_area_layout.addWidget(tracking_group, 3)
        main_layout.addLayout(top_area_layout)

        # --- 2. Playback Buttons ---
        buttons_layout = QHBoxLayout()
        button_height = 35
        self.start_btn.clicked.connect(self.start_video); self.start_btn.setFixedHeight(button_height)
        buttons_layout.addWidget(self.start_btn)
        self.play_pause_btn.clicked.connect(self.toggle_play_pause); self.play_pause_btn.setEnabled(False); self.play_pause_btn.setFixedHeight(button_height)
        buttons_layout.addWidget(self.play_pause_btn)
        self.prev_frame_btn.clicked.connect(self.prev_frame); self.prev_frame_btn.setEnabled(False); self.prev_frame_btn.setFixedHeight(button_height)
        buttons_layout.addWidget(self.prev_frame_btn)
        self.next_frame_btn.clicked.connect(self.next_frame); self.next_frame_btn.setEnabled(False); self.next_frame_btn.setFixedHeight(button_height)
        buttons_layout.addWidget(self.next_frame_btn)
        self.perf_report_btn.clicked.connect(self.show_performance_report); self.perf_report_btn.setFixedHeight(button_height)
        buttons_layout.addWidget(self.perf_report_btn)
        main_layout.addLayout(buttons_layout)

        # --- 3. Video Display Area: Original (Left) & Analysis (Right) ---
        video_display_layout = QHBoxLayout()
        video_display_layout.setSpacing(10)
        video_label_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        for label in [self.original_label, self.processed_label, self.court_label]:
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("background-color: #000; color: #555; border: 1px solid #333; min-height: 200px;")
            label.setSizePolicy(video_label_policy)
            label.setMinimumSize(300, 200)

        self.original_label.setText("原始影片")
        video_display_layout.addWidget(self.original_label, 1)

        analysis_tabs = QTabWidget()
        analysis_tabs.setSizePolicy(video_label_policy)
        self.processed_label.setText("追蹤預測")
        analysis_tabs.addTab(self.processed_label, "追蹤預測")
        self.court_label.setText("球場視圖")
        analysis_tabs.addTab(self.court_label, "球場視圖")
        video_display_layout.addWidget(analysis_tabs, 1)
        main_layout.addLayout(video_display_layout)
        
        # --- 4. Scoreboard (改為水平排列) ---
        self.scores_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scores_label.setStyleSheet("background-color: #2a2f38; color: white; padding: 8px; border-radius: 4px; margin-top: 8px;")
        self.scores_label.setWordWrap(True)
        self.scores_label.setMinimumHeight(120)
        main_layout.addWidget(self.scores_label)

        # --- 5. Current Player Status ---
        self.player_current_status_title = self.create_section_title_label("球員當前狀態")
        main_layout.addWidget(self.player_current_status_title)
        self.player_current_status_content.setReadOnly(True)
        self.player_current_status_content.setMinimumHeight(120)
        self.player_current_status_content.setStyleSheet("QTextEdit { border: 1px solid #3c3c3c; background-color: #1e1e1e; color: #e0e0e0; }")
        self.player_current_status_content.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.player_current_status_content.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_layout.addWidget(self.player_current_status_content)

        # --- 6. History Player Status ---
        self.player_history_status_title = self.create_section_title_label("球員歷史狀態")
        main_layout.addWidget(self.player_history_status_title)
        self.player_history_status_content.setReadOnly(True)
        self.player_history_status_content.setMinimumHeight(180)
        self.player_history_status_content.setStyleSheet("QTextEdit { border: 1px solid #3c3c3c; background-color: #1e1e1e; color: #e0e0e0; }")
        self.player_history_status_content.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.player_history_status_content.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_layout.addWidget(self.player_history_status_content)

        # --- 7. Detection Message ---
        self.detection_message_title = self.create_section_title_label("識別結果")
        main_layout.addWidget(self.detection_message_title)
        self.detection_message_content.setReadOnly(True)
        self.detection_message_content.setFixedHeight(50)
        self.detection_message_content.setStyleSheet("QTextEdit { border: 1px solid #3c3c3c; background-color: #1e1e1e; color: #e0e0e0; }")
        main_layout.addWidget(self.detection_message_content)

        # --- 8. Performance Report ---
        self.performance_textbox_title = self.create_section_title_label("性能報告")
        main_layout.addWidget(self.performance_textbox_title)
        self.performance_textbox_content.setReadOnly(True)
        self.performance_textbox_content.setMinimumHeight(120)
        self.performance_textbox_content.setFont(QFont("Courier New", 8))
        self.performance_textbox_content.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.performance_textbox_content.setStyleSheet("QTextEdit { border: 1px solid #3c3c3c; background-color: #1e1e1e; color: #e0e0e0; }")
        main_layout.addWidget(self.performance_textbox_content)
        
        main_layout.addStretch(1)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(main_content_widget)
        self.setCentralWidget(scroll_area)

    def select_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "選擇影片檔案", "", "影片檔案 (*.mp4 *.avi *.mov)")
        if file_name:
            self.video_path = file_name
            self.video_path_label.setText(f"已選擇: {os.path.basename(file_name)}")
            self.start_btn.setEnabled(True)
            self.original_label.clear(); self.original_label.setText("原始影片")
            self.processed_label.clear(); self.processed_label.setText("追蹤預測")
            self.court_label.clear(); self.court_label.setText("球場視圖")
            self.scores_label.setText("比分板")
            self.player_current_status_content.clear()
            self.player_history_status_content.clear()
            self.detection_message_content.clear()
            self.performance_textbox_content.clear()
            self.session_player_history = {}
            if self.video_thread and self.video_thread.isRunning(): self.video_thread.stop()
            self.play_pause_btn.setEnabled(False); self.prev_frame_btn.setEnabled(False); self.next_frame_btn.setEnabled(False)

    def start_video(self):
        if not hasattr(self, 'video_path'): self.detection_message_content.setText("請先選擇影片檔案"); return
        if not self.tracker.load_video(self.video_path): self.detection_message_content.setText("無法開啟影片"); return
        self.session_player_history = {}
        if self.video_thread is not None: self.video_thread.stop()
        self.video_thread = VideoThread(self.tracker)
        self.video_thread.change_pixmap_signal.connect(self.update_frames)
        self.video_thread.status_update_signal.connect(self.update_scores)
        self.video_thread.finished.connect(self.video_finished_actions)
        self.video_thread.start()
        self.play_pause_btn.setEnabled(True); self.play_pause_btn.setText("暫停")
        self.prev_frame_btn.setEnabled(True); self.next_frame_btn.setEnabled(True)
        self.detection_message_content.setText("影片播放中...")

    def video_finished_actions(self):
        self.detection_message_content.setText("影片播放完畢或已停止")
        self.play_pause_btn.setText("播放")

    def toggle_play_pause(self):
        if self.video_thread is not None and self.video_thread.isRunning():
            button_text = self.video_thread.toggle_play_pause()
            self.play_pause_btn.setText(button_text)
            self.detection_message_content.setText("影片已暫停" if self.video_thread.is_paused else "影片播放中...")
        elif hasattr(self, 'video_path') and self.tracker.video_capture and not self.tracker.video_capture.isOpened():
             self.start_video()

    def prev_frame(self):
        if not hasattr(self.tracker, 'video_capture') or self.tracker.video_capture is None:
             self.detection_message_content.setText("請先載入影片"); return
        if self.video_thread is not None and self.video_thread.isRunning() and not self.video_thread.is_paused:
            self.video_thread.pause(); self.play_pause_btn.setText("播放")
        processed_frame, court_frame, detected_players = self.tracker.prev_frame()
        if processed_frame is not None:
            display_frame = self.tracker.get_current_frame()
            self.update_frames(display_frame, processed_frame, court_frame, detected_players)
            self.update_scores(self.tracker.get_current_scores())
            self.detection_message_content.setText(f"上一幀 (幀: {self.tracker.current_frame_index if hasattr(self.tracker, 'current_frame_index') else 'N/A'})")
        else: self.detection_message_content.setText("已是第一幀")

    def next_frame(self):
        if not hasattr(self.tracker, 'video_capture') or self.tracker.video_capture is None:
             self.detection_message_content.setText("請先載入影片"); return
        if self.video_thread is not None and self.video_thread.isRunning() and not self.video_thread.is_paused:
            self.video_thread.pause(); self.play_pause_btn.setText("播放")
        processed_frame, court_frame, detected_players = self.tracker.next_frame()
        if processed_frame is not None:
            display_frame = self.tracker.get_current_frame()
            self.update_frames(display_frame, processed_frame, court_frame, detected_players)
            self.update_scores(self.tracker.get_current_scores())
            self.detection_message_content.setText(f"下一幀 (幀: {self.tracker.current_frame_index if hasattr(self.tracker, 'current_frame_index') else 'N/A'})")
        else: self.detection_message_content.setText("已是最後一幀")

    def _update_video_label(self, label_widget: QLabel, frame_data: np.ndarray, placeholder_text: str):
        if frame_data is not None and isinstance(frame_data, np.ndarray) and frame_data.size > 0 :
            try:
                rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                
                scaled_pixmap = pixmap.scaled(
                    label_widget.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
                label_widget.setPixmap(scaled_pixmap)

            except Exception as e:
                print(f"Error updating {label_widget.objectName() if label_widget.objectName() else 'label'}: {e}")
                label_widget.setText(placeholder_text + f"\nError: {e}")
        else:
            if label_widget.pixmap() is not None and not label_widget.pixmap().isNull():
                label_widget.clear()
            label_widget.setText(placeholder_text)

    def update_frames(self, display_frame, processed_frame, court_frame, detected_players):
        self._update_video_label(self.original_label, display_frame, "原始影片")
        self._update_video_label(self.processed_label, processed_frame, "追蹤預測")
        self._update_video_label(self.court_label, court_frame, "球場視圖")
        if detected_players is not None: self.update_player_status(detected_players)

    def update_scores(self, scores):
        """記分板 — 改用 HTML table 左右並排（較舊寫法）"""
        try:
            team_a_id, team_b_id = "3-s", "biv"
            team_a_name, team_b_name = "3'S COMPANY", "BIVOUAC"
            team_a_score = scores.get(team_a_id, 0)
            team_b_score = scores.get(team_b_id, 0)
            team_a_logo = self.tracker.get_player_image(team_a_id, team_a_id)
            team_b_logo = self.tracker.get_player_image(team_b_id, team_b_id)

            html = f"""
            <table align="center" cellpadding="10" cellspacing="0" bgcolor="#2a2f38">
              <tr>
                <td align="center">
                  <img src="{team_a_logo}" width="80" height="80" alt="{team_a_name}"><br>
                  <span style="font-size:13px;color:#999;text-transform:uppercase;">{team_a_name}</span><br>
                  <span style="font-size:48px;color:#61dafb;font-weight:bold;">{team_a_score}</span>
                </td>
                <td align="center" style="padding:0 20px;">
                  <span style="font-size:24px;color:#fff;font-weight:bold;">VS</span>
                </td>
                <td align="center">
                  <img src="{team_b_logo}" width="80" height="80" alt="{team_b_name}"><br>
                  <span style="font-size:13px;color:#999;text-transform:uppercase;">{team_b_name}</span><br>
                  <span style="font-size:48px;color:#ff6961;font-weight:bold;">{team_b_score}</span>
                </td>
              </tr>
            </table>
            """
            self.scores_label.setTextFormat(Qt.TextFormat.RichText)
            self.scores_label.setText(html)
        except Exception as e:
            self.scores_label.setTextFormat(Qt.TextFormat.RichText)
            self.scores_label.setText(f"<p style='color:red;text-align:center;'>更新錯誤: {e}</p>")


    def update_display_options(self):
        if hasattr(self, 'tracker') and self.tracker:
            self.tracker.update_display_options(
                self.player_check.isChecked(), self.ball_check.isChecked(),
                self.team_check.isChecked(), self.number_check.isChecked())

    def update_current_status(self, detected_players):
        """球員當前狀態 — HTML table，不用 pandas，體力條改為進度條樣式"""
        entries = []
        onerror = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        seen = set()

        for p in detected_players:
            team, num, info = p['team'], p['number'], p['info']
            key = f"{team}_{num}"
            if key in seen or info.empty:
                continue
            seen.add(key)
            name = info['Player'].iloc[0]
            pic = self.tracker.get_player_image(team, name)
            stamina = self.tracker.player_states.get(key, {}).get('stamina', 100)

            # — 修改：進度條樣式（帶邊框、百分比文字置中）
            col = '#ff0000' if stamina<=25 else '#ffd700' if stamina<=50 else '#00ff00' if stamina<=75 else '#0000ff'
            bar = f"""
              <div style="
                width:100px; height:14px;
                border:1px solid #888; border-radius:4px;
                background:#444; overflow:hidden; margin:auto;
              ">
                <div style="
                  width:{stamina}%; height:100%;
                  background:{col}; text-align:center;
                  line-height:14px; color:#fff; font-size:10px;
                ">
                  {stamina:.0f}%
                </div>
              </div>
            """

            entries.append({
                'Team': team,
                'Pic': f'<img src="{pic}" width="30" height="30" style="object-fit:cover;border-radius:2px;" onerror="this.src=\'{onerror}\';">',
                'Player': name,
                'Number': int(num),
                'Stamina': bar
            })

        if not entries:
            html = "<p style='text-align:center;color:#666;'>未檢測到球員</p>"
        else:
            html = """
            <table width="100%" border="1" cellspacing="0" cellpadding="4" bgcolor="#1e1e1e">
              <tr>
                <th bgcolor="#2a2f38" style="color:white;">Team</th>
                <th bgcolor="#2a2f38" style="color:white;">Pic</th>
                <th bgcolor="#2a2f38" style="color:white;">Player</th>
                <th bgcolor="#2a2f38" style="color:white;">Number</th>
                <th bgcolor="#2a2f38" style="color:white;">Stamina</th>
              </tr>
            """
            for e in sorted(entries, key=lambda x: (x['Team'], x['Number'])):
                html += f"""
                <tr>
                  <td align="center">{e['Team']}</td>
                  <td align="center">{e['Pic']}</td>
                  <td align="center">{e['Player']}</td>
                  <td align="center">{e['Number']}</td>
                  <td align="center">{e['Stamina']}</td>
                </tr>
                """
            html += "</table>"

        self.player_current_status_content.setHtml(html)

    def update_current_status(self, detected_players):
        """球員當前狀態 — HTML table，不用 pandas，體力條採 Gradio 版寫法"""
        entries = []
        onerror_ph = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        seen = set()

        for p in detected_players:
            team, num, info = p['team'], p['number'], p['info']
            key = f"{team}_{num}"
            if key in seen or info.empty:
                continue
            seen.add(key)
            name = info['Player'].iloc[0]
            pic  = self.tracker.get_player_image(team, name)
            stamina = self.tracker.player_states.get(key, {}).get('stamina', 100)

            # — 修改：使用 Gradio 版進度條 HTML 寫法 :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
            bar = f"""
              <div style="
                width:80px; height:14px;
                background-color:#555; border-radius:3px;
                overflow:hidden; margin:auto;
              ">
                <div style="
                  width:{stamina}%; height:14px;
                  background-color:{ 
                    '#ff0000' if stamina<=25 else 
                    '#ffd700' if stamina<=50 else 
                    '#00ff00' if stamina<=75 else 
                    '#0000ff'
                  };
                  text-align:center; color:#fff;
                  font-size:10px; line-height:14px;
                ">{stamina:.1f}%</div>
              </div>
            """

            entries.append({
                'Team':    team,
                'Pic':     f'<img src="{pic}" width="30" height="30" style="object-fit:cover;border-radius:2px;" onerror="this.src=\'{onerror_ph}\';">',
                'Player':  name,
                'Number':  int(num),
                'Stamina': bar
            })

        if not entries:
            html = "<p style='text-align:center;color:#666;'>未檢測到球員</p>"
        else:
            html = """
            <table width="100%" border="1" cellspacing="0" cellpadding="4" bgcolor="#1e1e1e">
              <tr>
                <th bgcolor="#2a2f38" style="color:white;">Team</th>
                <th bgcolor="#2a2f38" style="color:white;">Pic</th>
                <th bgcolor="#2a2f38" style="color:white;">Player</th>
                <th bgcolor="#2a2f38" style="color:white;">Number</th>
                <th bgcolor="#2a2f38" style="color:white;">Stamina</th>
              </tr>
            """
            for e in sorted(entries, key=lambda x:(x['Team'], x['Number'])):
                html += f"""
                <tr>
                  <td align="center">{e['Team']}</td>
                  <td align="center">{e['Pic']}</td>
                  <td align="center">{e['Player']}</td>
                  <td align="center">{e['Number']}</td>
                  <td align="center">{e['Stamina']}</td>
                </tr>
                """
            html += "</table>"

        self.player_current_status_content.setHtml(html)

    def update_player_status(self, detected_players):
        """球員歷史狀態 — HTML table，不用 pandas，圖片用 width/height 固定尺寸"""
        # 先更新當前狀態
        self.update_current_status(detected_players)

        onerror = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        new_msgs = []

        for p in detected_players:
            team, num, info = p['team'], p['number'], p['info']
            key = f"{team}_{num}"
            if key in self.session_player_history or info.empty:
                continue
            name = info['Player'].iloc[0]
            pic = self.tracker.get_player_image(team, name)
            img_html = f'<img src="{pic}" width="30" height="30" style="object-fit:cover;border-radius:2px;" onerror="this.src=\'{onerror}\';">'
            self.session_player_history[key] = {
                'Team': team,
                'Pic': img_html,
                'Player': name,
                'Number': int(info['Number'].iloc[0]),
                'PTS': int(info['PTS'].iloc[0]),
                'REB': int(info['REB'].iloc[0]),
                'AST': int(info['AST'].iloc[0]),
                'STL': int(info['STL'].iloc[0]),
                'BLK': int(info['BLK'].iloc[0]),
                'FGM': int(info['FGM'].iloc[0]),
                'FGA': int(info['FGA'].iloc[0]),
                'FG%': f"{info['FG%'].iloc[0]*100:.1f}%",
                '3PM': int(info['3PM'].iloc[0]),
                '3PA': int(info['3PA'].iloc[0]),
                '3PT%': f"{info['3PT%'].iloc[0]*100:.1f}%",
                '4PM': int(info['4PM'].iloc[0]),
                '4PA': int(info['4PA'].iloc[0]),
                '4PT%': f"{info['4PT%'].iloc[0]*100:.1f}%"
            }
            new_msgs.append(f"識別到 {team} {name}({num}號)")

        hist = sorted(self.session_player_history.values(),
                      key=lambda x:(0 if x['Team']=='3-s' else 1, x['Number']))

        if not hist:
            html = "<p style='text-align:center;color:#666;'>無歷史資料</p>"
        else:
            html = """
            <table width="100%" border="1" cellspacing="0" cellpadding="3" bgcolor="#1b1b1b">
              <tr>
                <th bgcolor="#2a2f38" style="color:white;">Team</th>
                <th bgcolor="#2a2f38" style="color:white;">Pic</th>
                <th bgcolor="#2a2f38" style="color:white;">Player</th>
                <th bgcolor="#2a2f38" style="color:white;">Number</th>
                <th bgcolor="#2a2f38" style="color:white;">PTS</th>
                <th bgcolor="#2a2f38" style="color:white;">REB</th>
                <th bgcolor="#2a2f38" style="color:white;">AST</th>
                <th bgcolor="#2a2f38" style="color:white;">STL</th>
                <th bgcolor="#2a2f38" style="color:white;">BLK</th>
                <th bgcolor="#2a2f38" style="color:white;">FGM</th>
                <th bgcolor="#2a2f38" style="color:white;">FGA</th>
                <th bgcolor="#2a2f38" style="color:white;">FG%</th>
                <th bgcolor="#2a2f38" style="color:white;">3PM</th>
                <th bgcolor="#2a2f38" style="color:white;">3PA</th>
                <th bgcolor="#2a2f38" style="color:white;">3PT%</th>
                <th bgcolor="#2a2f38" style="color:white;">4PM</th>
                <th bgcolor="#2a2f38" style="color:white;">4PA</th>
                <th bgcolor="#2a2f38" style="color:white;">4PT%</th>
              </tr>
            """
            for e in hist:
                html += "<tr>" + "".join(f"<td align='center'>{e[col]}</td>" for col in
                    ['Team','Pic','Player','Number','PTS','REB','AST','STL','BLK',
                     'FGM','FGA','FG%','3PM','3PA','3PT%','4PM','4PA','4PT%']
                ) + "</tr>"
            html += "</table>"

        self.player_history_status_content.setHtml(html)
        if new_msgs:
            self.detection_message_content.setText("\n".join(new_msgs))

    def show_performance_report(self):
        if not hasattr(self, 'tracker'): return
        self.tracker.print_performance_report()
        stats = self.tracker.get_performance_stats()
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
        self.performance_textbox_content.setText(report)

    def closeEvent(self, event):
        if self.video_thread is not None: self.video_thread.stop()
        event.accept()

def main():
    data_folder = 'big3_data'
    if not os.path.exists(data_folder): os.makedirs(data_folder, exist_ok=True)
    try:
        app = QApplication(sys.argv)
        app.setStyleSheet("""
            QWidget { color: #e0e0e0; background-color: #1e1e1e; }
            QMainWindow { background-color: #1e1e1e; }
            QGroupBox { 
                border: 1px solid #3c3c3c; margin-top: 10px; 
                padding: 10px 5px 5px 5px;
                background-color: #252526; border-radius: 4px;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; subcontrol-position: top left; 
                padding: 0px 8px;
                margin-left: 5px;
                color: #cccccc;
            }
            QLabel { background-color: transparent; }
            QPushButton { background-color: #3e3e42; border: 1px solid #303030; padding: 5px 10px; border-radius: 3px; color: #f0f0f0;}
            QPushButton:hover { background-color: #4a4a4f; }
            QPushButton:pressed { background-color: #55555a; }
            QPushButton:disabled { color: #777; background-color: #2d2d2d; }
            QCheckBox { spacing: 5px; background-color:transparent; }
            QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #555; border-radius: 3px;}
            QCheckBox::indicator:checked { background-color: #61dafb; border-color: #61dafb; }
            QTextEdit { background-color: #252526; border: 1px solid #3c3c3c; border-radius: 3px; padding: 3px; color: #e0e0e0; }
            QTabWidget::pane { border: 1px solid #3c3c3c; background-color: #252526; }
            QTabBar::tab { background: #3e3e42; border: 1px solid #3c3c3c; border-bottom: none; padding: 6px 12px; border-top-left-radius: 3px; border-top-right-radius: 3px; color: #f0f0f0; margin-right: 1px;}
            QTabBar::tab:selected { background: #252526; color: #61dafb;}
            QTabBar::tab:hover { background: #4a4a4f; }
            QScrollArea { border: none; background-color: #1e1e1e; }
            QScrollBar:vertical { border: none; background: #2d2d2d; width: 12px; margin: 0px; border-radius: 6px; }
            QScrollBar::handle:vertical { background: #505050; min-height: 25px; border-radius: 6px;}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { border: none; background: none; height:0px; }
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical { background: none; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
            QLineEdit { padding: 2px; border: 1px solid #3c3c3c; border-radius: 3px; background: #2d2d2d;}
        """)
        tracker = BasketballTracker(player_model_path='best_demo_v2.pt', court_model_path='Court_best.pt', data_folder=data_folder)
        tracker.set_court_reference('court_pic.jpg')
        interface = PyQtInterface(tracker)
        interface.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"程序運行時發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if sys.platform.startswith('win') or sys.platform.startswith('darwin'):
        try: mp.set_start_method('spawn', force=True)
        except RuntimeError: print("警告: MP 'spawn' failed.")
    main()