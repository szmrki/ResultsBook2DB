# Copyright (C) 2026 szmrki
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QFileDialog, QMessageBox, QFormLayout, QGroupBox, 
                             QRadioButton, QButtonGroup, QProgressBar,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QAbstractItemView, QPlainTextEdit, QSizePolicy)
from PySide6.QtCore import Qt, Signal, QTimer, QSize
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QMouseEvent, QFont, QColor, QTextCursor, QCursor
from pathlib import Path
from create_db import set_tables
from worker import Worker
import multiprocessing
import logging
from logger_config import setup_logging, add_qt_handler

# ロギングの初期化（戻り値の log_file_path を add_qt_handler に渡す用に保持）
LOG_FILE_PATH = setup_logging()
logger = logging.getLogger(__name__)
logger.info("Application starting...")

# ---------------------------------------------------------
# ドラッグ＆ドロップ専用のカスタムラベルを作成
# ---------------------------------------------------------
class FileDropLabel(QLabel):
    # ファイルがドロップされたことをメイン画面に知らせるシグナル
    filesDropped = Signal(list)
    def __init__(self) -> None:
        super().__init__()
        self.setText("\nここにPDFファイルをドラッグ&ドロップ\nまたはクリックして選択\n")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(100)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # ドロップを受け付ける設定
        self.setAcceptDrops(True)
        self.update_style(False)

    def update_style(self, active: bool) -> None:
        if active:
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #0078D7;
                    border-radius: 8px;
                    background-color: #eaf4ff;
                    color: #0078D7;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)
        else:
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #ccc;
                    border-radius: 8px;
                    background-color: #ffffff;
                    color: #666;
                    font-size: 14px;
                }
                QLabel:hover {
                    border-color: #0078D7;
                    background-color: #f0f7ff;
                }
            """)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.accept()
            self.update_style(True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self.update_style(False)

    def dropEvent(self, event: QDropEvent) -> None:
        """ドロップされた時の処理（複数ファイル対応）"""
        urls = event.mimeData().urls()
        if urls:
            pdf_files = []
            for url in urls:
                file_path = url.toLocalFile()
                if file_path.lower().endswith('.pdf'):
                    pdf_files.append(file_path)
            
            if pdf_files:
                self.process_files(pdf_files)
            else:
                self.setText("\nError:\nPDFファイルのみ対応しています\n")
                self.setStyleSheet("""
                QLabel {
                    border: 2px dashed red;
                    border-radius: 10px;
                    background-color: #ffeeee;
                    font-size: 14px;
                }
                """)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """クリックされた時の処理（複数選択対応）"""
        super().mousePressEvent(event)
        file_names, _ = QFileDialog.getOpenFileNames(
            self,
            "PDFファイルを選択（複数選択可）",
            str(Path.cwd()),
            "PDF Files (*.pdf)"
        )
        if file_names:
            self.process_files(file_names)

    def process_files(self, file_paths: list) -> None:
        """ドロップとクリックの共通処理（複数ファイル）"""
        count = len(file_paths)
        self.setText(f"\n{count}個のPDFファイルが追加されました\nさらに追加するにはドラッグ&ドロップ\n")
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #0078D7;
                border-radius: 10px;
                background-color: #eaf4ff;
                font-size: 14px;
                color: black;
            }
        """)
        # メインウィンドウに通知
        self.filesDropped.emit(file_paths)
    
    def clear(self) -> None:
        """ラベルを初期状態に戻す"""
        self.setText("\nここにPDFファイルをドラッグ&ドロップ\nまたはクリックして選択\n")
        self.update_style(False)

#データベース
class DatabaseSelector(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_folder = str(Path.cwd()) # デフォルトはカレントディレクトリ

        # 全体をグループボックスで囲む
        main_layout = QVBoxLayout(self)
        group_box = QGroupBox("データベース設定")
        main_layout.addWidget(group_box)
        
        layout = QVBoxLayout(group_box)

        # ------------------------------------------------
        # 1. モード選択 (ラジオボタン)
        # ------------------------------------------------
        mode_layout = QHBoxLayout()
        self.radio_existing = QRadioButton("既存のDBファイルを選択")
        self.radio_new = QRadioButton("新規作成")
        self.radio_existing.setCursor(Qt.CursorShape.PointingHandCursor)
        self.radio_new.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # デフォルトは「既存」にしておく
        self.radio_existing.setChecked(True)
        
        self.btn_group = QButtonGroup()
        self.btn_group.addButton(self.radio_existing)
        self.btn_group.addButton(self.radio_new)

        mode_layout.addWidget(self.radio_existing)
        mode_layout.addWidget(self.radio_new)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # ------------------------------------------------
        # 2. 【既存モード】用のUIエリア
        # ------------------------------------------------
        self.existing_widget = QWidget()
        existing_layout = QHBoxLayout(self.existing_widget)
        existing_layout.setContentsMargins(0, 0, 0, 0)

        self.path_input_existing = QLineEdit()
        self.path_input_existing.setPlaceholderText("ファイルを選択してください...")
        self.path_input_existing.setReadOnly(True)
        
        self.btn_browse_existing = QPushButton("参照...")
        self.btn_browse_existing.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_browse_existing.clicked.connect(self.select_existing_file)

        existing_layout.addWidget(self.path_input_existing)
        existing_layout.addWidget(self.btn_browse_existing)
        
        layout.addWidget(self.existing_widget)

        # ------------------------------------------------
        # 3. 【新規モード】用のUIエリア
        # ------------------------------------------------
        self.new_widget = QWidget()
        self.new_widget.setVisible(False) # 最初は隠しておく
        new_layout = QFormLayout(self.new_widget)
        new_layout.setContentsMargins(0, 0, 0, 0)

        # A. 保存先フォルダ選択
        folder_layout = QHBoxLayout()
        self.folder_input = QLineEdit(self.current_folder)
        self.folder_input.setReadOnly(True)
        self.btn_browse_folder = QPushButton("フォルダ変更...")
        self.btn_browse_folder.setCursor(Qt.CursorShape.PointingHandCursor)
        #self.btn_browse_folder.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_browse_folder.clicked.connect(self.select_folder)
        
        folder_layout.addWidget(self.folder_input)
        folder_layout.addWidget(self.btn_browse_folder)
        new_layout.addRow("保存先:", folder_layout)

        # B. ファイル名入力
        filename_layout = QHBoxLayout()
        self.filename_input = QLineEdit()
        
        # 拡張子ラベル (.db)
        ext_label = QLabel(".db")
        ext_label.setStyleSheet("font-weight: bold; color: #333333;")
        
        filename_layout.addWidget(self.filename_input)
        filename_layout.addWidget(ext_label)
        new_layout.addRow("ファイル名:", filename_layout)

        layout.addWidget(self.new_widget)

        # シグナル接続 (モード切替時の処理)
        self.btn_group.buttonToggled.connect(self.switch_ui)

    def switch_ui(self, button, checked):
        """ラジオボタンの変更に合わせてUIを出し分ける"""
        if not checked: return
        if button == self.radio_existing:
            self.existing_widget.setVisible(True)
            self.new_widget.setVisible(False)
        else:
            self.existing_widget.setVisible(False)
            self.new_widget.setVisible(True)

    def select_existing_file(self):
        """既存ファイルの選択"""
        path, _ = QFileDialog.getOpenFileName(
            self, "データベースを選択", "", "SQLite DB (*.db *.sqlite);;All Files (*)"
        )
        if path:
            self.path_input_existing.setText(path)

    def select_folder(self):
        """新規作成時の保存先フォルダ選択"""
        path: str = QFileDialog.getExistingDirectory(self, "保存先フォルダを選択", self.current_folder)
        if path:
            self.current_folder = path
            self.folder_input.setText(path)

    def get_active_db_path(self) -> tuple[Path|None, bool]:
        """
            現在選択されているモードに基づいて、
            「データベースの絶対パス」を返す。
            Returns:
                tuple[Path|None, bool]: (データベースのパス or None, is_new_mode)
        """
        if self.radio_existing.isChecked():
            # 既存モード
            path = self.path_input_existing.text().strip()
            return (Path(path) if path else None, False)
        else:
            # 新規モード
            folder = self.folder_input.text().strip()
            name = self.filename_input.text().strip()
            if not folder or not name:
                return (None, True)
            
            # 拡張子補完
            path_name = Path(name)
            if not path_name.suffix.lower() == ".db":
                path_name = path_name.with_suffix(".db")

            full_path = Path(folder) / path_name
            return (full_path, True)

# ---------------------------------------------------------
# メインウィンドウ
# ---------------------------------------------------------
class MainWindow(QMainWindow):
    log_signal = Signal(str, int) # メッセージ, ログレベル

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ResultsBook2DB")
        self.setMinimumSize(700, 600)
        self.setup_styles()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # log_viewer を先に作成（add_qt_handler 内の logging で log_write が呼ばれるため）
        self.log_viewer = QPlainTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setPlaceholderText("ここに詳細なログが表示されます...")
        self.log_viewer.setStyleSheet("background-color: #2c3e50; color: #ecf0f1; font-family: 'Consolas', 'Monaco', monospace; font-size: 11px;")

        self.log_signal.connect(self._handle_log_signal)
        add_qt_handler(lambda msg, level: self.log_signal.emit(msg, level), log_file_path=LOG_FILE_PATH)

        self.file_entries = []

        # --- Step 1: Database Setup ---
        self.db_selector = DatabaseSelector()
        main_layout.addWidget(self.db_selector)

        # --- Step 2: PDF Selection ---
        step2_group = QGroupBox("PDFファイルの選択")
        step2_layout = QVBoxLayout(step2_group)
        
        self.drop_area = FileDropLabel()
        self.drop_area.filesDropped.connect(self.update_file_paths)
        step2_layout.addWidget(self.drop_area)

        # Table Header (label + buttons)
        table_header = QHBoxLayout()
        table_label = QLabel("選択済みファイル:")
        table_label.setStyleSheet("font-weight: bold; color: #333333;")
        table_header.addWidget(table_label)
        table_header.addStretch()

        self.clear_all_button = QPushButton("すべてクリア")
        self.clear_all_button.setObjectName("secondaryButton")
        self.clear_all_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_all_button.clicked.connect(self.clear_all_files)
        table_header.addWidget(self.clear_all_button)

        self.delete_button = QPushButton("選択削除")
        self.delete_button.setObjectName("dangerButton")
        self.delete_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_button.clicked.connect(self.delete_selected_files)
        table_header.addWidget(self.delete_button)
        
        step2_layout.addLayout(table_header)

        self.file_table = QTableWidget(0, 2)
        self.file_table.setHorizontalHeaderLabels(["ファイル名", "Event Name"])
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.file_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.file_table.setMaximumHeight(180)
        self.file_table.cellChanged.connect(self.on_table_cell_changed)
        step2_layout.addWidget(self.file_table)
        
        main_layout.addWidget(step2_group)

        # --- Step 3: Analysis Config & Run ---
        step3_group = QGroupBox("解析設定と実行")
        step3_layout = QVBoxLayout(step3_group)
        
        config_layout = QHBoxLayout()
        md_layout, self.md_btn_group = self.__set_radio_button("4人制", "MD", default=0)
        self.is_md = False
        self.md_btn_group.idClicked.connect(self.md_clicked)
        config_layout.addLayout(md_layout)
        config_layout.addStretch()
        step3_layout.addLayout(config_layout)

        self.run_button = QPushButton("解析を開始")
        self.run_button.setObjectName("primaryButton")
        self.run_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_button.setMinimumHeight(40)
        self.run_button.clicked.connect(self.start_analysis)
        step3_layout.addWidget(self.run_button)

        # プログレス領域を1つのコンテナにまとめ、表示時は高さを確保・非表示時は高さ0で重なりを防ぐ
        self.progress_container = QWidget()
        progress_container_layout = QVBoxLayout(self.progress_container)
        progress_container_layout.setContentsMargins(0, 4, 0, 0)
        progress_container_layout.setSpacing(2)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(22)
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #333333; font-size: 12px;")
        self.progress_label.setMinimumHeight(18)
        progress_container_layout.addWidget(self.progress_bar)
        progress_container_layout.addWidget(self.progress_label)
        self.progress_container.setMaximumHeight(0)
        self.progress_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        step3_layout.addWidget(self.progress_container)
        
        main_layout.addWidget(step3_group)

        # --- Console Logs ---
        log_label = QLabel("処理ログ:")
        log_label.setStyleSheet("font-weight: bold; color: #333333;")
        main_layout.addWidget(log_label)
        main_layout.addWidget(self.log_viewer)

        main_layout.addStretch()

        self.worker = None

    def setup_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
                color: #333333;
            }
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                margin-top: 1.2em;
                padding-top: 10px;
                background-color: white;
                color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                left: 12px;
                color: #0078D7;
            }
            QPushButton {
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
                border: 1px solid #ced4da;
                background-color: #ffffff;
                color: #333333;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
            QPushButton#primaryButton {
                background-color: #0078D7;
                color: white;
                border: none;
                font-size: 16px;
            }
            QPushButton#primaryButton:hover {
                background-color: #005a9e;
            }
            QPushButton#primaryButton:disabled {
                background-color: #ccc;
                color: #666666;
            }
            QPushButton#secondaryButton {
                color: #495057;
            }
            QPushButton#dangerButton {
                background-color: #fff;
                color: #dc3545;
                border-color: #dc3545;
            }
            QPushButton#dangerButton:hover {
                background-color: #dc3545;
                color: white;
            }
            QLineEdit {
                padding: 6px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                color: #333333;
                background-color: #ffffff;
            }
            QTableWidget {
                border: 1px solid #dee2e6;
                gridline-color: #f1f3f5;
                background-color: white;
                color: #333333;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                color: #333333;
                padding: 4px;
                border: 1px solid #dee2e6;
                font-weight: bold;
            }
            QProgressBar {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                text-align: center;
                background-color: #f8f9fa;
                color: #333333;
                height: 15px;
            }
            QProgressBar::chunk {
                background-color: #28a745;
                border-radius: 3px;
            }
            QRadioButton {
                spacing: 8px;
                color: #333333;
            }
            QLabel {
                color: #333333;
            }
            QMessageBox {
                background-color: #ffffff;
                color: #333333;
            }
            QMessageBox QLabel {
                color: #333333;
            }
        """)

    def predict_event_name(self, filename: str) -> str:
        """
            ファイル名から大会名を推測する
            大会名は大文字略称＋年度＋(Men or Women)
        """
        text = filename.split('_')[0].upper()
        if "women" in filename.lower():
            text += "Women"
        elif "men" in filename.lower():
            text += "Men"
        return text

    def update_file_paths(self, paths: list) -> None:
        """
            ドロップエリアから複数パスを受け取り、テーブルに追加
        """
        self.file_table.blockSignals(True)  # cellChanged シグナルを一時停止
        added = 0
        for path_str in paths:
            path = Path(path_str)
            # 重複チェック
            if any(entry["path"] == path for entry in self.file_entries):
                continue
            
            event_name = self.predict_event_name(path.name)
            self.file_entries.append({"path": path, "event_name": event_name})
            
            row = self.file_table.rowCount()
            self.file_table.insertRow(row)
            
            # ファイル名（読み取り専用）
            name_item = QTableWidgetItem(path.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.file_table.setItem(row, 0, name_item)
            
            # Event Name（読み取り専用）
            event_item = QTableWidgetItem(event_name)
            event_item.setFlags(event_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.file_table.setItem(row, 1, event_item)
            added += 1
        
        self.file_table.blockSignals(False)
        logger.info(f"{added}個のファイルを追加 (合計: {len(self.file_entries)}個)")

    def on_table_cell_changed(self, row, column) -> None:
        """テーブルのEvent Name列が編集された時の処理"""
        if column == 1 and row < len(self.file_entries):
            new_name = self.file_table.item(row, 1).text().strip()
            self.file_entries[row]["event_name"] = new_name

    def clear_all_files(self) -> None:
        """全ての選択済みファイルをクリア"""
        self.file_table.setRowCount(0)
        self.file_entries.clear()
        self.drop_area.clear()
        self.log_write("ファイル一覧をクリアしました。")

    def delete_selected_files(self) -> None:
        """選択されたファイルをテーブルから削除"""
        selected_rows = sorted(
            set(index.row() for index in self.file_table.selectedIndexes()), 
            reverse=True
        )
        if not selected_rows:
            return
        
        for row in selected_rows:
            if row < len(self.file_entries):
                name = self.file_entries[row]["path"].name
                self.file_table.removeRow(row)
                self.file_entries.pop(row)
                self.log_write(f"削除しました: {name}")
        
        if not self.file_entries:
            self.drop_area.clear()
        
        logger.info(f"残りファイル数: {len(self.file_entries)}")

    def start_analysis(self) -> None:
        """
            解析開始ボタンが押されたときの処理
        """
        # 入力チェック
        if not self.file_entries:
            QMessageBox.warning(self, "入力エラー", "PDFファイルを選択してください。")
            return
        
        # Event Name の空チェックとファイル存在チェック
        for entry in self.file_entries:
            if not entry["event_name"]:
                QMessageBox.warning(self, "入力エラー", 
                    f"{entry['path'].name} のEvent Nameを特定できませんでした。")
                return
            if not entry["path"].exists():
                QMessageBox.warning(self, "入力エラー", 
                    f"{entry['path'].name} が見つかりません。")
                return
        
        # ウィジェットからパス情報を取得
        db_path, is_new = self.db_selector.get_active_db_path()

        if not db_path:
            QMessageBox.warning(self, "エラー", "データベースの設定が完了していません。\nパスまたはファイル名を入力してください。")
            return

        if is_new:
            # 既に同名ファイルがあるかチェックする
            if db_path.exists():
                ret = QMessageBox.question(self, "上書き確認", 
                    f"ファイルが既に存在します。\n追記しますか？\n{str(db_path)}",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if ret == QMessageBox.StandardButton.No:
                    return
                else:
                    logger.info(f"既存接続モード: {db_path}")
            else:
                logger.info(f"新規作成モード: {db_path}")
                set_tables(str(db_path), is_md=self.is_md)
        else:
            logger.info(f"既存接続モード: {db_path}")

        # -------------------------------------------------------
        # ここでバックグラウンド処理(QThread)を開始
        # -------------------------------------------------------
        file_list_text = "\n".join(
            [f"  {e['path'].name} → {e['event_name']}" for e in self.file_entries]
        )
        ret_start = QMessageBox.question(self, "確認", 
            f"以下の情報で解析を開始しますか？\n\n"
            f"ファイル ({len(self.file_entries)}件):\n{file_list_text}\n\n"
            f"データベース: {db_path.name}\n"
            f"形式: {'MD' if self.md_btn_group.checkedId() == 1 else '4人制'}",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        if ret_start == QMessageBox.StandardButton.Cancel:
            return

        # 2. UIを「処理中モード」にする
        self.run_button.setEnabled(False) # 二重押し防止
        self.progress_bar.setValue(0)

        # 3. Workerスレッドを作成
        self.worker = Worker(self.file_entries, db_path, self.is_md)

        # 4. シグナル（通信）を接続
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.analysis_finished)
        self.worker.error_signal.connect(self.analysis_error)
        self.worker.visible_signal.connect(self.progress_bar_set_visible)

        # 5. スレッド開始
        self.worker.start()
    
    def __set_radio_button(self, txt1, txt2, default=0) -> tuple[QHBoxLayout, QButtonGroup]:
        mode_layout = QHBoxLayout()
        radio1 = QRadioButton(txt1)
        radio2 = QRadioButton(txt2)
        radio1.setCursor(Qt.CursorShape.PointingHandCursor)
        radio2.setCursor(Qt.CursorShape.PointingHandCursor)

        if default == 0:
            radio1.setChecked(True)
        else:
            radio2.setChecked(True)
        
        btn_group = QButtonGroup(self)
        btn_group.addButton(radio1, 0)
        btn_group.addButton(radio2, 1)

        mode_layout.addWidget(radio1)
        mode_layout.addWidget(radio2)

        return mode_layout, btn_group
    
    def md_clicked(self, button_id) -> None:
        """
            ラジオボタンが切り替わった時に呼ばれる処理
        """
        if button_id == 0:
            self.is_md = False    
        elif button_id == 1:
            self.is_md = True

    def log_write(self, msg: str, color: QColor = None) -> None:
        """ログビューアにメッセージを書き込む"""
        now = datetime.datetime.now().strftime("%H:%M:%S")
        
        self.log_viewer.moveCursor(QTextCursor.MoveOperation.End)
        
        # 色の指定がない場合はデフォルト（またはログレベルに応じた色）
        if not color:
            color = QColor("#E0E0E0") # デフォルトの薄いグレー

        fmt = self.log_viewer.currentCharFormat()
        fmt.setForeground(color)
        self.log_viewer.setCurrentCharFormat(fmt)
        
        self.log_viewer.insertPlainText(f"[{now}] {msg}\n")
        self.log_viewer.moveCursor(QTextCursor.MoveOperation.End)

    def _handle_log_signal(self, msg: str, level: int) -> None:
        """ロギングシステムから送られてきたメッセージを処理する"""
        color = None
        if level >= logging.ERROR:
            color = QColor("#FF5252") # Red
        elif level >= logging.WARNING:
            color = QColor("#FFD740") # Amber
        elif level >= logging.INFO:
            # INFOログは通常の色
            color = QColor("#E0E0E0")
            
        self.log_write(msg, color)

    # --- 以下、スレッドから呼ばれる関数 ---
    def update_progress(self, val, msg) -> None:
        """進捗バーとその直下のラベルを更新（短い文言はログに出さず、logger 経由の詳細だけログに表示）"""
        self.progress_bar.setValue(val)
        self.progress_label.setText(msg)

    def analysis_finished(self, msg) -> None:
        """完了時の処理"""
        self.run_button.setEnabled(True)
        self.log_write(f"SUCCESS: {msg}", QColor("#2ecc71"))
        QMessageBox.information(self, "Complete", msg)
        self.progress_bar.setValue(100)
        self.worker = None
        self.file_entries.clear()
        self.file_table.setRowCount(0)
        self.drop_area.clear()

    def analysis_error(self, err_msg) -> None:
        """エラー時の処理"""
        self.run_button.setEnabled(True)
        self.log_write(f"ERROR: {err_msg}", QColor("#e74c3c"))
        QMessageBox.critical(self, "Error", err_msg)
        self.worker = None

    def progress_bar_set_visible(self, visible: bool) -> None:
        """プログレスバーと進捗メッセージの表示/非表示切替（コンテナの高さでレイアウトを安定させる）"""
        if visible:
            self.progress_container.setMaximumHeight(60)
            self.progress_bar.setValue(0)
        else:
            self.progress_container.setMaximumHeight(0)
            self.progress_label.setText("")

# アプリケーション起動
if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())        