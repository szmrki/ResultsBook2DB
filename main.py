import os
import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QFileDialog, QMessageBox, QFormLayout, QGroupBox, 
                             QRadioButton, QButtonGroup, QProgressBar)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QMouseEvent
from pathlib import Path
from create_db import set_tables
from worker import Worker
import multiprocessing

# ---------------------------------------------------------
# ドラッグ＆ドロップ専用のカスタムラベルを作成
# ---------------------------------------------------------
class FileDropLabel(QLabel):
    # ファイルがドロップされたことをメイン画面に知らせるシグナル
    fileDropped = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setText("\nここにPDFファイルをドラッグ&ドロップ\nまたはクリックして選択\n")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # マウスを乗せた時に「指マーク」にする
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.init_label = """
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f9f9f9;
                color: #555;
                font-size: 14px;
            }
            QLabel:hover {
                background-color: #e0e0e0;
                border-color: #0078D7;
            }
        """
        
        # デザイン設定（点線の枠線、背景色など）
        self.setStyleSheet(self.init_label)
        
        # ドロップを受け付ける設定
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """ドラッグされたものがエリアに入った時の処理"""
        if event.mimeData().hasUrls():
            # ファイルなら受け入れる
            event.accept()
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #0078D7;
                    border-radius: 10px;
                    background-color: #eaf4ff;
                    font-size: 14px;
                }
            """)
        else:
            event.ignore()

    def dragLeaveEvent(self) -> None:
        """ドラッグがエリアから出た時の処理（デザインを戻す）"""
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f9f9f9;
                font-size: 14px;      
            }
        """)

    def dropEvent(self, event: QDropEvent) -> None:
        """ドロップされた時の処理"""
        urls = event.mimeData().urls()
        if urls:
            # 最初のファイルのパスを取得
            file_path = urls[0].toLocalFile()
            
            # PDFかどうかの簡易チェック
            if file_path.lower().endswith('.pdf'):
                self.setText(f"\nSelected File:\n{os.path.basename(file_path)}\n")
                # 親ウィンドウにパスを通知
                self.fileDropped.emit(file_path)
            else:
                self.setText("\nError:\nPDFファイルのみ対応しています\n")
                # デザインを戻す
                self.setStyleSheet("""
                QLabel {
                    border: 2px dashed red;
                    border-radius: 10px;
                    background-color: #ffeeee;
                    font-size: 14px;
                }
                """)
                

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """クリックされた時の処理"""
        super().mousePressEvent(event)
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "PDFファイルを選択",
            str(Path.cwd()), # カレントディレクトリから開始
            "PDF Files (*.pdf)"
        )
        if file_name:
            self.process_file(file_name)

    # ドロップとクリックの共通処理
    def process_file(self, file_path) -> None:
        self.setText(f"\nSelected File:\n{os.path.basename(file_path)}\n")
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
        self.fileDropped.emit(file_path)
        print(file_path)
    
    def clear(self) -> None:
        """ラベルを初期状態に戻す"""
        self.setText("\nここにPDFファイルをドラッグ&ドロップ\nまたはクリックして選択\n")
        self.setStyleSheet(self.init_label)

#データベース
class DatabaseSelector(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_folder = str(Path.cwd()) # デフォルトはカレントディレクトリ

        # 全体をグループボックスで囲む
        main_layout = QVBoxLayout(self)
        group_box = QGroupBox("Database Config")
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
        self.btn_browse_folder.clicked.connect(self.select_folder)
        
        folder_layout.addWidget(self.folder_input)
        folder_layout.addWidget(self.btn_browse_folder)
        new_layout.addRow("保存先:", folder_layout)

        # B. ファイル名入力
        filename_layout = QHBoxLayout()
        self.filename_input = QLineEdit()
        
        # 拡張子ラベル (.db)
        ext_label = QLabel(".db")
        ext_label.setStyleSheet("font-weight: bold; color: #555;")
        
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

# メインウィンドウ
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ResultsBook2DB")
        self.setGeometry(100, 100, 500, 200)
        self.setFixedWidth(500)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 全体のレイアウト (縦並び)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.full_path: Path = None

        # -----------------
        # 入力エリア
        # -----------------
        # DB選択エリアを追加
        self.db_selector = DatabaseSelector()
        layout.addWidget(self.db_selector)

        # カスタムドロップエリアの配置
        self.drop_area = FileDropLabel()
        # シグナルを受け取って変数を更新する関数につなぐ
        self.drop_area.fileDropped.connect(self.update_file_path)
        layout.addWidget(self.drop_area)

        self.path_display = QLineEdit()
        #self.path_display.setPlaceholderText("PDFファイル名がここに表示されます")
        #self.path_display.setReadOnly(True)
        #layout.addWidget(self.path_display) 必要性が少ないため削除

        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight) # ラベルを右寄せ

        # 大会名入力
        self.tournament_input = QLineEdit()
        self.tournament_input.setPlaceholderText("e.g.: WWCC2025")
        self.path_display.textChanged.connect(self.set_pred_text)
        form_layout.addRow("Event Name ", self.tournament_input)
        
        # メインレイアウトにフォームを追加
        layout.addLayout(form_layout)
        layout.addSpacing(10)

        #MDかどうかのラジオボタン
        md_layout, self.md_btn_group = self.__set_radio_button("4人制", "MD", default=0)
        self.is_md = False #デフォルトは4人制
        layout.addLayout(md_layout)
        self.md_btn_group.idClicked.connect(self.md_clicked)

        # ---------------------------------------------------------
        # 実行アクションエリア
        # ---------------------------------------------------------
        # 少し余白を空ける
        layout.addSpacing(20)

        self.run_button = QPushButton("Start")
        self.run_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_button.setHeight = 50
        # ボタンを目立たせるスタイルシート
        self.run_button.setStyleSheet("""     
                QPushButton {
                background-color: #0078D7;
                color: white;
                font-weight: bold;
                padding: 6px;
                border-radius: 5px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #005A9E;
            }
        """)
        self.run_button.clicked.connect(self.start_analysis)
        layout.addWidget(self.run_button)

        # プログレスバーを追加、最初は隠しておく
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar) 
        
        self.worker = None # スレッド保持用変数

        # 下部にスペースを埋めるための伸縮アイテム
        layout.addStretch()

    def update_file_path(self, path: str) -> None:
        """
            ドロップエリアからパスを受け取る
        """
        self.full_path = Path(path)
        path = self.full_path.name
        self.path_display.setText(path)

    def start_analysis(self) -> None:
        """
            解析開始ボタンが押されたときの処理
        """
        tournament_name = self.tournament_input.text().strip()

        # 入力チェック
        if not tournament_name:
            QMessageBox.warning(self, "入力エラー", "大会名を入力してください。")
            return
        if not self.full_path or not self.full_path.exists():
            QMessageBox.warning(self, "入力エラー", "有効なPDFファイルを選択してください。")
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
                    print(f"既存接続モード: {db_path}")
            else:
                print(f"新規作成モード: {db_path}")
                set_tables(str(db_path))
        else:
            print(f"既存接続モード: {db_path}")

        # -------------------------------------------------------
        # ここでバックグラウンド処理(QThread)を開始
        # -------------------------------------------------------
        ret_start = QMessageBox.question(self, "確認", 
            f"以下の情報で解析を開始しますか？\n\n"
            f"大会名: {tournament_name}\n"
            f"ファイル: {self.full_path.name}\n"
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
        self.worker = Worker(self.full_path, tournament_name, db_path, self.is_md)

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

    def set_pred_text(self) -> None:
        """
            予測大会名を設定する
            大会名は大文字略称＋年度＋(Men or Woomen)
        """
        path = self.path_display.text()
        text = path.split('_')[0].upper()
        if "Men" in path:
            text += "Men"
        elif "Women" in path:
            text += "Women"
        self.tournament_input.setText(text)

    # --- 以下、スレッドから呼ばれる関数 ---
    def update_progress(self, val, msg) -> None:
        """
            進捗バーとメッセージを更新
        """
        self.progress_bar.setValue(val)
        self.statusBar().showMessage(msg) # ステータスバーがある場合
        # またはラベル等のテキストを更新

    def analysis_finished(self, msg) -> None:
        """
            完了時の処理
        """
        self.run_button.setEnabled(True) # ボタンを復活
        QMessageBox.information(self, "Complete", msg)
        self.progress_bar.setValue(100)
        self.worker = None # 後始末
        self.tournament_input.clear()
        self.path_display.clear()
        self.drop_area.clear()

    def analysis_error(self, err_msg) -> None:
        """
            エラー時の処理
        """
        self.run_button.setEnabled(True)
        QMessageBox.critical(self, "Error", err_msg)
        self.worker = None

    def progress_bar_set_visible(self, visible: bool) -> None:
        """
            プログレスバーの表示/非表示切替
        """
        self.progress_bar.setVisible(visible)
        # 非表示の時だけウィンドウを縮める
        if not visible:
            QTimer.singleShot(0, self.adjustSize)


# アプリケーション起動
if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())        