import os
import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QFileDialog, QMessageBox, QFormLayout)
from PySide6.QtCore import Qt, Signal
from pathlib import Path

# ---------------------------------------------------------
# ドラッグ＆ドロップ専用のカスタムラベルを作成
# ---------------------------------------------------------
class FileDropLabel(QLabel):
    # ファイルがドロップされたことをメイン画面に知らせるシグナル
    fileDropped = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setText("\nここにPDFファイルをドラッグ＆ドロップ\nまたはクリックして選択\n")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # マウスを乗せた時に「指マーク」にする（クリックできる感が出る）
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # デザイン設定（点線の枠線、背景色など）
        self.setStyleSheet("""
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
        """)
        
        # ★ここが重要：ドロップを受け付ける設定
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:
        """ドラッグされたものがエリアに入った時の処理"""
        if event.mimeData().hasUrls():
            # ファイルなら受け入れる
            event.accept()
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #0078D7;
                    background-color: #eaf4ff;
                }
            """)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        """ドラッグがエリアから出た時の処理（デザインを戻す）"""
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f9f9f9;
            }
        """)

    def dropEvent(self, event) -> None:
        """ドロップされた時の処理"""
        urls = event.mimeData().urls()
        if urls:
            # 最初のファイルのパスを取得
            file_path = urls[0].toLocalFile()
            
            # PDFかどうかの簡易チェック
            if file_path.lower().endswith('.pdf'):
                self.setText(f"選択されたファイル:\n{os.path.basename(file_path)}")
                # 親ウィンドウにパスを通知
                self.fileDropped.emit(file_path)
            else:
                self.setText("エラー: PDFファイルのみ対応しています")
                # デザインを戻す
                self.setStyleSheet("""
                QLabel {
                    border: 2px dashed red;
                    background-color: #ffeeee;
                }
                """)

    def mousePressEvent(self, event) -> None:
        """クリックされた時（必要ならここにファイル選択ダイアログ処理を書く）"""
        # 今回はクリック処理は親側で実装するか、ここで実装するか選べますが
        # シンプルに「クリックでも反応する」UIにする場合に使います
        super().mousePressEvent(event)
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "PDFファイルを選択",
            str(Path.cwd()), # カレントディレクトリから開始
            "PDF Files (*.pdf)"
        )
        if file_name:
            self.process_file(file_name)

    # ★追加: ドロップとクリックの共通処理
    def process_file(self, file_path) -> None:
        import os
        self.setText(f"選択完了:\n{os.path.basename(file_path)}")
        # メインウィンドウに通知
        self.fileDropped.emit(file_path)
        print(file_path)

# メインウィンドウ
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ResultsBook2DB")
        self.setGeometry(100, 100, 500, 200)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        #layout = QVBoxLayout(central_widget)

        # 全体のレイアウト (縦並び)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.full_path = None

        # ---------------------------------------------------------
        # 入力フォームエリア (大会名 & PDF選択)
        # ---------------------------------------------------------
        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight) # ラベルを右寄せ

        # A. 大会名入力
        self.tournament_input = QLineEdit()
        self.tournament_input.setPlaceholderText("e.g.: WWCC2025")
        form_layout.addRow("Event Name:", self.tournament_input)
        
        # メインレイアウトにフォームを追加
        layout.addLayout(form_layout)
        layout.addSpacing(20)

        # カスタムドロップエリアの配置
        self.drop_area = FileDropLabel()
        # シグナルを受け取って変数を更新する関数につなぐ
        self.drop_area.fileDropped.connect(self.update_file_path)
        layout.addWidget(self.drop_area)

        # 現在選択されているファイルパスを表示する（確認用）
        self.path_display = QLineEdit()
        self.path_display.setPlaceholderText("ファイルパスがここに表示されます")
        self.path_display.setReadOnly(True)
        layout.addWidget(self.path_display)

        # ---------------------------------------------------------
        # 実行アクションエリア
        # ---------------------------------------------------------
        # 少し余白を空ける
        layout.addSpacing(20)

        self.run_button = QPushButton("Start")
        self.run_button.setHeight = 50
        # ボタンを目立たせるスタイルシート（任意）
        self.run_button.setStyleSheet("""     
                QPushButton {
                background-color: #0078D7;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005A9E;
            }
        """)
        self.run_button.clicked.connect(self.start_analysis)
        layout.addWidget(self.run_button)

        # 下部にスペースを埋めるための伸縮アイテム
        layout.addStretch()

    def update_file_path(self, path) -> None:
            """ドロップエリアからパスを受け取る"""
            self.full_path = path
            path = os.path.basename(path)
            self.path_display.setText(path)

    def start_analysis(self) -> None:
        """解析開始ボタンが押されたときの処理"""
        tournament_name = self.tournament_input.text().strip()

        # 入力チェック
        if not tournament_name:
            QMessageBox.warning(self, "入力エラー", "大会名を入力してください。")
            return
        if not self.full_path or not os.path.exists(self.full_path):
            QMessageBox.warning(self, "入力エラー", "有効なPDFファイルを選択してください。")
            return

        # -------------------------------------------------------
        # ここでバックグラウンド処理(QThread)を開始します
        # -------------------------------------------------------
        QMessageBox.information(self, "確認", 
            f"以下の情報で解析を開始します。\n\n"
            f"大会名: {tournament_name}\n"
            f"ファイル: {os.path.basename(self.full_path)}\n\n"
            f"(※ここにYOLOの処理スレッドを接続します)"
        )

# アプリケーション起動
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())        