import time
import traceback
from PySide6.QtCore import QThread, Signal
import sqlite3
from pdf_convert import executemodel

class Worker(QThread):
    # メインスレッド（画面）に情報を送るための「通信線」
    progress_signal = Signal(int, str)  # 進捗率(%), メッセージ
    finished_signal = Signal(str)       # 完了時のメッセージ
    error_signal = Signal(str)          # エラー発生時のメッセージ

    def __init__(self, pdf_path, tournament_name, db_path):
        super().__init__()
        self.pdf_path = pdf_path
        self.tournament_name = tournament_name
        self.db_path = db_path

    def run(self):
        """
        ここが別スレッドで実行されます。
        重い処理（YOLO, DB接続）はすべてここに書きます。
        """
        try:
            # --- 処理開始の通知 ---
            self.progress_signal.emit(0, "PDFを読み込んでいます...")
            
            # 【ここにご自身の既存ロジックを組み込みます】
            # 例: pdf_images = convert_from_path(self.pdf_path)
            conn = sqlite3.connect(self.db_path)
            self.progress_signal.emit(50, "解析中...")
            executemodel(self.tournament_name, self.pdf_path, conn)
            
            """
            # ↓↓↓ ダミー処理（実装時はここを消してYOLOコードに置き換え） ↓↓↓
            total_steps = 5
            for i in range(total_steps):
                time.sleep(1) # 重い処理のシミュレーション
                
                # 進捗を通知 (例: 20%, 40%...)
                progress = int((i + 1) / total_steps * 100)
                self.progress_signal.emit(progress, f"ページ {i+1} を解析中...")
            self.progress_signal.emit(100, "解析が完了しました。DBに保存しています...")
            # ↑↑↑ ダミー処理 ここまで ↑↑↑
            """

            # 処理完了
            self.finished_signal.emit("すべての解析とDB保存が完了しました！")

        except Exception as e:
            # エラーが起きたら詳細を画面に送る
            error_msg = traceback.format_exc()
            self.error_signal.emit(f"エラーが発生しました:\n{e}\n{error_msg}")