"""
worker.py: バックグラウンド実行・解析進行管理モジュール
GUI画面がフリーズしないよう、QThreadを利用して非同期で以下の処理を統括します。
1. PDFからのストーン画像・スコア表の抽出 (`pdf_tools`)
2. YOLO推論による座標変換 (`detection`)
3. ショット情報や大会情報との紐付けおよびDB保存 (`create_db`)
"""
import time
import traceback
from PySide6.QtCore import QThread, Signal
import sqlite3
from event_processing import event_exists, extract_event, write_event, postprocess_event
from stone_matching import ensure_shot_order_column
import shutil
import sys
import io
import logging
from typing import Any
from pathlib import Path

logger = logging.getLogger(__name__)

class Worker(QThread):
    # メインスレッド（画面）に情報を送るための「通信線」
    progress_signal = Signal(int, str)  # 進捗率(%), メッセージ
    finished_signal = Signal(str)       # 完了時のメッセージ
    error_signal = Signal(str)          # エラー発生時のメッセージ
    cancelled_signal = Signal()         # 中断時のシグナル
    file_index_signal = Signal(int)     # 現在処理中ファイルのインデックス（0始まり）
    visible_signal = Signal(bool)      # プログレスバーの表示/非表示

    def __init__(self, pdf_entries: list[dict[str, Any]], db_path: str | Path, is_md: bool = False) -> None:
        super().__init__()
        # pdf_entries: list of {"path": Path, "event_name": str}
        self.pdf_entries = pdf_entries
        self.db_path = str(db_path)
        self.is_md = is_md

    def run(self) -> None:
        """
            別スレッドで実行される処理 (複数PDF対応)
        """
        # --- 偽の出力先を作成 ---
        if sys.stdout is None:
            sys.stdout = io.StringIO()
        if sys.stderr is None:
            sys.stderr = io.StringIO()
        # ------------------
        try:
            # --- 処理開始の通知 ---
            self.visible_signal.emit(True)
            self.progress_signal.emit(0, "Loading...")

            # セッション開始時に runs/detect 内の predict フォルダのみクリア（大会名フォルダは維持）
            self._cleanup_predict_dirs()

            # 処理本体
            self.conn = sqlite3.connect(self.db_path)

            # --- 外部キー制約をONにする ---
            self.conn.execute("PRAGMA foreign_keys = ON;")
            
            cur_init = self.conn.cursor()
            cur_init.execute('''CREATE TABLE IF NOT EXISTS lsds (
                id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER NOT NULL,
                team STRING, player_name STRING, distance_cm FLOAT,
                FOREIGN KEY(game_id) REFERENCES games(id) ON DELETE CASCADE ON UPDATE CASCADE)'''
            )
            # 既存DB向け: stones.shot_order カラムが無ければ追加
            ensure_shot_order_column(self.conn)
            self.conn.commit()
            
            start_time_all = time.time()
            errors = []
            
            for i, entry in enumerate(self.pdf_entries, start=1):
                # 中断チェック（ファイルごとのループ先頭）
                if self.isInterruptionRequested():
                    break

                # 現在処理中インデックスをUIに通知
                self.file_index_signal.emit(i - 1)  # 0始まりに変換

                # 解析中にファイルが追加された場合も分母を現在の総数に合わせる ([2/2] のように表示)
                total = max(len(self.pdf_entries), i)
                pdf_path = str(entry["path"])
                tournament_name = entry["event_name"]
                prefix = f"[{i}/{total}] "
                
                try:
                    success = self.executemodel(pdf_path, tournament_name, prefix)
                    if not success:
                        self.conn.rollback()  # 大会名重複した際はFalseが返された上で、ここでロールバック
                        err_msg = f"{entry['path'].name}: Event Name '{tournament_name}' は既に使用されています"
                        errors.append(err_msg)
                        logger.error(err_msg)
                except Exception as e:
                    self.conn.rollback()  # その他のエラーが起きた際は一貫してここでロールバック
                    error_msg = traceback.format_exc()
                    errors.append(f"{entry['path'].name}: {e}")
                    logger.error(error_msg)
            
            # 中断された場合 (処理中ファイルの未コミット分をロールバック)
            if self.isInterruptionRequested():
                self.conn.rollback()
                self.conn.close()
                self.cancelled_signal.emit()
                self.visible_signal.emit(False)
                self.progress_signal.emit(0, "")
                return
            # 処理完了後のDBのレコード数を取得し、結果をログに出力
            after_stats = self._get_db_stats()
            self._print_db_summary(after_stats)

            self.conn.close()

            # predict フォルダのクリーンアップ
            self._cleanup_predict_dirs()
            elapsed_all = time.time() - start_time_all
            logger.info(f"All processes completed in {elapsed_all:.2f}s")
            
            if errors:
                error_text = "\n".join(errors)
                if len(errors) == total:
                    self.error_signal.emit(f"全てのファイルでエラーが発生しました:\n{error_text}")
                else:
                    self.finished_signal.emit(
                        f"処理完了 ({total - len(errors)}/{total} 成功)\n\nエラー:\n{error_text}")
            else:
                self.progress_signal.emit(100, "Complete")
                time.sleep(1)
                self.finished_signal.emit(f"全{total}ファイルの保存が完了しました。")

        except Exception as e:
            # エラーが起きたら詳細を画面に送る
            error_msg = traceback.format_exc()
            self.error_signal.emit(f"An error has occurred.\n{e}\n{error_msg}")
            logger.error(error_msg)

        self.visible_signal.emit(False)
        self.progress_signal.emit(0, "")

    def _cleanup_predict_dirs(self) -> None:
        """runs/detect 内の predict フォルダを削除する"""
        runs_dir = Path("runs/detect")
        if not runs_dir.exists():
            return
        for d in runs_dir.iterdir():
            if d.is_dir() and d.name.startswith("predict"):
                try:
                    shutil.rmtree(d)
                    logger.debug(f"Removed predict directory: {d}")
                except Exception as e:
                    logger.warning(f"Failed to remove {d}: {e}")

    def _get_db_stats(self) -> dict:
        """データベースの主要なテーブルのレコード数を取得する"""
        stats = {"events": 0, "games": 0, "ends": 0, "shots": 0, "stones": 0}
        try:
            cur = self.conn.cursor()
            for key in stats.keys():
                cur.execute(f"SELECT COUNT(*) FROM {key}")
                stats[key] = cur.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to get DB stats: {e}")
        return stats

    def _print_db_summary(self, stats: dict) -> None:
        """データベースの現在の総件数をログに出力する"""
        summary_msg = (
            "\n=========================================\n"
            "【現在のデータベース統計情報】\n"
            f" ・ 登録大会数:   {stats['events']:,} 大会\n"
            f" ・ 総試合数:     {stats['games']:,} 試合\n"
            f" ・ 総エンド数:   {stats['ends']:,} エンド\n"
            f" ・ 総ショット数: {stats['shots']:,} ショット\n"
            f" ・ 総ストーン数: {stats['stones']:,} 件\n"
            "========================================="
        )
        logger.info(summary_msg)

    def executemodel(self, pdf_path: str, tournament_name: str, prefix: str = "") -> bool:
        """
            指定されたPDFを解析し、DBに情報を格納する。
            処理の本体は event_processing の3つの関数 ( 読み取り・書き込み・後処理 ) で、
            ここではそれらを順に呼び、進捗と中断を GUI につなぐ。

            Args:
                pdf_path (str): PDFファイルのパス
                tournament_name (str): 大会名
                prefix (str): 進捗メッセージの接頭辞 (例: "[1/3] ")

            Returns:
                bool : 処理が成功したらTrue、失敗したらFalse
        """
        game = tournament_name

        # events.name は UNIQUE のため、同名の大会は取り込めない。時間のかかる解析の前に確認する
        if event_exists(self.conn, game):
            logger.warning(f"Duplicate event name found in database: {game}")
            return False

        # 進捗は接頭辞 ( "[1/3] " など ) を付けて画面に送る
        progress_cb = lambda percent, message: self.progress_signal.emit(percent, f"{prefix}{message}")
        should_stop = self.isInterruptionRequested

        # 1. PDF → 解析結果 ( DB には触れない )
        result = extract_event(pdf_path, game, self.is_md, progress_cb, should_stop)
        # 検出途中で中断された場合は何も書き込まず、run側のロールバックに委ねる (大会ごと破棄)
        if result is None:
            return True

        # 2. 解析結果 → SQLite
        event_id, prepositioned_map = write_event(self.conn, result)

        # 3. DB 上の後処理 ( MD のハンマー補正 + ストーン同定 )
        start_time_match = time.time()
        updated = postprocess_event(self.conn, event_id, self.is_md, prepositioned_map, progress_cb, should_stop)
        # 同定中に中断された場合も、検出結果ごとコミットせず run側のロールバックに委ねる。
        # 検出+同定が揃って初めて1大会としてコミットすることで、shot_order=NULL の
        # 中途半端な状態を残さず「大会ごと破棄」に一本化する。
        if self.isInterruptionRequested():
            return True
        logger.info(f"[{game}] Stone matching complete: {updated} stones labeled "
                    f"(took {time.time() - start_time_match:.2f}s).")
        self.conn.commit()  # 検出結果と同定結果をまとめてコミット
        return True
