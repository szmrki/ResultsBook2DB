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
from pdf_tools import *
from yolo_tools import *
from utils import *
from detection import *
from stone_matching import ensure_shot_order_column, label_event_ends, correct_equidistant_blank_hammer
import sys
from itertools import zip_longest
import io
import logging
from typing import Any
from pathlib import Path

logger = logging.getLogger(__name__)

resource_path = lambda p: Path(getattr(
    sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__))
    )) / p

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
            指定された大会フォルダ内のPDFを解析し、DBに情報を格納する。
            解析にはYOLOモデルを使用し、必要に応じてファインチューニングも行う。
            解析結果はSQLiteデータベースに保存される。

            Args:
                pdf_path (str): PDFファイルのパス
                tournament_name (str): 大会名
                prefix (str): 進捗メッセージの接頭辞 (例: "[1/3] ")

            Returns:
                bool : 処理が成功したらTrue、失敗したらFalse
        """
        game = tournament_name
        num2color = {0: "red", 1: "yellow"}

        cur = self.conn.cursor()
        try:
            #eventテーブルに大会名、年、カテゴリを記述
            year, category = extract_year_and_category(game, self.is_md)
            cur.execute('INSERT INTO events(name, year, category) VALUES (?, ?, ?)', (game, year, category))
        except sqlite3.IntegrityError:
            logger.warning(f"Duplicate event name found in database: {game}")
            return False

        event_id = cur.lastrowid #event_idを取得

        doc = fitz.open(pdf_path)
        logger.info(f"Processing PDF: {pdf_path}")

        model = self._prepare_model(game, doc, prefix)

        # MD版: Prepositioned stoneの座標を end_id → list[dict] のマップで蓄積する
        prepositioned_map: dict[int, list[dict[str, Any]] | None] = {}
        start_time_det = time.time()
        with pdfplumber.open(pdf_path) as pdf:
            for pn in range(doc.page_count):
                # 中断チェック (ページごとのループ先頭)
                if self.isInterruptionRequested():
                    break
                self.progress_signal.emit(int(pn/doc.page_count*100), f"{prefix}Extracting data...")
                page_num = pn + 1
                page_plumber = pdf.pages[pn]
                page_mu = doc[pn]
                text = page_mu.get_text()
                if "Game Results" in text: #新たな試合
                    if self.is_md:
                        scores, power_play_ends = extract_game_result(page_plumber, self.is_md) #得点表のdfとPPエンドのリスト
                    else:
                        scores = extract_game_result(page_plumber) #得点表のdf
                    
                    hammers = get_hammer(scores, self.is_md)  #各エンドのハンマー情報
                    team_red = scores.at[0, "team"]
                    team_yellow = scores.at[1, "team"]
                    game_context = f"{team_red} vs {team_yellow}"
                    logger.debug(f"Scores:\n{scores}")
                    logger.debug(f"Hammers: {hammers}")
                    logger.info(f"[{game_context}] - Game Results page: {page_num}")
                    try:
                        fin_red = int(scores.at[0, "Total"]) #得点表のdfから最終得点を記録
                        fin_yellow = int(scores.at[1, "Total"])
                    except (ValueError, TypeError):
                        logger.warning(f"[{game_context}] Could not parse final scores from page {page_num}")
                        fin_red = None
                        fin_yellow = None
                    
                    # DB には3文字コードのみを保存する ( 国名部分の表記揺れを防ぐ )。
                    # team_red / team_yellow 変数自体は LSFE 照合や LSD 紐付けで
                    # フルネームのまま使うため、INSERT に渡す値だけ変換する。
                    team_red_code = to_team_code(team_red)
                    team_yellow_code = to_team_code(team_yellow)
                    cur.execute("""INSERT INTO games(event_id, page, team_red, team_yellow,
                                    final_score_red, final_score_yellow) VALUES (?, ?, ?, ?, ?, ?)""",
                                    (event_id, page_num, team_red_code, team_yellow_code, fin_red, fin_yellow))
                    game_id = cur.lastrowid #game_idを取得

                    # --- LSDデータを抽出・保存 ---
                    plumber_text = page_plumber.extract_text()
                    if plumber_text:
                        lsd_results = extract_lsd_from_text(plumber_text)
                        for lsd in lsd_results:
                            if lsd["player_red"] and lsd["lsd_red"] is not None:
                                cur.execute("INSERT INTO lsds (game_id, team, player_name, distance_cm) VALUES (?, ?, ?, ?)",
                                            (game_id, team_red_code, lsd["player_red"], lsd["lsd_red"]))
                            if lsd["player_yellow"] and lsd["lsd_yellow"] is not None:
                                cur.execute("INSERT INTO lsds (game_id, team, player_name, distance_cm) VALUES (?, ?, ?, ?)",
                                            (game_id, team_yellow_code, lsd["player_yellow"], lsd["lsd_yellow"]))

                    # ---------------------------
                    # ここでエンドテーブルに情報を一括挿入
                    ends_data = []
                    for i in range(len(hammers)):
                        if hammers[i] == None: break #コンシード済みのため
                        num_end_val = i + 1
                        str_end = str(num_end_val)
                        try:
                            score_red = int(scores.at[0, str_end]) #得点表のdfから得点を取得
                            score_yellow = int(scores.at[1, str_end])
                        except Exception:
                            score_red = None #存在しない場合はNULL
                            score_yellow = None
                        
                        try:
                            color_hammer = num2color[hammers[i]]
                        except Exception:
                            color_hammer = None
                        
                        # (game_id, page, number, color_hammer, score_red, score_yellow, [is_power_play])
                        # 初期段階では page は None
                        if self.is_md:
                            # Power Play情報の抽出ロジック
                            is_power_play = 1 if num_end_val in power_play_ends else 0
                            ends_data.append((game_id, None, num_end_val, color_hammer, score_red, score_yellow, is_power_play))
                        else:
                            ends_data.append((game_id, None, num_end_val, color_hammer, score_red, score_yellow))
                        
                    if self.is_md:
                        cur.executemany("""INSERT INTO ends(game_id, page, number, color_hammer, 
                                        score_red, score_yellow, is_power_play) VALUES (?, ?, ?, ?, ?, ?, ?)""", ends_data)
                    else:
                        cur.executemany("""INSERT INTO ends(game_id, page, number, color_hammer, 
                                        score_red, score_yellow) VALUES (?, ?, ?, ?, ?, ?)""", ends_data)
                    
                    num_end = 1

                elif "Shot by Shot" in text: #新たなエンド
                    # 該当するエンドのページ情報を更新し、end_idを取得
                    cur.execute("""UPDATE ends SET page = ? WHERE game_id = ? AND number = ?""", 
                                (page_num, game_id, num_end))
                    cur.execute("""SELECT id FROM ends WHERE game_id = ? AND number = ?""", 
                                (game_id, num_end))
                    end_id = cur.fetchone()[0]
                    
                    stones_end, shot_info, pre_stones_np = extract_shotbyshot(doc, page_mu, model, self.is_md)
                    logger.info(f"[{game_context}] End {num_end} - Shot-by-Shot page: {page_num} - Number of shots: {max(len(stones_end), len(shot_info))}")

                    # MD版: Prepositioned stone座標をマッチング用の辞書形式に変換して蓄積
                    if self.is_md:
                        if pre_stones_np is not None:
                            pre_stone_dicts: list[dict[str, Any]] = []
                            for row in pre_stones_np:
                                if row[5] == 1:  # insheet フラグが立っている行のみ
                                    pre_stone_dicts.append({
                                        'color': num2color[int(row[0])],
                                        'pos': (float(row[1]), float(row[2])),
                                        'label': 0,  # Prepositioned stone は shot_order=0
                                    })
                            # 有効なストーンが取れた場合のみマップに登録、取れなかった場合はNone
                            prepositioned_map[end_id] = pre_stone_dicts if pre_stone_dicts else None
                        else:
                            # MD版だがPrepositioned stone画像がなかった → スキップ対象
                            prepositioned_map[end_id] = None

                    for shot_num, (stones, info) in enumerate(zip_longest(stones_end, shot_info), start=1):
                        if info is not None: #正常時
                            shot_type = info["type"]; percent_score = info["score"]
                            turn = info["turn"]; team = info["team"]; player_name = info["player"]      
                        else: #ショット情報が取れない場合はNULLを挿入し、ストーン配置のみ保存する
                            shot_type = None; percent_score = None
                            turn = None; team = None; player_name = None
                            logger.warning(f"[{game_context}] End {num_end} - Shot {shot_num} - Shot info not found")

                        try:
                            shot_color = num2color[(hammers[num_end - 1] + (shot_num % 2)) % 2] #現在のショットの色を指定
                        except (TypeError, IndexError):
                            logger.warning(f"[{game_context}] End {num_end} - Shot {shot_num} - Shot color not found")
                            shot_color = None
                        cur.execute("""INSERT INTO shots(end_id, number, color, team, player_name, 
                                            type, turn, percent_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", 
                                            (end_id, shot_num, shot_color, team, player_name, 
                                            shot_type, turn, percent_score))    
                        shot_id = cur.lastrowid #shot_idを取得

                        if stones is not None: #正常時
                            rows = [(shot_id, num2color[int(row[0])], *row[1:]) for row in stones if row[5] == 1]
                        else: #ストーン情報が取れない場合
                            rows = []

                        if len(rows) == 0: #ストーンが存在しない場合はidのみ
                            rows = [(shot_id, None, None, None, None, None, None)]
                        #ストーンはまとめてinsert
                        cur.executemany("""INSERT INTO stones (shot_id, color, x, y, distance_from_center,  
                                        inhouse, insheet) VALUES (?, ?, ?, ?, ?, ?, ?)""", rows)
                    num_end += 1
                    #break
                else:
                    continue
        doc.close()
        # 検出途中で中断された場合はコミットせず、run側のロールバックに委ねる (大会ごと破棄)
        if self.isInterruptionRequested():
            return True
        elapsed_det = time.time() - start_time_det
        logger.info(f"[{game}] Detection complete (took {elapsed_det:.2f}s).")
        # MD版: 同距離ブランクエンド ( ハウス内に両チームの石が残った膠着ブランク ) では
        # 先攻後攻が交代しないため、get_hammer が入れた誤った交代を打ち消す。
        # 同定は color_hammer を色制約に使うため、補正は同定より前に行う。
        if self.is_md:
            corrected = correct_equidistant_blank_hammer(self.conn, event_id)
            if corrected:
                logger.info(f"[{game}] Equidistant-blank hammer correction: {corrected} ends updated.")
        # ストーン同定: この大会の全エンドについて各石の投球元(shot_order)を特定する
        self.progress_signal.emit(0, f"{prefix}Matching stones...")
        start_time_match = time.time()
        updated = label_event_ends(
            self.conn, event_id,
            progress_cb=lambda done, total: self.progress_signal.emit(
                int(done / total * 100) if total else 100,
                f"{prefix}Matching stones... ({done}/{total})"
            ),
            should_stop=self.isInterruptionRequested,  # 同定中も中止を受け付けて打ち切る
            prepositioned_map=prepositioned_map if self.is_md else None,
        )
        # 同定中に中断された場合も、検出結果ごとコミットせず run側のロールバックに委ねる。
        # 検出+同定が揃って初めて1大会としてコミットすることで、shot_order=NULL の
        # 中途半端な状態を残さず「大会ごと破棄」に一本化する。
        if self.isInterruptionRequested():
            return True
        logger.info(f"[{game}] Stone matching complete: {updated} stones labeled "
                    f"(took {time.time() - start_time_match:.2f}s).")
        self.conn.commit()  # 検出結果と同定結果をまとめてコミット
        return True

    def _prepare_model(self, game: str, doc, prefix: str) -> "YOLO":
        """ファインチューニング済みモデルが存在すればロード、なければ疑似ラベルでFTして保存する。"""
        work_dir = Path.cwd()
        model_dir = resource_path(Path("complete_model"))
        game_pt = model_dir / f"{game}.pt"

        if not game_pt.exists():
            start_time_ft = time.time()
            self.progress_signal.emit(0, f"{prefix}Preparing fine-tuning...")

            base_model_path = resource_path(model_dir / "base.pt")
            if not base_model_path.exists():
                raise FileNotFoundError(f"Base model not found at {base_model_path}. Please ensure 'complete_model/base.pt' exists.")

            model = YOLO(base_model_path)

            dataset_dir = work_dir / "yolo_dataset"
            image_dir = dataset_dir / "images"
            label_dir = dataset_dir / "labels"
            yaml_path = work_dir / "yaml" / "data.yaml"

            try:
                num_images = save_images(doc, output_dir=image_dir, save_num=400)
                num_labels = create_pseudo_label(model, image_dir=image_dir, output_dir=label_dir, threshold=0.75)
                logger.info(f"Dataset prepared: {num_labels} pseudo labels from {num_images} images.")
                split_train_val(image_dir, label_dir, train_ratio=0.8)
                create_yaml(yaml_path, dataset_dir)
            except Exception as e:
                logger.error(f"Failed to prepare dataset for fine-tuning: {e}")
                raise

            def on_train_epoch_end(trainer):
                curr = trainer.epoch + 1
                total = trainer.epochs
                self.progress_signal.emit(int(curr / total * 100), f"{prefix}Fine-tuning...")
                if self.isInterruptionRequested():
                    trainer.stop = True

            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            try:
                logger.info(f"Starting fine-tuning for event: {game}")
                results = model.train(
                    data=resource_path(yaml_path),
                    epochs=50,
                    imgsz=600,
                    iou=0.3,
                    conf=0.5,
                    save=True,
                    name=game,
                    exist_ok=False,
                    workers=0,
                    patience=10,
                )
                final_epoch = model.trainer.epoch + 1
                if not self.isInterruptionRequested():
                    if results and hasattr(results, 'results_dict'):
                        map50 = results.results_dict.get('metrics/mAP50(B)', 'N/A')
                        map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 'N/A')
                        precision = results.results_dict.get('metrics/precision(B)', 'N/A')
                        recall = results.results_dict.get('metrics/recall(B)', 'N/A')
                        logger.info(f"Fine-tuning complete. Results: mAP50={map50:.6f}, mAP50-95={map50_95:.6f}, Precision={precision:.6f}, Recall={recall:.6f}")
                    else:
                        logger.info("Fine-tuning complete. Accuracy metrics not available.")
            except Exception as e:
                logger.error(f"Fine-tuning failed for event '{game}': {e}")
                logger.error(traceback.format_exc())
                model.clear_callback("on_train_epoch_end")
                raise

            if not self.isInterruptionRequested():
                Path(game_pt).unlink(missing_ok=True)
                try:
                    save_dir = Path(model.trainer.save_dir)
                    best_pt = save_dir / "weights" / "best.pt"
                    shutil.copy2(best_pt, game_pt)
                    logger.info(f"Successfully saved fine-tuned model from {best_pt} as {game_pt.name}")
                except Exception as e:
                    logger.warning(f"Could not copy best.pt to {game_pt.name}: {e}. Attempting direct save.")
                    try:
                        model.save(game_pt)
                    except Exception as save_e:
                        logger.error(f"Failed to save model directly: {save_e}")
                        raise

            try:
                delete_files(image_dir / "train")
                delete_files(label_dir / "train")
                delete_files(image_dir / "val")
                delete_files(label_dir / "val")
            except Exception as e:
                logger.warning(f"Failed to clean up dataset directories: {e}")

            model.clear_callback("on_train_epoch_end")
            if not self.isInterruptionRequested():
                elapsed_ft = time.time() - start_time_ft
                logger.info(f"[{game}] Fine-tuning complete ({final_epoch} epochs) (took {elapsed_ft:.2f}s).")
                self.progress_signal.emit(100, f"{prefix}Fine-tuning complete.")

        return YOLO(game_pt)
