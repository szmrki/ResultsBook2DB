import time
import traceback
from PySide6.QtCore import QThread, Signal
import sqlite3
from tools import *
import sys
from itertools import zip_longest
import io
import logging
import re

logger = logging.getLogger(__name__)

resource_path = lambda p: Path(getattr(
    sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__))
    )) / p

class Worker(QThread):
    # メインスレッド（画面）に情報を送るための「通信線」
    progress_signal = Signal(int, str)  # 進捗率(%), メッセージ
    finished_signal = Signal(str)       # 完了時のメッセージ
    error_signal = Signal(str)          # エラー発生時のメッセージ
    visible_signal = Signal(bool)      # プログレスバーの表示/非表示

    def __init__(self, pdf_entries: list, db_path: Path, is_md=False) -> None:
        super().__init__()
        # pdf_entries: list of {"path": Path, "event_name": str}
        self.pdf_entries = pdf_entries
        self.db_path = str(db_path)
        self.is_md = is_md

    def run(self) -> None:
        """
            別スレッドで実行される処理（複数PDF対応）
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

            # セッション開始時に runs/detect ディレクトリをクリア
            runs_dir = Path("runs/detect")
            if runs_dir.exists():
                logger.info(f"Cleaning up runs directory: {runs_dir}")
                shutil.rmtree(runs_dir)
            
            # 処理本体
            self.conn = sqlite3.connect(self.db_path)
            
            start_time_all = time.time()
            errors = []
            
            for i, entry in enumerate(self.pdf_entries, start=1):
                # 解析中にファイルが追加された場合も分母を現在の総数に合わせる（[2/2] のように表示）
                total = max(len(self.pdf_entries), i)
                pdf_path = str(entry["path"])
                tournament_name = entry["event_name"]
                prefix = f"[{i}/{total}] "
                
                try:
                    success = self.executemodel(pdf_path, tournament_name, prefix)
                    if not success:
                        err_msg = f"{entry['path'].name}: Event Name '{tournament_name}' は既に使用されています"
                        errors.append(err_msg)
                        logger.error(err_msg)
                except Exception as e:
                    error_msg = traceback.format_exc()
                    errors.append(f"{entry['path'].name}: {e}")
                    logger.error(error_msg)
            
            self.conn.close()
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

    def executemodel(self, pdf_path: str, tournament_name: str, prefix: str = "") -> bool:
        """
            指定された大会フォルダ内のPDFを解析し、DBに情報を格納する。
            解析にはYOLOモデルを使用し、必要に応じてファインチューニングも行う。
            解析結果はSQLiteデータベースに保存される。

            Args:
                pdf_path (str): PDFファイルのパス
                tournament_name (str): 大会名
                prefix (str): 進捗メッセージの接頭辞（例: "[1/3] "）

            Returns:
                bool : 処理が成功したらTrue、失敗したらFalse
        """
        game = tournament_name
        num2color = {0: "red", 1: "yellow"}
        work_dir = Path.cwd()
        model_dir = resource_path(Path("complete_model"))

        cur = self.conn.cursor()
        try:
            #eventテーブルに大会名、年、カテゴリを記述
            year, category = self.__extract_year_and_category(game)
            cur.execute('INSERT INTO events(name, year, category) VALUES (?, ?, ?)', (game, year, category)) 
        except sqlite3.IntegrityError:
            self.conn.rollback()
            logger.warning(f"Duplicate event name found in database: {game}")
            return False

        event_id = cur.lastrowid #event_idを取得

        doc = fitz.open(pdf_path)
        logger.info(f"Processing PDF: {pdf_path}")

        #モデルの定義
        #該当の大会についてファインチューニング済みであればそれを用いる
        game_pt = model_dir / f"{game}.pt"
        if not game_pt.exists():
        #if False:
            start_time_ft = time.time()
            self.progress_signal.emit(0, f"{prefix}Preparing fine-tuning...")
            model = YOLO(resource_path(model_dir / "base.pt")) #ベースモデルを選択
        
            ### PDFファイルから画像を400枚程度抽出し、予測を行い疑似ラベルを生成
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

            # コールバック関数を定義 ---
            def on_train_epoch_end(trainer):
                curr = trainer.epoch + 1
                total = trainer.epochs
                self.progress_signal.emit(int(curr/total*100), f"{prefix}Fine-tuning...")

            # コールバック登録
            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            ### 疑似ラベルを用いてモデルのファインチューニングを行う
            try:
                logger.info(f"Starting fine-tuning for event: {game}")
                results = model.train(
                    data=resource_path(yaml_path),    # データセット（train/val のパスを含む）
                    epochs=50,
                    imgsz=600,
                    iou=0.3,
                    conf=0.5,
                    save=True,
                    exist_ok=False, # フォルダを自動でインクリメント(train, train2...)
                    workers=0,      # 動作安定のため、シングルスレッドによる実行
                    patience=5,     # Early Stoppingを5エポックに設定
                )
                # 学習結果の要約をログに記録
                final_epoch = model.trainer.epoch + 1
                if results and hasattr(results, 'results_dict'):
                    map50 = results.results_dict.get('metrics/mAP50(B)', 'N/A')
                    map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 'N/A')
                    precision = results.results_dict.get('metrics/precision(B)', 'N/A')
                    recall = results.results_dict.get('metrics/recall(B)', 'N/A')
                    logger.info(f"""Fine-tuning complete. Results: mAP50={map50:.6f}, mAP50-95={map50_95:.6f}, Precision={precision:.6f}, Recall={recall:.6f}""")
                else:
                    logger.info(f"Fine-tuning complete. Accuracy metrics not available.")
            except Exception as e:
                logger.error(f"Fine-tuning failed for event '{game}': {e}")
                logger.error(traceback.format_exc())
                model.clear_callback("on_train_epoch_end")
                raise

            Path(game_pt).unlink(missing_ok=True) #game_ptが存在する場合削除
            try:
                # model.trainer.save_dir から実際の保存先を取得してコピー
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
            
            #画像とラベルを削除
            try:
                delete_files(image_dir / "train")
                delete_files(label_dir / "train")
                delete_files(image_dir / "val")
                delete_files(label_dir / "val")
            except Exception as e:
                logger.warning(f"Failed to clean up dataset directories: {e}")

            # 後始末
            model.clear_callback("on_train_epoch_end")
            elapsed_ft = time.time() - start_time_ft
            logger.info(f"[{game}] Fine-tuning complete ({final_epoch} epochs) (took {elapsed_ft:.2f}s).")
            self.progress_signal.emit(100, f"{prefix}Fine-tuning complete.")
        else: pass
        #break
        model = YOLO(game_pt) #ファインチューニング済みモデルをロード

        start_time_det = time.time()
        with pdfplumber.open(pdf_path) as pdf:
            for pn in range(doc.page_count):
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
                    except ValueError:
                        fin_red = None
                        fin_yellow = None
                    
                    cur.execute("""INSERT INTO games(event_id, page, team_red, team_yellow, 
                                    final_score_red, final_score_yellow) VALUES (?, ?, ?, ?, ?, ?)""", 
                                    (event_id, page_num, team_red, team_yellow, fin_red, fin_yellow))
                    game_id = cur.lastrowid #game_idを取得

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
                    
                    stones_end, shot_info = extract_shotbyshot(doc, page_mu, model, self.is_md)
                    logger.info(f"[{game_context}] End {num_end} - Shot-by-Shot page: {page_num} - Number of shots: {max(len(stones_end), len(shot_info))}")

                    for shot_num, (stones, info) in enumerate(zip_longest(stones_end, shot_info), start=1):
                        if info is not None: #正常時
                            shot_type = info["type"]; percent_score = info["score"]
                            turn = info["turn"]; team = info["team"]; player_name = info["player"]      
                        else: #ショット情報が取れない場合はNULLを挿入し、ストーン配置のみ保存する
                            shot_type = None; percent_score = None
                            turn = None; team = None; player_name = None
                            logger.warning(f"[{game_context}] End {num_end} - Shot {shot_num} - Shot info not found")

                        shot_color = num2color[(hammers[num_end - 1] + (shot_num % 2)) % 2] #現在のショットの色を指定
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
        self.conn.commit()
        elapsed_det = time.time() - start_time_det
        logger.info(f"[{game}] Detection complete (took {elapsed_det:.2f}s).")
        return True

    def __extract_year_and_category(self, game):
        # 大会名(game)から西暦(year)を抽出
        year_match = re.search(r'\d{4}', game)
        year = int(year_match.group()) if year_match else None
        
        # カテゴリの特定
        category = None
        if self.is_md:
            category = "MD"
        else:
            if "WJCC" in game:
                if "Women" in game:
                    category = "Junior Women"
                elif "Men" in game:
                    category = "Junior Men"
            else:
                if "Women" in game:
                    category = "Women"
                elif "Men" in game:
                    category = "Men"
                else:
                    if "WMCC" in game:
                        category = "Men"
                    elif "WWCC" in game:
                        category = "Women"
                    else:
                        category = None
        return year, category