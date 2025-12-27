import time
import traceback
from PySide6.QtCore import QThread, Signal
import sqlite3
from tools import *
import sys
from itertools import zip_longest

resource_path = lambda p: Path(getattr(
    sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__))
    )) / p

class Worker(QThread):
    # メインスレッド（画面）に情報を送るための「通信線」
    progress_signal = Signal(int, str)  # 進捗率(%), メッセージ
    finished_signal = Signal(str)       # 完了時のメッセージ
    error_signal = Signal(str)          # エラー発生時のメッセージ
    visible_signal = Signal(bool)      # プログレスバーの表示/非表示

    def __init__(self, pdf_path: Path, tournament_name: str, db_path: Path, is_md=False) -> None:
        super().__init__()
        self.pdf_path = str(pdf_path)
        self.tournament_name = tournament_name
        self.db_path = str(db_path)
        self.is_md = is_md

    def run(self) -> None:
        """
            別スレッドで実行される処理
        """
        try:
            # --- 処理開始の通知 ---
            self.visible_signal.emit(True)
            self.progress_signal.emit(0, "Loading...")
            
            # 処理本体
            self.conn = sqlite3.connect(self.db_path)
            action = self.executemodel()
            self.conn.close()

            if action:
                # 処理完了
                self.progress_signal.emit(100, "Complete")
                time.sleep(1)
                self.finished_signal.emit("Saved to DB successfully.")

        except Exception as e:
            # エラーが起きたら詳細を画面に送る
            error_msg = traceback.format_exc()
            self.error_signal.emit(f"An error has occurred.\n{e}\n{error_msg}")
            print(error_msg)

        self.visible_signal.emit(False)
        self.progress_signal.emit(0, "")

    def executemodel(self) -> bool:
        """
            指定された大会フォルダ内のPDFを解析し、DBに情報を格納する。
            解析にはYOLOモデルを使用し、必要に応じてファインチューニングも行う。
            解析結果はSQLiteデータベースに保存される。

            Returns:
                bool : 処理が成功したらTrue、失敗したらFalse
        """
        game = self.tournament_name
        num2color = {0: "red", 1: "yellow"}
        work_dir = Path.cwd()
        model_dir = resource_path(Path("complete_model"))

        cur = self.conn.cursor()
        try:    
            cur.execute('INSERT INTO events(name) VALUES (?)', (game,)) #eventテーブルに大会の名前を記述
        except sqlite3.IntegrityError:
            self.conn.rollback()
            self.finished_signal.emit("Event Name has already been used.")
            return False

        event_id = cur.lastrowid #event_idを取得

        doc = fitz.open(self.pdf_path)
        print(self.pdf_path)

        #モデルの定義
        #該当の大会についてファインチューニング済みであればそれを用いる
        game_pt = model_dir / f"{game}.pt"
        if not game_pt.exists():
        #if False:
            self.progress_signal.emit(0, "Preparing fine-tuning...")
            model = YOLO(resource_path(model_dir / "base.pt")) #ベースモデルを選択
        
            ### PDFファイルから画像を400枚程度抽出し、予測を行い疑似ラベルを生成
            dataset_dir = work_dir / "yolo_dataset"
            image_dir = dataset_dir / "images"
            label_dir = dataset_dir / "labels"
            yaml_path = work_dir / "yaml" / "data.yaml"     
            save_images(doc, output_dir=image_dir, save_num=500)
            create_pseudo_label(model, image_dir=image_dir, output_dir=label_dir, threshold=0.75)
            split_train_val(image_dir, label_dir, train_ratio=0.8)
            create_yaml(yaml_path, dataset_dir)

            # コールバック関数を定義 ---
            def on_train_epoch_end(trainer):
                curr = trainer.epoch + 1
                total = trainer.epochs
                self.progress_signal.emit(int(curr/total*100), "Fine-tuning...")

            # コールバック登録
            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            ### 疑似ラベルを用いてモデルのファインチューニングを行う
            model.train(
                data=resource_path(yaml_path),    # データセット（train/val のパスを含む）
                epochs=50,
                imgsz=600,
                iou=0.3,
                conf=0.5,
                save=True,
                exist_ok=True,
                workers=0,      #動作安定のため、シングルスレッドによる実行
                patience=10     #Early Stoppingを10エポックに設定
            )

            Path(game_pt).unlink(missing_ok=True) #game_ptが存在する場合削除
            try:
                #best.ptをcomplete_modelにコピーし、大会名にリネーム
                shutil.copy2(Path("runs/detect/train/weights/best.pt"), game_pt)
            except FileNotFoundError:
                model.save(game_pt)
            
            #画像とラベルを削除
            delete_files(image_dir / "train")
            delete_files(label_dir / "train")
            delete_files(image_dir / "val")
            delete_files(label_dir / "val")

            # 後始末
            model.clear_callback("on_train_epoch_end")
            self.progress_signal.emit(100, "Fine-tuning complete.")
        else: pass
        #break
        model = YOLO(game_pt) #ファインチューニング済みモデルをロード

        with pdfplumber.open(self.pdf_path) as pdf:
            for pn in range(doc.page_count):
                self.progress_signal.emit(int(pn/doc.page_count*100), "Extracting data...")
                page_num = pn + 1
                page_plumber = pdf.pages[pn]
                page_mu = doc[pn]
                text = page_mu.get_text()
                if "Game Results" in text: #新たな試合
                    print(f"Game Results page: {page_num}")
                    
                    scores = extract_game_result(page_plumber) #得点表のdf
                    print(scores)
                    hammers = get_hammer(scores, self.is_md)  #各エンドのハンマー情報
                    print(hammers)
                    team_red = scores.at[0, "team"]
                    team_yellow = scores.at[1, "team"]
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
                    num_end = 1

                elif "Shot by Shot" in text: #新たなエンド
                    #print(num_end)
                    str_end = str(num_end)
                    try:
                        score_red = int(scores.at[0, str_end]) #得点表のdfから得点を取得
                        score_yellow = int(scores.at[1, str_end])
                    except Exception:
                        score_red = None #存在しない場合はNULL
                        score_yellow = None
                    
                    try:
                        color_hammer = num2color[hammers[num_end - 1]]
                    except Exception:
                        color_hammer = None
                    cur.execute("""INSERT INTO ends(game_id, page, number, color_hammer, 
                                    score_red, score_yellow) VALUES (?, ?, ?, ?, ?, ?)""", 
                                    (game_id, page_num, num_end, color_hammer, 
                                    score_red, score_yellow))
                    end_id = cur.lastrowid #end_idを取得
                    print(f"Shot-by-Shot page: {page_num}")
                    stones_end, shot_info = extract_shotbyshot(doc, page_mu, model, self.is_md)
                    #print(stones_end[0])
                    #count += len(shot_info)
                    #count2 += stones_end.shape[0]
                    #print(shot_info)
                    print("num shots: ", len(shot_info))

                    for shot_num, (stones, info) in enumerate(zip_longest(stones_end, shot_info), start=1):
                        if info is not None: #正常時
                            shot_type = info["type"]; percent_score = info["score"]
                            turn = info["turn"]; team = info["team"]; player_name = info["player"]      
                        else: #ショット情報が取れない場合はNULLを挿入し、ストーン配置のみ保存する
                            shot_type = None; percent_score = None
                            turn = None; team = None; player_name = None

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
        return True