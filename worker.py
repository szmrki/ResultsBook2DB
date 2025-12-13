import time
import traceback
from PySide6.QtCore import QThread, Signal
import sqlite3
from tools import *

class Worker(QThread):
    # メインスレッド（画面）に情報を送るための「通信線」
    progress_signal = Signal(int, str)  # 進捗率(%), メッセージ
    finished_signal = Signal(str)       # 完了時のメッセージ
    error_signal = Signal(str)          # エラー発生時のメッセージ
    visible_signal = Signal(bool)      # プログレスバーの表示/非表示

    def __init__(self, pdf_path, tournament_name, db_path) -> None:
        super().__init__()
        self.pdf_path = pdf_path
        self.tournament_name = tournament_name
        self.db_path = db_path

    def run(self) -> None:
        """
            別スレッドで実行される処理
        """
        try:
            # --- 処理開始の通知 ---
            self.visible_signal.emit(True)
            self.progress_signal.emit(0, "PDFを読み込んでいます...")
            
            # 【ここにご自身の既存ロジックを組み込みます】
            # 例: pdf_images = convert_from_path(self.pdf_path)
            self.conn = sqlite3.connect(self.db_path)
            #self.progress_signal.emit(0, "解析中...")
            self.executemodel(self.tournament_name, self.pdf_path)
            self.conn.close()
            
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
            self.progress_signal.emit(100, "解析が完了しました。DBに保存しています...")
            time.sleep(1)
            self.finished_signal.emit("すべての解析とDB保存が完了しました！")
            self.visible_signal.emit(False)
            self.progress_signal.emit(0, "")

        except Exception as e:
            # エラーが起きたら詳細を画面に送る
            error_msg = traceback.format_exc()
            self.error_signal.emit(f"エラーが発生しました:\n{e}\n{error_msg}")

    def executemodel(self, game, file_path, is_MD=False) -> None:
        """
            指定された大会フォルダ内のPDFを解析し、DBに情報を格納する。
            解析にはYOLOモデルを使用し、必要に応じてファインチューニングも行う。
            解析結果はSQLiteデータベースに保存される。
            Args:
                game (str): 大会名（フォルダ名）
                file_path (str): PDFファイルのパス
                is_MD (bool): MD版かどうかのフラグ
            Returns:
                None
        """
        #color2num = {"red": 0, "yellow": 1}
        num2color = {0: "red", 1: "yellow"}
        work_dir = Path.cwd()
        model_dir = work_dir / "complete_model"

        cur = self.conn.cursor()
        cur.execute('INSERT INTO event(name) VALUES (?)', (game,)) #eventテーブルに大会の名前を記述
        event_id = cur.lastrowid #event_idを取得

        #file_path = glob.glob(f"{game}/*.pdf")[0]
        doc = fitz.open(file_path)
        print(file_path)

        #モデルの定義
        #該当の大会についてファインチューニング済みであればそれを用いる
        game_pt = model_dir / f"{game}.pt"
        if not game_pt.exists():
        #if False:
            self.progress_signal.emit(0, "ファインチューニング準備中...")
            model = YOLO(model_dir / "base.pt") #ベースモデルを選択
        
            ### PDFファイルから画像を400枚程度抽出し、予測を行い疑似ラベルを生成
            dataset_dir = work_dir / "yolo_dataset"
            image_dir = dataset_dir / "images"
            label_dir = dataset_dir / "labels"
            yaml_path = work_dir / "yaml" / "data.yaml"     
            save_images(doc, output_dir=image_dir, save_num=500)
            create_pseudo_label(model, image_dir=image_dir, output_dir=label_dir, threshold=0.75)
            split_train_val(image_dir, label_dir, train_ratio=0.8)
            create_yaml(yaml_path, dataset_dir)

            # --- コールバック関数をここで定義 ---
            def on_train_epoch_end(trainer):
                curr = trainer.epoch + 1
                total = trainer.epochs
                # ★ selfを使ってシグナルを送る
                self.progress_signal.emit(int(curr/total*100), "ファインチューニング中...")

            # コールバック登録
            model.add_callback("on_train_epoch_end", on_train_epoch_end)

            ### 疑似ラベルを用いてモデルのファインチューニングを行う
            model.train(
                data=yaml_path,    # データセット（train/val のパスを含む）
                epochs=50,
                imgsz=600,
                iou=0.3,
                conf=0.5,
                save=False,
                exist_ok=True,
                #verbose=False
            )

            Path(game_pt).unlink(missing_ok=True) #game_ptが存在する場合削除
            try:
                #best.ptをcomplete_modelに移動し、大会名にリネーム
                Path("runs/detect/train/weights/best.pt").rename(game_pt) 
            except FileNotFoundError:
                model.save(game_pt)
            
            #画像とラベルを削除
            delete_files(image_dir / "train")
            delete_files(label_dir / "train")
            delete_files(image_dir / "val")
            delete_files(label_dir / "val")

            # 後始末
            model.clear_callback("on_train_epoch_end")

            self.progress_signal.emit(100, "ファインチューニング完了")
        else: pass
        #break
        model = YOLO(game_pt) #ファインチューニング済みモデルをロード

        with pdfplumber.open(file_path) as pdf:
            for pn in range(doc.page_count):
                self.progress_signal.emit(int(pn/doc.page_count*100), "解析中...")
                page_num = pn + 1
                page_plumber = pdf.pages[pn]
                page_mu = doc[pn]
                text = page_mu.get_text()
                if "Game Results" in text: #新たな試合
                    print(f"Game Results page: {page_num}")
                    
                    scores = extract_game_result(page_plumber) #得点表のdf
                    print(scores)
                    hammers = get_hammer(scores, is_MD)  #各エンドのハンマー情報
                    print(hammers)
                    team_red = scores.at[0, "team"]
                    team_yellow = scores.at[1, "team"]
                    try:
                        fin_red = int(scores.at[0, "Total"])
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
                        score_red = int(scores.at[0, str_end])
                        score_yellow = int(scores.at[1, str_end])
                    except Exception:
                        score_red = None
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
                    stones_end, shot_info = extract_shotbyshot(doc, page_mu, model, is_MD)
                    #print(stones_end[0])
                    #count += len(shot_info)
                    #count2 += stones_end.shape[0]
                    #print(shot_info)
                    print("num shots: ", len(shot_info))
                    
                    if stones_end.shape[0] != len(shot_info):
                        print("num images: ", stones_end.shape[0])
                        break
                    
                    #if page_num == 146: break

                    for shot_num, (stones, info) in enumerate(zip(stones_end, shot_info), start=1):
                        shot_type = info["type"]; percent_score = info["score"]
                        turn = info["turn"]; team = info["team"]; player_name = info["player"]
                        shot_color = num2color[(hammers[num_end - 1] + (shot_num % 2)) % 2] #現在のショットの色を指定
                        cur.execute("""INSERT INTO shots(end_id, number, color, team, player_name, 
                                        type, turn, percent_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", 
                                        (end_id, shot_num, shot_color, team, player_name, 
                                        shot_type, turn, percent_score))
                        shot_id = cur.lastrowid #shot_idを取得

                        rows = [(shot_id, num2color[int(row[0])], *row[1:]) for row in stones if row[5] == 1]
                        if len(rows) == 0:
                            rows = [(shot_id, None, None, None, None, None, None)]
                        cur.executemany("""INSERT INTO stones (shot_id, color, x, y, dist, 
                                        inhouse, insheet) VALUES (?, ?, ?, ?, ?, ?, ?)""", rows)
                    num_end += 1
                    #break
                else:
                    continue
        doc.close()
        self.conn.commit()