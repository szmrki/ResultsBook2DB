import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from detection import *
import cv2
import re
import pdfplumber
import yaml
import random
import shutil

def extract_shotbyshot(doc: fitz.Document, page: fitz.Page, model, is_md=False) -> tuple[np.ndarray, 
                                                list[dict[str, int, str, str, str, int]]]:
    """
        ページからショットバイショット画像を抽出するメソッド
        Args:
            doc : PyMuPDFのオブジェクト
            page : PyMuPDFのページオブジェクト
            model : ストーン検出モデル
            is_md : MD or 4人制
        Returns:
            tuple[np.ndarray, list[tuple[str, str, str, str, int]]] : 
                ストーン座標の配列（num_shots x 16 x 6）とショット情報のリスト
    """
    text = page.get_text()
    text = text.splitlines()
    #print(text)
    shot_info_list = __get_shot_info(text)

    #========================画像取得===========================
    # ページ内の全画像情報を取得（整数の XREF）
    shotbyshot_list, bboxes = __extract_images(doc, page)

    # ショット情報と画像の枚数の確認をする
    # 画像の方が枚数が少ない場合、正しく取得できていない画像が存在するため、補完する
    if len(shot_info_list) > len(shotbyshot_list):
        missings = __found_missing_bbox(bboxes) #欠損位置の検出
        missing_num = len(shot_info_list) - len(shotbyshot_list)
        if missing_num < len(missings): #欠損している数に合わせる
            missings = sorted(missings, key=lambda r: (r.y0, r.x0))
            missings = missings[:missing_num]

        # --- ページ全体をレンダリング ---
        scale = 16   # 16倍解像度
        matrix = fitz.Matrix(scale, scale)   
        full_pix = page.get_pixmap(matrix=matrix)
        full_img = __pixmap2cv2(full_pix)

        # --- PDF座標 → ピクセル座標 ---
        for missing_bbox in missings:
            x0 = int(missing_bbox.x0 * scale)
            y0 = int(missing_bbox.y0 * scale)
            x1 = int(missing_bbox.x1 * scale)
            y1 = int(missing_bbox.y1 * scale)
            cropped = full_img[y0:y1, x0:x1]
            cropped = cv2.resize(cropped, (300, 600))

            shotbyshot_list.append({
                "img": cropped,
                "x": missing_bbox.x0,
                "y": missing_bbox.y0,
            })
            #print("x0=", shotbyshot_list[-1]["x"])
            #print("y0=", shotbyshot_list[-1]["y"])

    # 上→下、左→右でソート(投球順に合わせる)
    shotbyshot_list.sort(key=lambda im: (im["y"], im["x"]))

    if is_md and "prepositioned stones" in [t.lower() for t in text]:
        del shotbyshot_list[0]  #先頭画像を削除
            
    stones_end_list = []
    for img in shotbyshot_list:
        img = img["img"]

        stones = get_stones_pos(img, model)
        stones_end_list.append(stones)

    stones_end = np.array(stones_end_list)  #(num_shots, 16, 6)

    return stones_end, shot_info_list

def extract_game_result(page, is_md=False) -> pd.DataFrame:
    """
        ページからゲーム結果のスコア表を抽出するメソッド
        Args:
            page : pdfplumberのページオブジェクト
            is_md : MD版かどうか
        Returns:
            pd.DataFrame : スコア表データフレーム
            list[int] : パワープレイのエンド番号のリスト
    """
    text = page.extract_text()
    #print(text)
    team_texts = re.findall(r'\b[A-Z]{3} - [^\s\n]+\b', text) #チーム名取得条件を緩和
    #print(team_texts)
    team_red = team_texts[0]
    team_yellow = team_texts[1]
    
    if is_md:
        power_play_ends = []
        # Power Playの情報を抽出
        # 行ごとに分割
        lines = text.split('\n')
        for line in lines:
                # "power play: end " の後ろにある数値のみ抽出
                # 大文字小文字は区別しない
                nums = re.findall(r'power play:\s*end\s+(\d+)', line, re.IGNORECASE)
                power_play_ends.extend([int(n) for n in nums])

    tabs = page.find_tables()
    # 得点表のテーブルを取得
    for table in tabs:
        table = table.extract()
        if any('*' in row for row in table):
            break

    n_cols = len(table[0])
    columns = ["LSFE"] + [str(i) for i in range(1, n_cols-1)] + ["Total"]
    df = pd.DataFrame([[__try_int(cell) for cell in row] for row in table], columns=columns)
    df.insert(0, "team", [team_red, team_yellow])

    if is_md:
        return df, power_play_ends
    return df

def __get_shot_info(all_texts) -> list[dict[str, int, str, str, str, int]]:
    """
        ショットバイショットのテキスト情報から特定の投球の情報を取得する
        Args:     
            all_texts : ショットバイショットのすべてのテキスト情報のリスト
            #is_MD : MDかどうか
        Returns:
            list[dict[str, int, str, str, str, int]] : 
                (チーム名, プレイヤー名, ショットタイプ, 回転方向, ショットスコア)の辞書のリスト
    """
    score_pattern = re.compile(r"^\d+%$|^-$")
    turn_pattern = ("↺", "↻")
    player_pattern = re.compile(r"^[A-Z]{3}: .+$")

    shots = []
    i = 0
    while i < len(all_texts) - 2:
        # --- パターン A：回転あり（4要素） ---
        if i <= len(all_texts) - 4:
            type, score, turn, player = all_texts[i:i+4]

            if score_pattern.match(score) and turn in turn_pattern \
            and player_pattern.match(player):
                if turn == "↻":
                    turn = "cw"
                elif turn == "↺":
                    turn = "ccw"
                else: turn = None
                team, player = player.split(": ")
                score = int(score.rstrip('%')) if '%' in score else None

                shots.append({
                    "type": type,
                    "score": score,
                    "turn": turn,
                    "team": team,
                    "player": player,
                })
                i += 4
                continue

        # --- パターン B：回転なし（3要素） ---
        type, score, player = all_texts[i:i+3]
        if score_pattern.match(score) and player_pattern.match(player):
            team, player = player.split(": ")
            score = int(score.rstrip('%')) if '%' in score else None
            shots.append({
                "type": type,
                "score": score,
                "turn": None,  # 欠損扱い
                "team": team,
                "player": player,
            })
            i += 3
            continue

        # どちらにも該当しない場合は1進める
        i += 1
    
    return shots

def get_hammer(scores: pd.DataFrame, is_md=False) -> list[int]: 
    """
        スコア表ベースでエンドごとのハンマーのindexを取得する
        Args:
            score : スコア表のデータフレーム
            is_md : MDのときはハンマーの取得方法が変わる
        Returns:
            list[int] : 0 or 1のリスト、長さはエンド数
    """
    #LSFE列に*があるチームがラストストーンエンド
    hammer_list = []
    try:
        start = int(scores[scores["LSFE"].astype(str).str.contains(r"\*")].index[0])
    except Exception:
        start = None
    hammer_list.append(start)

    exclude_cols = ["team", "LSFE", "Total"]
    # 対象列（エンド列）を抽出
    end_cols = [col for col in scores.columns if col not in exclude_cols]
    # いずれかの行について中身が空でないセルを True とする
    non_empty = scores[end_cols].astype(str).apply(
                            lambda col: col.str.strip().ne("").any())
    # NaN でないものだけ数える
    total_ends = non_empty.sum()

    for end in range(1, total_ends):
        str_end = str(end)
        try:
            if int(scores.at[0, str_end]) > int(scores.at[1, str_end]): #team0が得点した場合
                hammer_list.append(1)
            elif int(scores.at[0, str_end]) < int(scores.at[1, str_end]): #team1が得点した場合
                hammer_list.append(0)
            else: #ブランクの場合
                if is_md:
                    hammer_list.append(1-hammer_list[-1])  #前のエンドから交代
                else:
                    hammer_list.append(hammer_list[-1])    #前のエンドと同じ
        except ValueError:
            hammer_list.append(None)

    return hammer_list

def save_images(doc: fitz.open, output_dir: Path, save_num: int) -> None:
    """
        PDFからシート画像を指定した枚数抽出し保存する
        Args:
            doc : PyMuPDFのオブジェクト
            output_dir : 画像出力先ディレクトリ名
            save_num : 保存する枚数
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    num_images = 0
    for pn in range(doc.page_count):
        page = doc[pn]
        text = page.get_text()
        if "Shot by Shot" in text:
            shotbyshot_list, _ = __extract_images(doc, page)

            for i,img in enumerate(shotbyshot_list, start=1):
                img = img["img"]
                img[:20,1:-2] = 255
                img[-19:,1:-2] = 255 #白マスク
                cv2.imwrite(output_dir / f"page{pn+1}_{i}.png", img)

            num_images += len(shotbyshot_list)
            if num_images >= save_num: break
        else: continue

def __extract_images(doc: fitz.Document, page: fitz.Page) -> tuple[list, list]:
    """
        PDFからシート画像を抽出し,辞書形式で保持する
        Args:
            doc : PyMuPDFのファイルオブジェクト
            page : PyMuPDFのページオブジェクト
        Returns:
            shotbyshot_list : 各画像の情報の辞書形式をまとめたリスト
                    "img": 画像のnumpy配列
                    "x": 画像左上のx座標
                    "y": 画像左上のy座標
            bboxes : 各画像のbboxの座標をまとめたリスト
    """
    # ページ内の全画像情報を取得（整数の XREF）
    img_list = page.get_images(full=True)
    #print("img_list: ", img_list)

    tmp_shotbyshot_list = []
    for img in img_list:
        width = img[2]
        height = img[3]
        #print(f"width: {width}, height: {height}")
        if 298 <= width <= 302 and 598 <= height <= 602: #基本は300x600
            tmp_shotbyshot_list.append(img)
    #print(len(tmp_shotbyshot_list))

    shotbyshot_list = []
    bboxes = []    #画像補完用
    # ページ内の画像情報を取得
    for img in tmp_shotbyshot_list:
        # 画像のページ上の座標を取得
        bbox = page.get_image_bbox(img)
        x0, y0, x1, y1 = bbox  # 左上(x0,y0)と右下(x1,y1)
        
        #print(f"bbox={bbox}, x0={x0}, y0={y0}, x1={x1}, y1={y1}")
        bboxes.append(bbox)
        
        #画像に変換する
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        img = __pixmap2cv2(pix)
        if __black_more_than_white(img):
            img = 255 - img  #反転

        shotbyshot_list.append({
            "img": img,
            "x": x0,
            "y": y0,
        })
    return shotbyshot_list, bboxes

def split_train_val(image_dir: Path, label_dir: Path, train_ratio=0.8, seed=42) -> None:
    """
        画像とラベルを訓練用と検証用に分割する
        Args:
            image_dir : 画像のパス
            label_dir : ラベルのパス
            train_ratio : 訓練データの割合(0~1)
            seed : 乱数用のシード
    """
    random.seed(seed)

    # 出力フォルダ作成
    train_img_dir = image_dir / "train"
    val_img_dir = image_dir / "val"
    train_lbl_dir = label_dir / "train"
    val_lbl_dir = label_dir / "val"

    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    #画像フォルダ初期化
    delete_files(train_img_dir)
    delete_files(train_lbl_dir)
    delete_files(val_img_dir)
    delete_files(val_lbl_dir)

    # 画像一覧取得（.png限定）
    images = [f for f in os.listdir(image_dir) if f.endswith(".png")]

    for img_name in images:
        base = os.path.splitext(img_name)[0]
        lbl_name = base + ".txt"
        img_path = image_dir / img_name
        lbl_path = label_dir / lbl_name

        # ラベルが無い場合はスキップ
        if not lbl_path.exists():
            print(f"[警告] ラベルが無いためスキップ: {img_name}")
            continue

        # train or val に振り分け
        if random.random() < train_ratio:
            dst_img = train_img_dir
            dst_lbl = train_lbl_dir
        else:
            dst_img = val_img_dir
            dst_lbl = val_lbl_dir

        # ファイル移動
        shutil.move(img_path, dst_img / img_name)
        shutil.move(lbl_path, dst_lbl / lbl_name)

def create_yaml(
        save_path: Path,
        dataset_root: Path,
        train_path=Path("images/train"),
        val_path=Path("images/val"),
        ) -> None:
    """
        YOLOの学習用のyamlファイルを作成する
        Args: 
            save_path : 保存ファイル名
            dataset_root : データのパス
            train : 訓練画像のパス
            val : 検証画像のパス
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    names = ["red", "yellow"]

    yaml_dict = {
        "path": str(dataset_root),
        "train": str(train_path),
        "val": str(val_path),
        "names": {i: name for i, name in enumerate(names)}
    }

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, allow_unicode=True)

def delete_files(dir) -> None:
    """
        指定されたディレクトリ内のすべてのファイルを削除する
        Args: 
            dir : 削除対象のディレクトリ
    """
    # フォルダ内の全ファイルを削除（サブフォルダは無視）
    for file in Path(dir).iterdir():
        if file.is_file():
            file.unlink()

# 数値に変換できるものは int、できないものはそのまま
def __try_int(x):
    try:
        return int(x)
    except ValueError:
        return x

def __found_missing_bbox(bboxes) -> list[fitz.Rect]:
    """
        検出できずに欠落している画像の位置を検出する
        Args: 
            bboxes : 検出済みの画像 bbox（Rect）のリスト
        Returns: 
            missings : 欠落位置のRectオブジェクトのリスト
    """
    # bbox から (x0, y0) のみに簡略化して抽出
    points = [(round(b.x0, 4), round(b.y0, 4)) for b in bboxes]
    actual = set(points)

    # ユニークな x 行列・y 行列をソート
    xs = sorted({p[0] for p in points})
    ys = sorted({p[1] for p in points})

    expected = set()

    # 上2行（6枚）
    for y in ys[:2]:        # 1行目・2行目
        for x in xs:        # 全 x（6個）
            expected.add((x, y))

    # 3行目（4枚・左詰め）
    lower4 = xs[:4]         # 左側から4つ
    for x in lower4:
        expected.add((x, ys[2]))
    
    missing = expected - actual
    print("欠落している画像位置:", missing)

    # 幅・高さの推定（最も安定）
    # 同じ行の既存画像と比較する
    missings = []
    for mx, my in missing:
        row_y = my
        same_row = [b for b in bboxes if round(b.y0,1) == row_y]

        if same_row:
            # 行内の幅は同じはず
            width = same_row[0].width
            height = same_row[0].height
        else:
            # fallback（近い行の画像サイズ）
            width = bboxes[0].width
            height = bboxes[0].height
        missing_bbox = fitz.Rect(mx, my, mx + width, my + height)
        missings.append(missing_bbox)

    return missings

def __pixmap2cv2(pix: fitz.Pixmap) -> np.ndarray:
    """
        pixmapをBGR形式のnumpy配列に変換する
        Args:
            pix : fitz.Pixmapオブジェクト
        Returns:
            img : BGR形式のnumpy配列
    """
    if pix.n >= 5:
        pix = fitz.Pixmap(fitz.csRGB, pix) #RGBに変換
    # Pixmap.samples は bytes なので numpy配列に変換
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    # 高さ・幅・チャンネル数に reshape
    img = img.reshape(pix.height, pix.width, pix.n)
    # RGB → BGR（OpenCV形式）
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img

def __black_more_than_white(image_array: np.ndarray) -> bool:
    """
        画像内の白(255, 255, 255)と黒(0, 0, 0)のピクセル数を比較する
        Args:
            image_array : 画像のnumpy配列
        Returns:
            bool : 黒≧白ならTrue、そうでなければFalse
    """
    # 白ピクセルの判定: 各ピクセルの(R,G,B)がすべて255であるか
    white_pixels = np.all(image_array == [255, 255, 255], axis=-1)
    white_count = np.sum(white_pixels)
    
    # 黒ピクセルの判定: 各ピクセルの(R,G,B)がすべて0であるか
    black_pixels = np.all(image_array == [0, 0, 0], axis=-1)
    black_count = np.sum(black_pixels)
    
    if white_count >= black_count:
        return False
    else:
        return True
    
if __name__ == "__main__":
    """work_dir = Path.cwd()
    dataset_dir = work_dir / "yolo_dataset"
    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"
    yaml_path = work_dir / "yaml" / "data.yaml"
    model_dir = work_dir / "complete_model"
    model = YOLO(Path("complete_model/base.pt"))
    #画像とラベルを削除
    delete_files(image_dir / "train")
    delete_files(label_dir / "train")
    delete_files(image_dir / "val")
    delete_files(label_dir / "val")"""
    
    file_path = Path("rb_data/data_md/WMDCC2025/WMDCC2025_ResultsBook.pdf")
    #file_path = Path("rb_data/data_4p/ECC2023Men/ECC2023_ResultsBook_Men_A-Division.pdf")
    doc = fitz.open(file_path)
    page = doc[9]
    text = page.get_text()
    text = text.splitlines()
    print("prepositioned stones" in [t.lower() for t in text])
    shot_info_list = __get_shot_info(text)
    print(len(shot_info_list))
    shotbyshot_list, bboxes = __extract_images(doc, page)
    if "prepositioned stones" in [t.lower() for t in text]:
        del shotbyshot_list[0]  #先頭画像を削除
        del bboxes[0]
    print(len(shotbyshot_list), len(bboxes))
    #print(text)
    """
    save_images(doc, output_dir=image_dir, save_num=500)
    create_pseudo_label(model, image_dir=image_dir, output_dir=label_dir, threshold=0.75)
    """
    """
    split_train_val(image_dir, label_dir, train_ratio=0.8)
    create_yaml(yaml_path, dataset_dir)

    ### 疑似ラベルを用いてモデルのファインチューニングを行う
    model.train(
        data=yaml_path,    # データセット（train/val のパスを含む）
        epochs=50,
        imgsz=600,
        iou=0.3,
        conf=0.5,
        #save=False,
        exist_ok=True,
    )
    game_pt = model_dir / "OWG2022.pt"
    try:
        #best.ptをcomplete_modelに移動し、大会名にリネーム
        if game_pt.is_file():
            game_pt.unlink()
        Path("runs/detect/train/weights/best.pt").rename(game_pt) 
    except FileNotFoundError:
        model.save(game_pt)

    models = [model_dir / "base.pt", game_pt]
    for m in models:
        model = YOLO(m)
        metrics = model.val(
            data=yaml_path,
            imgsz=600,
            iou=0.3,
            conf=0.5,
        )
        print(m, metrics.box.map50, metrics.box.map)
    #scores = extract_game_result(page)
    #print(scores)
    """

    """
    # --- ページ全体をレンダリング ---
    scale = 1
    matrix = fitz.Matrix(scale, scale)   
    full_pix = page.get_pixmap(matrix=matrix)
    test_img = __pixmap2cv2(full_pix)
    """
    """
    img_list, _ = __extract_images(doc, page)
    test_img = img_list[0]["img"]
    #test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
    print("before", __black_more_than_white(test_img))
    test_img = 255 - test_img
    print("after", __black_more_than_white(test_img))
    #test_img = test_img[..., ::-1]
    print(test_img.shape)
    """
    #cv2.imshow("test", test_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
