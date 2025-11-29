import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from detection import get_stones_pos
import cv2
import re
import pdfplumber

def extract_shotbyshot(doc, page, model, is_MD = False) -> tuple[np.ndarray, 
                                                                 list[dict[str, int, str, str, str, int]]]:
    """
        ページからショットバイショット画像を抽出するメソッド
        Args:
            doc : PyMuPDFのオブジェクト
            page : PyMuPDFのページオブジェクト
            model : ストーン検出モデル
            is_MD : MDかどうか
        Returns:
            tuple[np.ndarray, list[tuple[str, str, str, str, int]]] : 
                ストーン座標の配列（num_shots x 16 x 6）とショット情報のリスト
    """
    text = page.get_text()
    text = text.splitlines()
    print(text)
    shot_info_list = __get_shot_info(text)

    #========================画像取得===========================
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

        shotbyshot_list.append({
            "img": img,
            "x": x0,
            "y": y0,
        })

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
            
    stones_end_list = []
    for i,img in enumerate(shotbyshot_list, start=1):
        img = img["img"]
        #cv2.imwrite(f"C:/b4/rb2db/tmp/page{page.number+1}_{i}.png", img)

        stones = get_stones_pos(img, model)
        stones_end_list.append(stones)

    stones_end = np.array(stones_end_list)  #(num_shots, 16, 6)

    return stones_end, shot_info_list

def extract_game_result(page) -> pd.DataFrame:
    """
        ページからゲーム結果のスコア表を抽出するメソッド
        Args:
            page : pdfplumberのページオブジェクト
        Returns:
            pd.DataFrame : スコア表データフレーム
    """
    text = page.extract_text()
    team_texts = re.findall(r'\b[A-Z]{3} - [A-Za-z]+\b', text)
    #print(team_texts)
    team_red = team_texts[0]
    team_yellow = team_texts[1]
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

    return df

def __get_shot_info(all_texts) -> list[dict[str, int, str, str, str, int]]:
    """
        ショットバイショットのテキスト情報から特定の投球の情報を取得するメソッド
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

def get_hammer(scores, is_md=False) -> list[int]: 
    """
        スコア表ベースでエンドごとのハンマーのindexを取得するメソッド
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
        """
        if end <= MAX_END:
            str_end = str(end)
        else:
            extra_num = end - MAX_END
            col_idx = scores.columns.get_loc(str(MAX_END)) 
            str_end = scores.columns[col_idx + extra_num]     # 右隣の列名
        """
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
    # Pixmap.samples は bytes なので NumPy 配列に変換
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    # 高さ・幅・チャンネル数に reshape
    img = img.reshape(pix.height, pix.width, pix.n)
    # RGB → BGR（OpenCV形式）
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img
    
if __name__ == "__main__":
    file_path = "rb_data/data_4p/PCCC2022Men/PCCC2022_ResultsBook_Men_A-Division.pdf"
    #doc = fitz.open(file_path)
    #json = pymupdf4llm.to_json(doc)
    #print(json)
    #page = doc[6]
    #scores = extract_game_result(page)
    #print(scores)