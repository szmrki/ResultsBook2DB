import fitz  # PyMuPDF
#import pymupdf.layout
#import pymupdf4llm
import pandas as pd
import numpy as np
from detection import get_stones_pos
import cv2
import re
import pdfplumber

def extract_shotbyshot(page, doc, pn, model, is_MD = False) -> tuple[np.ndarray, list[tuple[str, str, str, str, int]]]:
    """
        ページからショットバイショット画像を抽出するメソッド
        Args:
            page : pdfplumberのページオブジェクト
            doc : PyMuPDFのオブジェクト
            pn : ページ番号
            model : ストーン検出モデル
            is_MD : MDかどうか
        Returns:
            tuple[np.ndarray, list[tuple[str, str, str, str, int]]] : 
                ストーン座標の配列（num_shots x 16 x 6）とショット情報のリスト
    """
    text = page.extract_text()
    text = text.splitlines()
    #print(text)
    shot_info_list = __get_shot_info(text, is_MD=is_MD)

    #========================画像取得===========================
    page_img = doc[pn]
    # ページ内の全画像情報を取得（整数の XREF）
    img_list = page_img.get_images(full=True)
    #print(img_list)

    tmp_shotbyshot_list = []
    for img in img_list:
        width = img[2]
        height = img[3]
        if width == 300 and height == 600:
            tmp_shotbyshot_list.append(img)
    #print(len(tmp_shotbyshot_list))

    shotbyshot_list = []
    # ページ内の画像情報を取得
    for img in tmp_shotbyshot_list:
        # 画像のページ上の座標を取得
        bbox = page_img.get_image_bbox(img)
        x0, y0, x1, y1 = bbox  # 左上(x0,y0)と右下(x1,y1)
        
        #print(f"bbox={bbox}, x0={x0}, y0={y0}, x1={x1}, y1={y1}")

        shotbyshot_list.append({
            "item": img,
            "x": x0,
            "y": y0,
        })

    # 上→下、左→右でソート(投球順に合わせる)
    shotbyshot_list.sort(key=lambda im: (im["y"], im["x"]))

    stones_end_list = []
    for img in shotbyshot_list:
        xref = img["item"][0]
        pix = fitz.Pixmap(doc, xref)

        if pix.n >= 5:
            pix = fitz.Pixmap(fitz.csRGB, pix) #RGBに変換

        # Pixmap.samples は bytes なので NumPy 配列に変換
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        # 高さ・幅・チャンネル数に reshape
        img = img.reshape(pix.height, pix.width, pix.n)
        # RGB → BGR（OpenCV形式）
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

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
    #team_texts = [t for t in text.splitlines() if re.search(r"[A-Z]{3} - ", t)]
    print(team_texts)
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

def __get_shot_info(all_texts, is_MD=False) -> list[tuple[str, str, str, str, int]]:
    """
        ショットバイショットのテキスト情報から特定の投球の情報を取得するメソッド
        Args:     
            all_texts : ショットバイショットのすべてのテキスト情報のリスト
            is_MD : MDかどうか
        Returns:
            list[tuple[str, str, str, str, int]] : 
                (チーム名, プレイヤー名, ショットタイプ, 回転方向, ショットスコア)のタプルのリスト
    """
    
    tplayers = [t for t in all_texts if t and ": " in t]
    players = [
        [part.strip() for part in re.split(r'(?=[A-Z]{3}: )', s) if part.strip() != ""]
        for s in tplayers
        ]
    players = [item for sublist in players for item in sublist]
    players = [t.split(": ") for t in players]

    shots = [t for t in all_texts if any(k in t for k in ('↺', '↻'))]
    
    # ↺100% / ↻100% の前にスペースを挿入（これで split が可能になる）
    normalized = [re.sub(r'(?<=[↺↻])(?=\d+%)', ' ', s) for s in shots]
    # リスト内包表記だけで “技名を1語にまとめて抽出”
    shots = [
        token.strip()
        for s in normalized
        for token in re.findall(r'[A-Za-z\- ]+|↺|↻|\d+%', s)
        if token.strip()
    ]
    
    turns = [t for t in shots if t in ('↺', '↻')]
    turns = ['ccw' if t == '↺' else 'cw' for t in turns]
    scores = [t for t in shots if t and '%' in t]
    scores = [int(s.rstrip('%')) for s in scores]
    shot_types = [
        s for s in shots 
        if re.fullmatch(r'[A-Za-z\- ]+', s)
    ]

    print("shots: ", shots)
    print("players: ", players)
    print("turns: ", turns)
    print("scores: ", scores)

    tn = 10 if is_MD else 16
    while len(turns) < tn: #エラー回避のため長さを最大投球数にそろえる
        turns.append(None)
    while len(scores) < tn:
        scores.append(None)
    while len(players) < tn: #WMDCC2023において1投目にプレイヤーが記載されていないことがあったため先頭に空白を追加
        if len(shot_types) == tn: #1エンドすべて投球されている場合
            players.insert(0, [None, None])
        else:  #コンシード等ですべて投球されていない場合
            players.append([None, None])
    while len(shot_types) < tn:
        shot_types.append(None)
    print(shot_types)

    shot_info = []
    for idx in range(tn):
        shot_team = players[idx][0]
        shot_player = players[idx][1]
        shot_type = shot_types[idx]
        shot_turn = turns[idx]
        shot_score = scores[idx]

        shot_info.append((shot_team, shot_player, shot_type, shot_turn, shot_score))
    
    return shot_info

def get_hammer(scores, is_md=False) -> list[int]: 
    """
        スコア表ベースでエンドごとのハンマーのindexを取得するメソッド
        Args:
            score : スコア表のデータフレーム
            is_md : MDのときはハンマーの取得方法が変わる
        Returns:
            list[int] : 0 or 1のリスト、長さはエンド数
    """
    MAX_END = 8 if is_md else 10  #最大エンド数
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
    # 数値に変換（数値でなければ NaN になる）
    #numeric = pd.to_numeric(scores[end_cols].iloc[0], errors="coerce")
    # 1行目（例）について中身が空でないセルを True とする
    non_empty = scores[end_cols].iloc[0].astype(str).str.strip().ne("")
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

    
if __name__ == "__main__":
    file_path = "rb_data/data_4p/PCCC2022Men/PCCC2022_ResultsBook_Men_A-Division.pdf"
    #doc = fitz.open(file_path)
    #json = pymupdf4llm.to_json(doc)
    #print(json)
    #page = doc[6]
    #scores = extract_game_result(page)
    #print(scores)