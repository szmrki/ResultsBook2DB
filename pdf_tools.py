"""
pdf_tools.py: PDFの解析・画像抽出処理を担うモジュール
PyMuPDFやpdfplumberを使用して、競技結果のスコア表やストーンの座標画像を抽出・整形します。
"""
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from detection import get_stones_pos
import cv2
import re
import pdfplumber
import logging
from ultralytics import YOLO
from typing import Any
from pathlib import Path

logger = logging.getLogger(__name__)

def extract_shotbyshot(doc: fitz.Document, page: fitz.Page, model: YOLO, is_md: bool = False) -> tuple[np.ndarray, 
                                                list[dict[str, str | int | None]]]:
    """
        ページからショットバイショット画像を抽出するメソッド
        Args:
            doc : PyMuPDFのオブジェクト
            page : PyMuPDFのページオブジェクト
            model : ストーン検出モデル
            is_md : MD or 4人制
    Returns:
        tuple[np.ndarray, list[dict[str, str | int | None]]]: 
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
            #logger.debug(f"x0={shotbyshot_list[-1]['x']}")
            #logger.debug(f"y0={shotbyshot_list[-1]['y']}")

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

def extract_game_result(page: pdfplumber.page.Page, is_md: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, list[int]]:
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
    if len(team_texts) == 0:
        logger.warning("Team names not found.")
        team_red = None
        team_yellow = None
    elif len(team_texts) == 1:
        logger.warning("Team yellow names not found.")
        team_red = team_texts[0]
        team_yellow = None
    else:
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

    if df.empty:
        logger.warning("Extracted game result dataframe is empty.")
    else:
        logger.debug(f"Successfully extracted game result:\n{df}")
    
    if is_md:
        return df, power_play_ends
    return df

def __get_shot_info(all_texts: list[str]) -> list[dict[str, str | int | None]]:
    """
        ショットバイショットのテキスト情報から特定の投球の情報を取得する
        Args:     
            all_texts : ショットバイショットのすべてのテキスト情報のリスト
            #is_MD : MDかどうか
        Returns:
            list[dict[str, str | int | None]] : 
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

def save_images(doc: fitz.Document, output_dir: Path, save_num: int) -> int:
    """
        PDFからシート画像を指定した枚数抽出し保存する
        Args:
            doc : PyMuPDFのオブジェクト
            output_dir : 画像出力先ディレクトリ名
            save_num : 保存する枚数
        Returns:
            int : 保存した枚数
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
    return num_images

def __extract_images(doc: fitz.Document, page: fitz.Page) -> tuple[list[dict[str, np.ndarray | float]], list[fitz.Rect]]:
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

# 数値に変換できるものは int、できないものはそのまま
def __try_int(x: Any) -> int | str:
    try:
        return int(x)
    except ValueError:
        return x

def __found_missing_bbox(bboxes: list[fitz.Rect]) -> list[fitz.Rect]:
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
    logger.info(f"欠落している画像位置: {missing}")

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
