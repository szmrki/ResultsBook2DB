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
                                                list[dict[str, str | int | None]],
                                                np.ndarray | None]:
    """
        ページからショットバイショット画像を抽出するメソッド
        Args:
            doc : PyMuPDFのオブジェクト
            page : PyMuPDFのページオブジェクト
            model : ストーン検出モデル
            is_md : MD or 4人制
    Returns:
        tuple[np.ndarray, list[dict[str, str | int | None]], np.ndarray | None]: 
            ストーン座標の配列 (num_shots x 16 x 6) 、ショット情報のリスト、
            Prepositioned stoneの座標配列 ((16, 6))。
            Prepositioned stoneが存在しない場合は None。
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

    # MD版: Prepositioned stone画像を取り出してYOLO推論し、座標を取得する
    pre_stones = None
    if is_md and "prepositioned stones" in [t.lower() for t in text]:
        pre_entry = shotbyshot_list.pop(0)  # 先頭画像を取り出す (削除ではなく座標抽出用)
        pre_result = get_stones_pos(
            [pre_entry["img"]], model, [pre_entry.get("is_negated", False)]
        )
        pre_stones = pre_result[0]  # (16, 6) の numpy配列
            
    imgs = [entry["img"] for entry in shotbyshot_list]
    is_negated_list = [entry.get("is_negated", False) for entry in shotbyshot_list]
    stones_end_list = get_stones_pos(imgs, model, is_negated_list)
    stones_end = np.array(stones_end_list)  #(num_shots, 16, 6)

    return stones_end, shot_info_list, pre_stones

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
    # 得点表のテーブルを取得（'*' を含むテーブルを優先）
    table = None
    for t in tabs:
        extracted = t.extract()
        if any('*' in row for row in extracted):
            table = extracted
            break

    # フォールバック: '*' が検出されなかった場合
    # (pdfplumberがLSFE列を認識できないPDFでは、得点表から '*' 列ごと欠落する)
    if table is None:
        for t in tabs:
            extracted = t.extract()
            if len(extracted) == 2 and len(extracted[0]) >= 10 and all(
                (cell or '').strip() in ('', 'X', '*') or (cell or '').strip().lstrip('-').isdigit()
                for row in extracted for cell in row
            ):
                table = extracted
                break
        if table is not None:
            # テキストからLSFE情報を補完して先頭列に追加
            lsfe_red, lsfe_yellow = __extract_lsfe_from_text(text, team_red, team_yellow)
            table[0].insert(0, lsfe_red)
            table[1].insert(0, lsfe_yellow)
            logger.info(f"LSFE column reconstructed from text: red='{lsfe_red}', yellow='{lsfe_yellow}'")

    if table is None:
        logger.warning("Score table not found on Game Results page.")
        # 得点表が見つからない場合も、列構成（LSFE, 1-10, Total）を用意する
        cols = ["team", "LSFE"] + [str(i) for i in range(1, 11)] + ["Total"]
        data = [
            [team_red] + [None] * (len(cols) - 1),
            [team_yellow] + [None] * (len(cols) - 1)
        ]
        df = pd.DataFrame(data, columns=cols)
        if is_md:
            return df, power_play_ends
        return df


    n_cols = len(table[0])
    columns = ["LSFE"] + [str(i) for i in range(1, n_cols-1)] + ["Total"]
    df = pd.DataFrame([[__try_int(cell) for cell in row] for row in table], columns=columns)
    df.insert(0, "team", [team_red, team_yellow])

    logger.debug(f"Successfully extracted game result:\n{df}")
    
    if is_md:
        return df, power_play_ends
    return df

def __extract_lsfe_from_text(text: str, team_red: str | None, team_yellow: str | None) -> tuple[str, str]:
    """
        ページテキストから各チームのLSFE値('*' または '')を取得する
        Args:
            text : ページのテキスト
            team_red : 赤チーム名 (例: 'CAN - Canada')
            team_yellow : 黄チーム名
        Returns:
            tuple[str, str] : (赤のLSFE値, 黄のLSFE値)
    """
    def find_lsfe(team: str | None) -> str:
        if not team:
            return ''
        for line in text.split('\n'):
            if team in line:
                after = line.split(team, 1)[1]
                # team名が途中で切れている場合 (例: "USA - United" と抽出され、本来は
                # "USA - United States of America") に備え、スコア開始位置まで読み進める
                for tok in after.split():
                    if tok == '*':
                        return '*'
                    if tok == 'X' or tok.lstrip('-').isdigit():
                        return ''
                return ''
        return ''
    return find_lsfe(team_red), find_lsfe(team_yellow)

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
        # --- パターン A：回転あり (4要素) ---
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

        # --- パターン B：回転なし (3要素) ---
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
    # ページ内の全画像情報を取得 (整数の XREF)
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
        xref = img[0]

        # 画像の中身（pixmap）は同一xrefの全配置で共通なので一度だけデコードする
        pix = fitz.Pixmap(doc, xref)
        img_cv = __pixmap2cv2(pix)
        is_negated = __black_more_than_white(img_cv)
        if is_negated:
            img_cv = 255 - img_cv  #反転

        # 元PDFが同一画像を複数箇所に配置している (盤面が同一のショットで重複排除される) 場合、
        # get_image_bbox は1配置しか返さず取りこぼす。get_image_rects で全配置を取得する。
        for bbox in page.get_image_rects(xref):
            x0, y0 = bbox.x0, bbox.y0  # 画像左上の座標
            bboxes.append(bbox)
            shotbyshot_list.append({
                "img": img_cv.copy(),  # 配置ごとに独立した配列を持たせる
                "x": x0,
                "y": y0,
                "is_negated": is_negated,
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

    # 幅・高さの推定 (最も安定)
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
            # fallback (近い行の画像サイズ)
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
            bool : 黒 > 白ならTrue、そうでなければFalse
    """
    # 白ピクセルの判定：各ピクセルの(R,G,B)がすべて255であるか
    white_count = np.sum(np.all(image_array == 255, axis=-1))
    # 黒ピクセルの判定：各ピクセルの(R,G,B)がすべて0であるか
    black_count = np.sum(np.all(image_array == 0, axis=-1))
    
    return black_count > white_count

def extract_lsd_from_text(text: str) -> list[dict[str, str | float | None]]:
    """
    Game ResultsページのテキストからLSD情報を抽出する
    """
    results = []
    # 数値+cm を抽出する正規表現
    lsd_value_pattern = re.compile(r'(\d+\.?\d*)\s*cm')

    lines = text.split('\n')
    lsd_section = False
    lsd_lines = []
    for line in lines:
        if "Last Stone Draw Distance" in line:
            lsd_section = True
            continue
        if lsd_section:
            if line.strip().startswith("Total"):
                lsd_section = False
                continue
            if "cm" in line:
                lsd_lines.append(line)
            else:
                lsd_section = False

    for line in lsd_lines:
        matches = list(lsd_value_pattern.finditer(line))
        if len(matches) >= 2:
            player_red = line[:matches[0].start()].strip()
            lsd_red = float(matches[0].group(1))
            player_yellow = line[matches[0].end():matches[1].start()].strip()
            lsd_yellow = float(matches[1].group(1))

            results.append({
                "player_red": player_red,
                "player_yellow": player_yellow,
                "lsd_red": lsd_red,
                "lsd_yellow": lsd_yellow,
            })
        elif len(matches) == 1:
            player_red = line[:matches[0].start()].strip()
            lsd_red = float(matches[0].group(1))
            results.append({
                "player_red": player_red,
                "player_yellow": None,
                "lsd_red": lsd_red,
                "lsd_yellow": None,
            })

    return results


def extract_year_and_category(game: str, is_md: bool) -> tuple[int | None, str | None]:
    """
    大会名文字列から西暦とカテゴリを抽出する。

    Args:
        game: 大会名文字列 (例: "WWCC2024")
        is_md: 混合ダブルス（MD）かどうか

    Returns:
        (year, category) のタプル。取得できない場合は None。
    """
    # 大会名(game)から西暦(year)を抽出
    year_match = re.search(r'\d{4}', game)
    year = int(year_match.group()) if year_match else None

    # カテゴリの特定
    category = None
    if is_md:
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


# ============================================================================
# 大会メタデータ ( 最終順位・会場 ) の抽出
#   Final Standings ページ ( 五輪以外の世界選手権系フォーマット ) を対象とする。
#   詳細な設計は docs/event_metadata_design.md を参照。
# ============================================================================

# 順位行のパターン: 行頭の順位 + 3文字コード + " - " ( 例: "1  GER - Germany ..." )
# 順位は数値 ( "1" ) のほか、上位3位が Gold / Silver / Bronze と表記される
# フォーマット ( 古い MD の WMDCC 等 ) があるためメダル語も許容する。
# 直後に "3文字コード + ' - '" を要求するため、"Gold Medal" 等の選手行語句には誤爆しない。
# 選手行は行頭が Position ( 4/3/2/1 ) だが "XXX - 国名" を伴わないため、このパターンには一致しない。
_STANDINGS_ROW_RE = re.compile(r'^\s*(\d+|Gold|Silver|Bronze)\s+([A-Z]{3}) - ', re.MULTILINE)

# メダル語表記を数値順位へ正規化する ( Gold=1, Silver=2, Bronze=3 )
_MEDAL_TO_RANK = {"Gold": 1, "Silver": 2, "Bronze": 3}


def is_standings_page(page: fitz.Page) -> bool:
    """
        指定ページが最終順位表 ( Final Standings ) のページかどうかを判定する。

        単純な "Final Standings" の包含判定では Competition Summary 等の
        後続ページも誤検出するため、見出し行での厳密判定を行う。
        判定は sort=True で取得したテキストの各行を strip してから行う
        ( 見出し行は先頭に大量の空白が付くため )。

        Args:
            page: PyMuPDF のページオブジェクト。

        Returns:
            bool: 単独見出し行 "Final Standings" と列見出し行 ( Rank/Team/Player を含む )
                の両方を冒頭に持つ場合 True。
    """
    # 読み順を座標でソートして取得し、各行を strip する
    lines = [line.strip() for line in page.get_text(sort=True).split("\n")]
    head = lines[:14]  # 見出しはページ冒頭付近。多言語ページでも 14 行以内に収まる
    has_fs_header = any(line == "Final Standings" for line in head)
    # 列見出しは 4人制 = "Players"、MD = "Female Player / Male Player" と表記が分かれるため、
    # 複数形の "s" を含めず "Player" で両対応する ( 古い MD の検出漏れ対策 )。
    has_column_header = any(
        ("Rank" in line and "Team" in line and "Player" in line) for line in head
    )
    return has_fs_header and has_column_header


def extract_standings(page: fitz.Page) -> list[tuple[int, str]]:
    """
        Final Standings ページから ( 順位, チームコード ) のリストを抽出する。

        順位表は複数ページにわたる場合があるため、本関数は1ページ分のみを処理する。
        呼び出し側で順位ページごとの結果を連結すること。

        Args:
            page: Final Standings の1ページ ( is_standings_page が True のページ )。

        Returns:
            list[tuple[int, str]]: ( rank, team ) のリスト。team は3文字コード。
                順位行が無ければ空リスト。
    """
    text = page.get_text(sort=True)
    results: list[tuple[int, str]] = []
    for m in _STANDINGS_ROW_RE.finditer(text):
        raw_rank = m.group(1)
        # 数値ならそのまま、メダル語 ( Gold/Silver/Bronze ) なら 1/2/3 に正規化する
        rank = _MEDAL_TO_RANK[raw_rank] if raw_rank in _MEDAL_TO_RANK else int(raw_rank)
        team = m.group(2)  # 既に3文字コード
        results.append((rank, team))
    return results


def extract_venue(page: fitz.Page, game: str = "") -> tuple[str | None, str | None]:
    """
        Final Standings ページから会場情報 ( 所在地, 会場名 ) を抽出する。

        会場情報はどの順位ページにも同一のものが載るため、先頭の順位ページから
        1回だけ呼び出せばよい。書式の揺れに応じて所在地と会場名を分離する
        ( 詳細は docs/event_metadata_design.md 5.2 )。

        五輪 ( OWG ) は他フォーマットと行構造が異なる ( 会場名のみで所在地が無く、
        カンマ区切りの会場行を持たない ) ため、大会名に "OWG" を含む場合は専用の
        分岐で会場名のみを抽出する ( 判定材料はファイル名由来の大会名 game )。

        Args:
            page: Final Standings の1ページ。
            game: 大会名 ( ファイル名由来 )。"OWG" を含む場合に五輪専用ロジックへ分岐する。

        Returns:
            tuple[str | None, str | None]: ( location, venue )。
                - OWG: ( None, 会場名 )。所在地 ( 都市名 ) はテキストに載らないため None
                - 空白で2分割できる: ( 所在地, 会場名 )
                - " - " 区切り: ( ハイフン前, ハイフン後 )
                - 会場名が先頭で分離困難: ( 会場行全体, None )
                - 会場行が無い: ( None, None )
    """
    lines = [line.strip() for line in page.get_text(sort=True).split("\n")]

    if "OWG" in game:
        # 五輪 ( OWG ): 順位ページ行0の先頭要素が会場名 ( 例: "Cortina Curling Olympic Stadium" )。
        # 行0は "会場名   ( 大量の空白 )   競技名/言語対訳" の形で、2連続以上の空白で分割した
        # 先頭要素だけが会場名。所在地 ( 都市名 ) はテキストに存在しないため location は None とする。
        line0 = lines[0] if lines else ""
        venue = re.split(r"\s{2,}", line0)[0].strip() if line0 else ""
        return None, (venue or None)

    # "Final Standings" 見出し行より前で、カンマを含み Championship を含まない行を会場行とみなす
    venue_line = None
    for line in lines:
        if line == "Final Standings":
            break
        if "," in line and "Championship" not in line:
            venue_line = line
            break

    if not venue_line:
        # 会場行が検出できない ( 所在地が無い多言語ページ等 )
        return None, None

    # 2連続以上の空白で分割を試みる ( 所在地と会場名が空白で隔てられているケース )
    parts = [p.strip() for p in re.split(r"\s{2,}", venue_line) if p.strip()]
    if len(parts) >= 2:
        # パターン①: 前半=所在地, 後半=会場名
        return parts[0], parts[1]

    # 1要素のみ ( 空白で分割できない )
    if " - " in venue_line:
        # パターン②a: " - " の前が所在地, 後ろが会場名
        location, venue = venue_line.split(" - ", 1)
        return location.strip(), venue.strip()

    # パターン②b: 会場名が先頭に紛れ分離困難 → 全体を location に生保存し venue は None
    return venue_line, None


# ============================================================================
# 選手ロースター ( rosters ) の抽出 ( 4人制のみ )
#   Final Standings ページの選手行から「チーム・選手名・ポジション・役割」を抽出する。
#   4人制の選手行は末尾に Position-Function 記号 ( "4 S" / "3 V" / "2" / "1" / "A" / "C" 等 ) を持つ。
#   MD は記載フォーマットが異なる ( Gender 列 ) ため、本関数は 4人制専用。MD 対応は別ステップ。
# ============================================================================

# 順位行の先頭部分: 行頭順位 ( 数値 or メダル語 ) + 3文字コード + " - " + 国名。
# standings と同じ判定基準だが、ここでは「順位行かどうか」の判別と、行から
# チームコードを取り出す目的で使う ( 国名以降には選手名と Position-Function が続く )。
_ROSTER_RANK_HEAD_RE = re.compile(r'^\s*(?:\d+|Gold|Silver|Bronze)\s+([A-Z]{3}) - \S')

# Position-Function 記号 ( 行末 )。
#   - 投球順のみ: "1" 〜 "4"
#   - 投球順 + 役割: "4 S" ( Skip ) / "3 V" ( Vice-skip )。順序は <数字> <英字>。
#   - 役割のみ: "A" ( Alternate ) / "C" ( Coach )
# 選手名にも空白が含まれる ( 例: "PETERSSON Haavard Vad" ) ため、
# 「行末に固定された Position-Function 記号」だけを取り出す形でアンカーする。
_POS_FUNC_TAIL_RE = re.compile(r'\s+([1-4])(?:\s+([SV]))?\s*$|\s+([AC])\s*$')

# 降格マーカー ( 行末 )。A-Division 大会の降格圏チームのスキップ行 ( 順位行 ) には
# Position-Function の後ろに ">B" ( B-Division へ降格 ) が付く ( 例: "... 4 S   >B" )。
# このマーカーを先に剥がさないと Position-Function を行末アンカーで取り出せず、
# スキップ行を丸ごと取りこぼす。">" + 英字1文字以上の塊を行末から除去する。
_RELEGATION_TAIL_RE = re.compile(r'\s+>[A-Za-z]+\s*$')


def extract_rosters(page: fitz.Page) -> list[dict[str, str | int | None]]:
    """
        Final Standings ページ ( 4人制 ) から選手ロースターを抽出する。

        1チームは「順位行 ( 先頭選手を含む ) + 継続行 ( 2人目以降 )」の複数行で構成される。
        順位行が現れるたびに team を切り替え、以降の選手行を同じ team に紐づける。
        各選手行の末尾 Position-Function 記号から position / is_skip / is_vice / role を解釈する。

        Args:
            page: Final Standings の1ページ ( is_standings_page が True のページ )。

        Returns:
            list[dict[str, str | int | None]]:
                各選手の情報辞書のリスト。キーは
                team / player_name / role / position / is_skip / is_vice。
                欠員行 ( 選手名が "-" ) はスキップする。
    """
    lines = page.get_text(sort=True).split("\n")
    rosters: list[dict[str, str | int | None]] = []
    current_team: str | None = None

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            continue

        # 順位行なら team を更新する。順位行は先頭選手も同じ行に含むため、
        # ここで team を確定したうえで、行から順位・チーム部分を取り除いて選手部分を得る。
        head = _ROSTER_RANK_HEAD_RE.match(line)
        if head:
            current_team = head.group(1)
            # "<順位>  <CODE> - <国名>" の部分を取り除き、選手名 + Position-Function を残す。
            # 国名は可変長 ( 単語数不定 ) だが、選手名は必ず大文字連続の姓で始まるため、
            # " - " 以降を1回だけ空白分割で読み飛ばすのではなく、末尾の Position-Function を
            # 起点に切り出す ( 下の共通処理に委ねる )。ここでは行全体を選手行として扱う。
            player_segment = line
        else:
            # 継続行 ( 2人目以降 )。team がまだ無い ( 見出し前の行など ) 場合は無視する。
            if current_team is None:
                continue
            player_segment = line

        parsed = __parse_roster_player(player_segment, is_rank_line=bool(head))
        if parsed is None:
            # 順位行 ( 各チームの先頭選手行 ) が解釈できないのは強い異常シグナル。
            # Position-Function 記号は 1/2/3/4・A・C・S・V の閉じた集合のため、
            # ここに到達するのは未知の記号・書式変化・抽出崩れが疑われるケース。
            # 記号集合が破綻していないかを検知できるよう警告ログを残す ( 挿入はスキップ )。
            if head:
                logger.warning(f"Roster rank-line could not be parsed (team={current_team}): {line.strip()!r}")
            continue
        parsed["team"] = current_team
        rosters.append(parsed)

    return rosters


def __parse_roster_player(segment: str, is_rank_line: bool) -> dict[str, str | int | None] | None:
    """
        選手行1行から選手名と Position-Function を解釈する ( 4人制 )。

        Args:
            segment: 選手1人分の行。順位行の場合は "<順位> <CODE> - <国名> <選手名> <記号>"、
                継続行の場合は "<選手名> <記号>" ( いずれも前後空白は許容 )。
            is_rank_line: 順位行 ( 先頭に順位+チームを含む ) かどうか。
                True の場合は選手名の前にある順位・チーム部分を取り除く。

        Returns:
            dict[str, str | int | None] | None:
                team を除く選手情報 ( player_name / raw_code / position / is_skip / is_vice / role )。
                Position-Function が読めない行や欠員行 ( 選手名 "-" ) は None。
    """
    # 降格圏スキップ行の行末マーカー ">B" を先に剥がす ( 付いていなければ無変化 )。
    # これを残すと Position-Function を行末アンカーで取り出せない。
    segment = _RELEGATION_TAIL_RE.sub("", segment)

    # 末尾の Position-Function 記号を取り出す。
    m = _POS_FUNC_TAIL_RE.search(segment)
    if not m:
        # 末尾に記号が無い行 ( 見出し・注記など ) はロースター行ではない。
        return None

    # 記号より前が「選手名 ( 順位行では順位+チーム+選手名 )」。
    name_part = segment[:m.start()].strip()

    if is_rank_line:
        # 順位行は "<順位>  <CODE> - <国名>  <選手名>" の形。" - " の後ろ ( 国名+選手名 ) から
        # 国名を読み飛ばして選手名だけを残す。国名・選手名とも空白を含みうるため、
        # 選手名は「最後の大文字始まり姓 + 名」を安定して切り出すのが難しい。
        # ここでは実データ ( "1  SCO - Scotland   MOUAT Bruce" ) の構造を用い、
        # 2連続以上の空白で「<順位 CODE - 国名>」と「<選手名>」が隔てられている点を利用する。
        parts = re.split(r'\s{2,}', name_part)
        # 末尾要素が選手名。ソート済みテキストでは "1  SCO - Scotland   MOUAT Bruce" のように
        # 順位+チーム部と選手名が2連続空白で分かれる ( 実データ全ファイルで確認 )。
        player_name = parts[-1].strip() if parts else name_part
    else:
        player_name = name_part

    # 欠員行 ( 選手名が "-" ) は選手として登録しない。
    if not player_name or player_name == "-":
        return None

    # --- Position-Function 記号を解釈する ---
    # マッチは2系統: (group1,group2) = <数字>,<S/V?> または (group3) = A/C
    position: int | None = None
    is_skip = 0
    is_vice = 0
    role: str

    if m.group(1) is not None:
        # 投球順あり ( 1-4 )。role は player。S/V があれば skip/vice フラグを立てる。
        position = int(m.group(1))
        func = m.group(2)  # "S" / "V" / None
        if func == "S":
            is_skip = 1
        elif func == "V":
            is_vice = 1
        role = "player"
    else:
        # 役割のみ ( A = Alternate / C = Coach )。
        func = m.group(3)
        if func == "A":
            # 補欠 ( Alternate ) も氷上でプレーする選手なので role は player とし、
            # 投球順は「フィフス」の呼称に合わせて position=5 で表す ( 正選手 1-4 の続き )。
            role = "player"
            position = 5
        else:
            # コーチ ( C ) は氷上でプレーしないため role=coach、投球順は持たない ( NULL )。
            role = "coach"

    return {
        "player_name": player_name,
        "role": role,
        "position": position,
        "is_skip": is_skip,
        "is_vice": is_vice,
    }


# ============================================================================
# 選手ロースター ( rosters ) の抽出 ( MD = 混合ダブルス )
#   MD の Final Standings ページは4人制と異なり Position-Function を持たない。
#   記載フォーマットは2系統 ( 詳細は docs/event_metadata_design.md 9節 ):
#     - 新MD ( 2019, 2021-2026 + 五輪MD ): 1行1人で末尾付近に Gender ( F/M/C ) を持つ。
#     - 旧MD ( 2016-2018 ): 1行に女子・男子2人を並記 ( 役割記号なし )。
#   列見出し行で系統を判定し、対応するパーサに振り分ける。
# ============================================================================

# 新MD の Gender トークン: 名前の後に現れる単独の1文字 ( F/M/C )。
# 女子行は Gender の後ろに Group Rank ( 例 "A4" ) や DSC ( 例 "22.22 cm" ) が続くため
# 行末アンカーは使えない。前後を空白で挟まれた単独の F/M/C を最初に1個だけ拾う。
# ( 名前の姓は2文字以上なので単独1文字トークンと衝突しない )
_MD_GENDER_TOKEN_RE = re.compile(r'(?<!\S)([FMC])(?!\S)')

# 旧MD で名前の後ろに続く非名前トークン ( Group Rank = 英字+数字、DSC = "... cm" )。
# 並記2列から女子名・男子名を切り出す際に、これらを名前候補から除外する。
_MD_GROUP_RANK_RE = re.compile(r'^[A-Z]\d+$')

# ページ末尾の凡例行 ( 例 "Team members are identified as follows: F = Female, M = Male, C = Coach" )
# は単独の F/M/C を含み選手行と誤認しうる。凡例では記号の直後が " = 意味" になっている点で
# 選手行 ( 記号の後ろは Group Rank/DSC か行末 ) と区別できるため、"F =" 形式を除外に使う。
_MD_LEGEND_LINE_RE = re.compile(r'\b[FMC]\s*=')

# 性別記号 ( PDF 記載の F/M ) を DB 保存用の単語へ展開する。
# 既存の color='red'/'yellow'・role='player'/'coach' と表記スタイルを揃えるため単語で持つ。
_MD_GENDER_WORD = {"F": "Female", "M": "Male"}


def extract_rosters_md(page: fitz.Page) -> list[dict[str, str | None]]:
    """
        Final Standings ページ ( MD ) から選手ロースターを抽出する。

        列見出し行を見て新MD ( Gender 列 ) か旧MD ( Female Player / Male Player 列 ) かを
        判定し、対応するパーサで選手行を解釈する。順位行が現れるたびに team を切り替える。

        Args:
            page: Final Standings の1ページ ( is_standings_page が True のページ )。

        Returns:
            list[dict[str, str | None]]:
                各選手の情報辞書のリスト。キーは team / player_name / role / gender。
                欠員行 ( 選手名が "-" ) はスキップする。
    """
    lines = page.get_text(sort=True).split("\n")
    # 列見出し行 ( Rank と Team を含む ) を探して系統を判定する。
    is_old_md = False
    for line in lines:
        s = line.strip()
        if "Rank" in s and "Team" in s:
            # 旧MD は "Female Player" と "Male Player" の2列見出しを持つ。
            is_old_md = ("Female Player" in s and "Male Player" in s)
            break

    rosters: list[dict[str, str | None]] = []
    current_team: str | None = None

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            continue

        head = _ROSTER_RANK_HEAD_RE.match(line)
        if head:
            current_team = head.group(1)
        elif current_team is None:
            # 見出し行より前の行は team が未確定なので無視する。
            continue

        if is_old_md:
            # 旧MD: 1行に女子・男子が並記される。順位行のみが選手を含む
            # ( 継続行は無い ) ため、順位行だけを処理する。
            if not head:
                continue
            parsed = __parse_roster_md_old(line)
            # 旧MD は女子・男子2人が基本。2人揃わない場合は空白崩れ等による
            # 抽出漏れの疑いがあるため警告ログを残す ( 取れた分は挿入する )。
            if len(parsed) != 2:
                logger.warning(
                    f"MD ( old ) roster row yielded {len(parsed)} name(s) (team={current_team}): {line.strip()!r}")
            for p in parsed:
                p["team"] = current_team
                rosters.append(p)
        else:
            # 新MD: 1行1人。順位行・継続行いずれも1人分の選手行として扱う。
            p = __parse_roster_md_new(line, is_rank_line=bool(head))
            if p is None:
                continue
            p["team"] = current_team
            rosters.append(p)

    return rosters


def __parse_roster_md_new(segment: str, is_rank_line: bool) -> dict[str, str | None] | None:
    """
        新MD の選手行1行から選手名と Gender を解釈する。

        Args:
            segment: 選手1人分の行。順位行では "<順位> <CODE> - <国名> <選手名> <F/M/C> ..."、
                継続行では "<選手名> <F/M/C>" ( Gender の後ろに Group Rank/DSC が続く場合あり )。
            is_rank_line: 順位行かどうか。True の場合は選手名の前の順位・チーム部分を取り除く。

        Returns:
            dict[str, str | None] | None:
                team を除く選手情報 ( player_name / role / gender )。
                Gender トークンが無い行や欠員行 ( 選手名 "-" ) は None。
    """
    # 凡例行 ( "F = Female, ..." ) は選手行ではないため除外する。
    if _MD_LEGEND_LINE_RE.search(segment):
        return None

    # スキップ行末の出場権マーカー ">OWG" 等を先に剥がす ( 付いていなければ無変化 )。
    segment = _RELEGATION_TAIL_RE.sub("", segment)

    # 名前の後にある単独の Gender トークン ( F/M/C ) を1個目だけ拾う。
    m = _MD_GENDER_TOKEN_RE.search(segment)
    if not m:
        return None

    name_part = segment[:m.start()].strip()

    if is_rank_line:
        # 順位行は "<順位>  <CODE> - <国名>  <選手名>" の形。2連続以上の空白で
        # 順位+チーム部と選手名が分かれるため、末尾要素を選手名とする。
        parts = re.split(r'\s{2,}', name_part)
        player_name = parts[-1].strip() if parts else name_part
    else:
        player_name = name_part

    # 欠員行 ( 選手名が "-" ) は登録しない。
    if not player_name or player_name == "-":
        return None

    gender_code = m.group(1)
    if gender_code == "C":
        # コーチは氷上でプレーせず性別記載も無い。
        return {"player_name": player_name, "role": "coach", "gender": None}
    # F ( 女子 ) / M ( 男子 ) の選手。記号は単語 ( Female/Male ) に展開して保存する。
    return {"player_name": player_name, "role": "player", "gender": _MD_GENDER_WORD[gender_code]}


def __parse_roster_md_old(segment: str) -> list[dict[str, str | None]]:
    """
        旧MD の順位行1行から女子・男子2人分の選手を解釈する。

        行は "<順位>  <CODE> - <国名>  <女子名>  <男子名>  [Group Rank]  [DSC]" の形。
        全フィールドが2連続以上の空白で区切られる。順位+チーム部を取り除いた残りのうち、
        Group Rank ( 英字+数字 ) ・DSC ( "... cm" ) ・メダル語を除いた先頭2要素を
        女子名・男子名とする。

        Args:
            segment: 順位行1行。

        Returns:
            list[dict[str, str | None]]:
                女子・男子の選手情報のリスト ( 最大2件、role=player )。
                欠員 ( 名前 "-" ) は除外する。取得できなければ空リスト。
    """
    # "<順位>  <CODE> - <国名>" を取り除く。順位行ヘッダは
    # 「行頭の順位 ( 数値/メダル語 ) + 3文字コード + ' - '」なので、その直後 ( 国名以降 ) を得る。
    m = re.match(r'^\s*(?:\d+|Gold|Silver|Bronze)\s+[A-Z]{3} - (.*)$', segment)
    if not m:
        return []
    rest = m.group(1)

    # 2連続以上の空白で分割し、国名 ( 先頭 ) ・Group Rank・DSC・メダル語を除いて名前候補を得る。
    tokens = [t.strip() for t in re.split(r'\s{2,}', rest) if t.strip()]
    # 先頭要素は国名 ( 例 "Russia" / "United States" )。これを捨てる。
    if tokens:
        tokens = tokens[1:]

    names = []
    for tok in tokens:
        if _MD_GROUP_RANK_RE.match(tok):
            continue  # Group Rank ( 例 "D4" )
        if "cm" in tok:
            continue  # DSC ( 例 "28.23 cm" )
        if tok in ("Gold Medal", "Silver Medal", "Bronze Medal"):
            continue  # メダル表記
        if tok == "-":
            continue  # 欠員
        names.append(tok)

    # 先頭2要素を女子名・男子名とする ( 実データ構造: 女子が先、男子が後 )。
    # gender は新MD と同じく単語 ( Female/Male ) で保存する。
    result: list[dict[str, str | None]] = []
    if len(names) >= 1:
        result.append({"player_name": names[0], "role": "player", "gender": _MD_GENDER_WORD["F"]})
    if len(names) >= 2:
        result.append({"player_name": names[1], "role": "player", "gender": _MD_GENDER_WORD["M"]})
    return result
