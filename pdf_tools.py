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

# 順位行のパターン: 行頭の数値 ( rank ) + 3文字コード + " - " ( 例: "1  GER - Germany ..." )
# 選手行は行頭が Position ( 4/3/2/1 ) だが "XXX - 国名" を伴わないため、このパターンには一致しない。
_STANDINGS_ROW_RE = re.compile(r'^\s*(\d+)\s+([A-Z]{3}) - ', re.MULTILINE)


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
            bool: 単独見出し行 "Final Standings" と列見出し行 ( Rank/Team/Players を含む )
                の両方を冒頭に持つ場合 True。
    """
    # 読み順を座標でソートして取得し、各行を strip する
    lines = [line.strip() for line in page.get_text(sort=True).split("\n")]
    head = lines[:14]  # 見出しはページ冒頭付近。多言語ページでも 14 行以内に収まる
    has_fs_header = any(line == "Final Standings" for line in head)
    has_column_header = any(
        ("Rank" in line and "Team" in line and "Players" in line) for line in head
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
        rank = int(m.group(1))
        team = m.group(2)  # 既に3文字コード
        results.append((rank, team))
    return results


def extract_venue(page: fitz.Page) -> tuple[str | None, str | None]:
    """
        Final Standings ページから会場情報 ( 所在地, 会場名 ) を抽出する。

        会場情報はどの順位ページにも同一のものが載るため、先頭の順位ページから
        1回だけ呼び出せばよい。書式の揺れに応じて所在地と会場名を分離する
        ( 詳細は docs/event_metadata_design.md 5.2 )。

        Args:
            page: Final Standings の1ページ。

        Returns:
            tuple[str | None, str | None]: ( location, venue )。
                - 空白で2分割できる: ( 所在地, 会場名 )
                - " - " 区切り: ( ハイフン前, ハイフン後 )
                - 会場名が先頭で分離困難: ( 会場行全体, None )
                - 会場行が無い: ( None, None )
    """
    lines = [line.strip() for line in page.get_text(sort=True).split("\n")]
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
