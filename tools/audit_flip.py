"""
audit_flip.py: 盤面の図 ( Shot by Shot ) の上下反転の判定を監査するスクリプト

Results Book の盤面の図には、ハウスが上にある図と下にある図 ( 逆さまの図 ) が混ざっている。
detection.needs_flip は逆さまの図を見分けて180°回転させるが、この判定を誤ると、
その図の石の座標がすべて180°回転した位置で記録されてしまう ( issue #24 ) 。

このスクリプトは、すべての PDF のすべての盤面の図について、
    - detection.needs_flip の判定 ( 実際に取り込みで使われるもの )
    - 判定とは別の手がかりから作った「正解」
を比べ、食い違う図を洗い出す。

正解は、次の2つの手がかりから作る。
    - ハウスの色: 彩度の高い画素 ( ハウスの色付きの円 ) の重心が下半分にあれば、逆さま
    - ホッグライン: 上端の付近に黒い横線があれば逆さま、下端の付近にあれば正しい向き
2つとも判定できて一致すればそれを正解とし、片方しか判定できなければその片方を正解とする。
2つが食い違う図と、どちらも判定できない図は「正解が不明」として別に数える。

図ごとの結果は次の4つに分かれる。
    ok        : 判定が正解と一致した
    MISS      : 判定が正解と食い違った ( 反転の取りこぼし、または誤った反転 )
    undecided : 判定できなかった ( needs_flip が None を返した )
    unclear   : 正解が不明で、判定の正しさを確かめられなかった
これとは別に、主の判定 ( 上下の白い画素の数 ) で決まらず、副の判定 ( ホッグライン ) に
回った図の数を by_hog として表示する。

使い方:
    # すべての PDF を監査する ( 結果は --out のディレクトリに audit.csv として保存される )
    uv run python tools/audit_flip.py --out scratch/flip_audit/after

    # PDF を個別に指定する
    uv run python tools/audit_flip.py --out scratch/flip_audit/after --pdf C:/ResultsBook/data_4p/WJCC2026_ResultsBook_Men.pdf

終了コード:
    0 = MISS も undecided も無い / 1 = MISS または undecided がある / 2 = 引数の誤り
"""
import argparse
import collections
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import fitz
import numpy as np

# tools/ から実行してもリポジトリ直下のモジュール ( detection.py など ) を import できるようにする
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import detection  # noqa: E402
import pdf_tools  # noqa: E402

# Results Book の置き場所
PDF_ROOT = Path("C:/ResultsBook")

# 図の結果の種類 ( 集計と表示の順番に使う )
STATUSES = ("ok", "MISS", "undecided", "unclear")

# 種類ごとに保存する図の最大数 ( 1つの PDF あたり ) 。目で見て確かめるための見本
MAX_SAVED_IMAGES = 3

# pdf_tools.__extract_images は名前が "__" で始まるため、from ... import では取り込めない。
# getattr で名前を文字列として渡せば取り出せる ( 取り込みと同じ方法で図を取り出すために使う )
extract_images = getattr(pdf_tools, "__extract_images")
# 主の判定 ( 上下の白い画素の数 ) だけを呼び、副の判定 ( ホッグライン ) に回った図を数えるために使う
flip_by_white = getattr(detection, "__flip_by_white")


def hog_signal(img: np.ndarray) -> bool | None:
    """ホッグライン ( 黒い横線 ) の位置から、逆さまの図かどうかを判定する。

    逆さまの図では、ホッグラインが上端の付近に来る。

    Args:
        img: 盤面の図 ( 高さ600 x 幅300 x 3色 ) の配列

    Returns:
        bool | None: 逆さまなら True、正しい向きなら False、判定できなければ None
    """
    # 各画素について「3色とも 80 未満 ( ほぼ黒 ) か」を調べ、行ごとにその割合を出す。
    # 線に石が少し重なっていても拾えるよう、行の8割がほぼ黒なら「黒い横線」とみなす
    dark_rows = (img[:, 1:detection.WIDTH].max(axis=2) < 80).mean(axis=1) > 0.8
    # ホッグラインは端から19〜20行目にある。探す範囲は detection の判定とそろえる
    # ( 範囲を広げると、図の枠線やバックラインなど別の線をホッグラインとみなすおそれがあるため )
    hog_from, hog_to = detection.FLIP_HOG_FROM, detection.FLIP_HOG_TO
    top = bool(dark_rows[hog_from:hog_to].any())      # 上端の付近に黒い横線があるか
    bottom = bool(dark_rows[-hog_to:-hog_from].any())  # 下端の付近に黒い横線があるか
    if top and not bottom:
        return True
    if bottom and not top:
        return False
    return None


def house_signal(img: np.ndarray) -> bool | None:
    """ハウスの色の位置から、逆さまの図かどうかを判定する。

    ハウスは色付きの大きな円なので、彩度の高い画素の重心がハウスの位置になる。
    逆さまの図では、ハウスが下半分に来る。

    Args:
        img: 盤面の図 ( 高さ600 x 幅300 x 3色 ) の配列

    Returns:
        bool | None: 逆さまなら True、正しい向きなら False、判定できなければ None
    """
    # HSV の S ( 彩度 ) が高い画素 = 色の付いた画素。上下25行は、盤面の外に出た石が
    # 並ぶ帯なので除く ( 石の色に引きずられないようにするため )
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ys = np.nonzero(hsv[25:-25, :, 1] > 40)[0]
    if len(ys) < 500:
        return None  # 色の付いた画素が少なすぎる ( ハウスが描かれていない ) 場合は判定しない
    # ys は 25行目からの相対位置なので、25 を足して元の行番号に戻す
    return bool(ys.mean() + 25 > img.shape[0] / 2)


def make_truth(house: bool | None, hog: bool | None) -> bool | None:
    """2つの手がかりから正解を作る。

    Args:
        house: ハウスの色による判定
        hog: ホッグラインによる判定

    Returns:
        bool | None: 逆さまなら True、正しい向きなら False、正解が不明なら None
    """
    if hog is None:
        return house  # ホッグラインで判定できなければ、ハウスの色に従う ( これも None なら不明 )
    if house is None:
        return hog
    return house if house == hog else None  # 2つが食い違う図は不明とする


def audit_pdf(pdf_path: Path, out_dir: Path) -> tuple[collections.Counter, list[list]]:
    """PDF 1件のすべての盤面の図を監査する。

    Args:
        pdf_path: 監査する PDF のパス
        out_dir: 見本の図の保存先ディレクトリ

    Returns:
        tuple[collections.Counter, list[list]]:
            結果の種類ごとの枚数と、ok 以外の図の一覧 ( audit.csv の行 )
    """
    counts: collections.Counter = collections.Counter()
    rows: list[list] = []
    doc = fitz.open(pdf_path)
    for page_index in range(doc.page_count):
        page = doc[page_index]
        if "Shot by Shot" not in page.get_text():
            continue
        entries, _ = extract_images(doc, page)
        for i, entry in enumerate(entries):
            img = entry["img"]
            judged = detection.needs_flip(img)
            house, hog = house_signal(img), hog_signal(img)
            truth = make_truth(house, hog)

            if judged is None:
                status = "undecided"
            elif truth is None:
                status = "unclear"
            elif judged == truth:
                status = "ok"
            else:
                status = "MISS"
            counts[status] += 1
            # 主の判定で決まらず、副の判定に回った図の数 ( 結果の種類とは別に数える )
            if flip_by_white(img) is None:
                counts["by_hog"] += 1
            if status == "ok":
                continue

            rows.append([pdf_path.name, page_index + 1, i, round(entry["x"], 1), round(entry["y"], 1),
                         entry["is_negated"], judged, house, hog, truth, status])
            if counts[status] <= MAX_SAVED_IMAGES:
                cv2.imwrite(str(out_dir / f"{status}_{pdf_path.stem}_p{page_index + 1}_{i}.png"), img)
    doc.close()
    return counts, rows


def main() -> None:
    """コマンドライン引数を解釈し、指定された PDF ( 省略時はすべての PDF ) を監査する。"""
    parser = argparse.ArgumentParser(description="盤面の図の上下反転の判定を監査する")
    parser.add_argument("--out", required=True, type=Path, help="結果 ( audit.csv と見本の図 ) の出力先ディレクトリ")
    parser.add_argument("--pdf", nargs="+", type=Path, help="監査する PDF ( 省略時は C:/ResultsBook/data_* 以下のすべて )")
    parser.add_argument("--jobs", type=int, default=4, help="同時に処理する PDF の数 ( 既定: 4 )")
    args = parser.parse_args()

    pdf_paths = args.pdf or sorted(PDF_ROOT.glob("data_*/*.pdf"))
    missing = [p for p in pdf_paths if not p.exists()]
    if not pdf_paths or missing:
        print(f"PDF が見つかりません: {missing or PDF_ROOT}")
        sys.exit(2)

    args.out.mkdir(parents=True, exist_ok=True)

    # PDF ごとに別のプロセスで処理する ( 図の取り出しに時間がかかるため ) 。
    # map は渡した順に結果を返すので、表示と CSV の並びは毎回同じになる
    total: collections.Counter = collections.Counter()
    all_rows: list[list] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        results = executor.map(audit_pdf, pdf_paths, [args.out] * len(pdf_paths))
        for pdf_path, (counts, rows) in zip(pdf_paths, results):
            print(pdf_path.name, {s: counts[s] for s in (*STATUSES, "by_hog") if counts[s]}, flush=True)
            total += counts
            all_rows += rows

    with open(args.out / "audit.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pdf", "page", "idx", "x", "y", "negated", "judged", "house", "hog", "truth", "status"])
        writer.writerows(all_rows)

    # by_hog は結果の種類とは別の数え方なので、枚数の合計には含めない
    print("TOTAL", {"total": sum(total[s] for s in STATUSES), **{s: total[s] for s in (*STATUSES, "by_hog")}})
    sys.exit(1 if total["MISS"] or total["undecided"] else 0)


# Windows では別のプロセスがこのファイルを import し直すため、この条件が無いと main() が再び実行されてしまう
if __name__ == "__main__":
    main()
