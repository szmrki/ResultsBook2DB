"""
make_golden_db.py: リファクタリング検証用の「正解 DB」を作るスクリプト

GUI と同じ Worker ( worker.py ) を、スレッドを起動せずに直接呼び出して
PDF を1件ずつ別々の SQLite に取り込む。リファクタリングの前後でこのスクリプトを
実行し、出力先ディレクトリ同士を tools/compare_db.py で比較することで、
「同じ PDF から同じ DB ができる」ことを確かめる ( issue #22 ) 。

complete_model/ に大会ごとのファインチューニング済みモデル ( {大会名}.pt ) が
ある大会では学習がスキップされるため、何度実行しても同じ結果になる。
モデルが無い大会を指定すると学習が走り、結果が毎回変わりうるので警告を出す。

使い方:
    # 代表的な PDF ( PRESET ) をまとめて取り込む
    uv run python tools/make_golden_db.py --out db/golden/before

    # PDF を個別に指定する ( MD は --md を付ける )
    uv run python tools/make_golden_db.py --out db/golden/before --pdf C:/ResultsBook/data_4p/WJCC2022_ResultsBook_Men.pdf
    uv run python tools/make_golden_db.py --out db/golden/before --md --pdf C:/ResultsBook/data_md/WMDCC2016_ResultsBook.pdf
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# tools/ から実行してもリポジトリ直下のモジュール ( worker.py など ) を import できるようにする
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from PySide6.QtCore import QCoreApplication  # noqa: E402

from create_db import set_tables  # noqa: E402
from worker import Worker  # noqa: E402

logger = logging.getLogger(__name__)

# Results Book の置き場所
PDF_ROOT = Path("C:/ResultsBook")

# 代表的な PDF の組み合わせ ( (相対パス, MD かどうか) ) 。
# いずれも complete_model/ に学習済みモデルがあり、学習なしで再現できるものを選んでいる。
PRESET: list[tuple[str, bool]] = [
    # 4人制
    ("data_4p/WJCC2022_ResultsBook_Men.pdf", False),   # 2022年の古い描画スタイル ( 小さく速い )
    ("data_4p/WJCC2026_ResultsBook_Men.pdf", False),   # JPEG の図 ( issue #20 で座標系の回転が見つかった大会 )
    ("data_4p/WMCC2024_ResultsBook.pdf", False),       # 世界選手権 ( issue #20 でゴースト石が多かった大会 )
    ("data_4p/OWG2026_ResultsBook_men.pdf", False),    # 五輪 ( 順位・会場ページの形式が異なる )
    ("data_4p/ECC2024_ResultsBook_Women_A-Division.pdf", False),  # 欧州選手権
    # MD ( 混合ダブルス )
    ("data_md/WMDCC2016_ResultsBook.pdf", True),       # 旧MD形式 ( 2016-2018 )
    ("data_md/WMDCC2026_ResultsBook.pdf", True),       # 新MD形式 ( 事前配置石あり )
    ("data_md/OWG2026_ResultsBook_MD.pdf", True),      # 五輪MD
]


def predict_event_name(filename: str) -> str:
    """ファイル名から大会名を推測する。

    main.py の MainWindow.predict_event_name と同じ規則 ( 大文字略称 + 年度 + Men/Women ) 。
    GUI クラスのメソッドで import できないため、issue #22 で関数として切り出すまでは複製して使う。

    Args:
        filename: PDF のファイル名 ( 例: "WJCC2022_ResultsBook_Men.pdf" )

    Returns:
        str: 大会名 ( 例: "WJCC2022Men" )
    """
    text = filename.split('_')[0].upper()
    # "women" は "men" を含むため、先に women を判定する
    if "women" in filename.lower():
        text += "Women"
    elif "men" in filename.lower():
        text += "Men"
    return text


def make_one(pdf_path: Path, is_md: bool, out_dir: Path) -> Path:
    """PDF を1件、空の SQLite に取り込む。

    出力先に同名の DB がある場合は削除して作り直す ( 毎回まっさらな状態から作るため ) 。

    Args:
        pdf_path: 取り込む PDF のパス
        is_md: MD ( 混合ダブルス ) の PDF なら True
        out_dir: DB の出力先ディレクトリ

    Returns:
        Path: 作成した DB のパス
    """
    event_name = predict_event_name(pdf_path.name)
    db_path = out_dir / f"{event_name}.db"

    # 学習済みモデルが無いと学習が走り、結果が毎回変わりうる
    if not (REPO_ROOT / "complete_model" / f"{event_name}.pt").exists():
        logger.warning(f"{event_name}: complete_model/{event_name}.pt が無いため学習が実行されます ( 結果が再現しない可能性があります )")

    if db_path.exists():
        db_path.unlink()
    set_tables(db_path, is_md)

    # GUI と同じ Worker を使う。start() ( 別スレッドでの実行 ) ではなく run() を直接呼ぶことで、
    # このスクリプトのスレッドで同期的に処理させる。シグナルは接続先が無いので何も起きない。
    worker = Worker([{"path": pdf_path, "event_name": event_name}], db_path, is_md=is_md)
    worker.run()
    return db_path


def main() -> None:
    """コマンドライン引数を解釈し、指定された PDF ( 省略時は PRESET ) を取り込む。"""
    parser = argparse.ArgumentParser(description="リファクタリング検証用の正解 DB を作る")
    parser.add_argument("--out", required=True, type=Path, help="DB の出力先ディレクトリ ( 例: db/golden/before )")
    parser.add_argument("--pdf", nargs="+", type=Path, help="取り込む PDF ( 省略時は PRESET を使う )")
    parser.add_argument("--md", action="store_true", help="--pdf で指定した PDF が MD の場合に付ける")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # 学習済みモデルや runs/ などは相対パスで扱われるため、リポジトリ直下を作業ディレクトリにする
    os.chdir(REPO_ROOT)

    # QThread ( Worker ) を使うために Qt のアプリケーションオブジェクトを1つ作っておく
    _app = QCoreApplication.instance() or QCoreApplication(sys.argv)

    if args.pdf:
        targets = [(p, args.md) for p in args.pdf]
    else:
        targets = [(PDF_ROOT / rel, is_md) for rel, is_md in PRESET]

    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path, is_md in targets:
        if not pdf_path.exists():
            logger.error(f"PDF が見つかりません: {pdf_path}")
            continue
        db_path = make_one(pdf_path, is_md, out_dir)
        logger.info(f"作成: {db_path}")


if __name__ == "__main__":
    main()
