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

    # Worker を使わず、解析結果を JSON ファイルに書き出して読み込み直してから DB に書き込む
    # ( 「PDF の解析」と「DB への書き込み」を別々に行う経路の確認用 )
    uv run python tools/make_golden_db.py --out db/golden/after_json --via-json
"""
import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path

# tools/ から実行してもリポジトリ直下のモジュール ( worker.py など ) を import できるようにする
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from PySide6.QtCore import QCoreApplication  # noqa: E402

from create_db import set_tables  # noqa: E402
from event_processing import (  # noqa: E402
    extract_event, load_event_result, postprocess_event, save_event_result, write_event,
)
from utils import predict_event_name  # noqa: E402
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


def make_one(pdf_path: Path, is_md: bool, out_dir: Path, via_json: bool = False) -> Path:
    """PDF を1件、空の SQLite に取り込む。

    出力先に同名の DB がある場合は削除して作り直す ( 毎回まっさらな状態から作るため ) 。

    Args:
        pdf_path: 取り込む PDF のパス
        is_md: MD ( 混合ダブルス ) の PDF なら True
        out_dir: DB の出力先ディレクトリ
        via_json: True なら Worker を使わず、解析結果を JSON ファイルに書き出して
            読み込み直してから DB に書き込む

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

    if via_json:
        ingest_via_json(pdf_path, event_name, is_md, db_path, out_dir / "json" / f"{event_name}.json")
        return db_path

    # GUI と同じ Worker を使う。start() ( 別スレッドでの実行 ) ではなく run() を直接呼ぶことで、
    # このスクリプトのスレッドで同期的に処理させる。シグナルは接続先が無いので何も起きない。
    worker = Worker([{"path": pdf_path, "event_name": event_name}], db_path, is_md=is_md)
    worker.run()
    return db_path


def ingest_via_json(pdf_path: Path, event_name: str, is_md: bool, db_path: Path, json_path: Path) -> None:
    """Worker を使わずに PDF を取り込む。解析結果はいったん JSON ファイルを経由させる。

    「解析の担当が PDF を解析して解析結果ファイルを作り、サーバがそれを読んで DB に書き込む」
    という分担 ( issue #21 ) と同じ経路を、1つのプロセスの中で再現する。

    Args:
        pdf_path: 取り込む PDF のパス
        event_name: 大会名
        is_md: MD ( 混合ダブルス ) の PDF なら True
        db_path: 書き込み先の DB ( テーブル作成済み ) のパス
        json_path: 解析結果ファイルの書き出し先
    """
    # 1. PDF → 解析結果 → JSON ファイル ( ここまでは DB に触れない )
    result = extract_event(pdf_path, event_name, is_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    save_event_result(result, json_path)

    # 2. JSON ファイル → 解析結果。書き出す前と同じ内容に戻ることを確かめる
    loaded = load_event_result(json_path)
    if loaded != result:
        logger.error(f"{event_name}: JSON を読み込み直した解析結果が、書き出す前と一致しません")

    # 3. 解析結果 → DB → 後処理
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        event_id, prepositioned_map = write_event(conn, loaded)
        postprocess_event(conn, event_id, is_md, prepositioned_map)
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    """コマンドライン引数を解釈し、指定された PDF ( 省略時は PRESET ) を取り込む。"""
    parser = argparse.ArgumentParser(description="リファクタリング検証用の正解 DB を作る")
    parser.add_argument("--out", required=True, type=Path, help="DB の出力先ディレクトリ ( 例: db/golden/before )")
    parser.add_argument("--pdf", nargs="+", type=Path, help="取り込む PDF ( 省略時は PRESET を使う )")
    parser.add_argument("--md", action="store_true", help="--pdf で指定した PDF が MD の場合に付ける")
    parser.add_argument("--via-json", action="store_true",
                        help="Worker を使わず、解析結果を JSON ファイル経由で DB に書き込む ( <out>/json/ に書き出す )")
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
        db_path = make_one(pdf_path, is_md, out_dir, args.via_json)
        logger.info(f"作成: {db_path}")


if __name__ == "__main__":
    main()
