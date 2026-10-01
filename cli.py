"""
cli.py: GUI を使わずに PDF を解析・取り込むためのコマンドライン入口

処理の本体は GUI と共通の event_processing モジュールにあり、ここではコマンドライン引数を
受け取って呼び出すだけ。「PDF の解析」と「DB への書き込み」を別々のコマンドに分けてあるので、
解析を GPU のあるマシンで行い、できた解析結果ファイルを別のマシンで DB に取り込める ( issue #21 ) 。

使い方:
    # 1. PDF を解析して、解析結果ファイル ( JSON ) を作る ( DB には触れない )
    #    大会名はファイル名から推測する。MD は --md を付ける
    uv run python cli.py extract C:/ResultsBook/data_4p/WJCC2026_ResultsBook_Men.pdf --out results
    uv run python cli.py extract C:/ResultsBook/data_md/WMDCC2026_ResultsBook.pdf --out results --md

    # 2. 解析結果ファイルを DB に取り込む ( DB が無ければ作る )
    uv run python cli.py ingest results/WJCC2026Men.json --db db/four.db
    uv run python cli.py ingest results/WMDCC2026.json --db db/md.db
"""
import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Callable

from create_db import set_tables
from event_processing import (
    event_exists, extract_event, load_event_result, postprocess_event, save_event_result, write_event,
)
from stone_matching import ensure_shot_order_column
from utils import predict_event_name

logger = logging.getLogger(__name__)


def make_progress_logger() -> Callable[[int, str], None]:
    """
        進捗をログに出すコールバックを作る
        同じメッセージで進捗率だけが細かく変わる通知は、10% 刻みに間引いて出力する
        Returns:
            Callable[[int, str], None] : progress_cb(percent, message) として渡せる関数
    """
    last: dict[str, object] = {"message": None, "step": None}

    def progress_cb(percent: int, message: str) -> None:
        # "Matching stones... (12/345)" のような件数付きの部分を除いた見出しで、同じ処理かを判断する
        head = message.split(" (")[0]
        step = percent // 10
        if head != last["message"] or step != last["step"]:
            logger.info(f"{percent:3d}% {message}")
            last["message"], last["step"] = head, step

    return progress_cb


def is_md_database(conn: sqlite3.Connection) -> bool:
    """
        DB が MD ( 混合ダブルス ) 用かどうかを判定する
        MD 用の DB は ends テーブルに is_power_play 列を持つ ( create_db.set_tables を参照 )
        Args:
            conn : SQLite の接続
        Returns:
            bool : MD 用の DB なら True
    """
    columns = [row[1] for row in conn.execute("PRAGMA table_info(ends)").fetchall()]
    return "is_power_play" in columns


def cmd_extract(args: argparse.Namespace) -> int:
    """
        extract コマンド: PDF を解析し、解析結果ファイルを書き出す
        Args:
            args : コマンドライン引数 ( pdf, out, md, event_name )
        Returns:
            int : 終了コード ( 0 = 成功 )
    """
    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        logger.error(f"PDF が見つかりません: {pdf_path}")
        return 1
    event_name = args.event_name or predict_event_name(pdf_path.name)
    if not event_name:
        logger.error(f"大会名を決められません。--event-name で指定してください: {pdf_path.name}")
        return 1

    result = extract_event(pdf_path, event_name, args.md, progress_cb=make_progress_logger())

    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / f"{event_name}.json"
    save_event_result(result, json_path)
    logger.info(f"解析結果ファイルを書き出しました: {json_path}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """
        ingest コマンド: 解析結果ファイルを DB に取り込む
        1ファイル ( = 1大会 ) ごとにコミットする。失敗した大会はロールバックし、次のファイルに進む。
        Args:
            args : コマンドライン引数 ( results, db )
        Returns:
            int : 終了コード ( 0 = すべて成功 / 1 = 1件以上の失敗 )
    """
    db_path: Path = args.db
    failed = 0
    conn: sqlite3.Connection | None = None
    try:
        for json_path in args.results:
            try:
                result = load_event_result(json_path)

                # 最初のファイルを読んだ時点で DB を開く ( DB が無ければ、その大会の種別で作る )
                if conn is None:
                    if not db_path.exists():
                        db_path.parent.mkdir(parents=True, exist_ok=True)
                        set_tables(db_path, result.is_md)
                        logger.info(f"DB を作成しました: {db_path} ( {'MD' if result.is_md else '4人制'} )")
                    conn = sqlite3.connect(db_path)
                    conn.execute("PRAGMA foreign_keys = ON;")
                    ensure_shot_order_column(conn)  # 古い DB には shot_order 列が無い場合がある
                    conn.commit()

                # 4人制と MD では DB のスキーマが違うため、種別の違う DB には取り込めない
                if is_md_database(conn) != result.is_md:
                    raise ValueError("DB と解析結果ファイルで、4人制 / MD の種別が一致しません")
                if event_exists(conn, result.name):
                    raise ValueError(f"Event Name '{result.name}' は既に使用されています")

                event_id, prepositioned_map = write_event(conn, result)
                updated = postprocess_event(conn, event_id, result.is_md, prepositioned_map,
                                            progress_cb=make_progress_logger())
                conn.commit()  # 書き込みと同定が揃ってから、1大会としてコミットする
                logger.info(f"取り込みました: {result.name} ( {updated} stones labeled )")
            except Exception as e:
                if conn is not None:
                    conn.rollback()
                failed += 1
                logger.error(f"{json_path}: {e}")
    finally:
        if conn is not None:
            conn.close()
    return 1 if failed else 0


def main() -> int:
    """
        コマンドライン引数を解釈し、対応するコマンドを実行する
        Returns:
            int : 終了コード
    """
    parser = argparse.ArgumentParser(description="Results Book ( PDF ) の解析と DB への取り込み")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="PDF を解析して解析結果ファイル ( JSON ) を作る")
    p_extract.add_argument("pdf", type=Path, help="解析する PDF")
    p_extract.add_argument("--out", type=Path, required=True, help="解析結果ファイルの出力先ディレクトリ")
    p_extract.add_argument("--md", action="store_true", help="MD ( 混合ダブルス ) の PDF の場合に付ける")
    p_extract.add_argument("--event-name", help="大会名 ( 省略時はファイル名から推測する )")
    p_extract.set_defaults(func=cmd_extract)

    p_ingest = sub.add_parser("ingest", help="解析結果ファイルを DB に取り込む")
    p_ingest.add_argument("results", nargs="+", type=Path, help="解析結果ファイル ( 複数指定可 )")
    p_ingest.add_argument("--db", type=Path, required=True, help="取り込み先の SQLite ( 無ければ作る )")
    p_ingest.set_defaults(func=cmd_ingest)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
