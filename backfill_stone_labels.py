"""
backfill_stone_labels.py: 既存DBへのストーン同定バックフィルスクリプト

既に生成済みの SQLite DB（db/*.db）に対して、ストーン同定（shot_order）を
後付けで実行するためのスタンドアロンスクリプト。

stones.shot_order カラムが無ければ自動で追加し（ALTER TABLE）、
DB内の全エンドを舐めて各ストーンの投球元を特定・書き込む。
ストーン同定は YOLO 推論を伴わないため非常に高速（DB全体でも数秒オーダー）で、
アルゴリズムを改善したときに何度でも再実行できる。

使い方:
    # 単一DBを処理
    uv run python backfill_stone_labels.py --db db/OWG2022.db

    # 複数DBをまとめて処理
    uv run python backfill_stone_labels.py --db db/OWG2022.db db/women2025.db

    # 未ラベルのエンドのみ処理（再実行時に既ラベル分をスキップして高速化）
    uv run python backfill_stone_labels.py --db db/OWG2022.db --only-unlabeled
"""
import argparse
import logging
import sqlite3
import sys
from pathlib import Path

# 同定ロジックは worker と共通のコアモジュールを再利用する
from stone_matching import label_database

# スクリプト単体実行時はINFOログを標準出力に流す
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def is_md_database(conn: sqlite3.Connection) -> bool:
    """DBがMD ( 混合ダブルス ) 用かどうかを判定する。

    MD用DBは ends テーブルに is_power_play 列を持つ ( create_db.set_tables を参照 ) ため、
    その列の有無で判別する。MDのストーン同定には Prepositioned stone の座標が必要だが、
    その座標はDBに永続化されていない ( worker 実行時にYOLO推論で得てメモリ上で使うだけ ) ため、
    バックフィルでは正しいMDマッチングを再構成できない。よってMD DBは処理対象から弾く。

    Args:
        conn: SQLite 接続。

    Returns:
        bool: ends テーブルに is_power_play 列があれば True ( = MD用DB )。
    """
    cur = conn.cursor()
    # PRAGMA table_info は (cid, name, type, notnull, dflt_value, pk) のタプルを返す
    cur.execute("PRAGMA table_info(ends)")
    columns = {row[1] for row in cur.fetchall()}
    return "is_power_play" in columns


def backfill_one(db_path: Path, only_unlabeled: bool) -> int:
    """単一のDBファイルにストーン同定を適用する。

    Args:
        db_path: 対象SQLiteファイルのパス。
        only_unlabeled: True なら未ラベルのエンドのみ処理する。

    Returns:
        int: 更新したストーン行数。MD用DBはスキップし 0 を返す。
    """
    # 外部キー制約を有効化して接続（既存の運用と揃える）
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        # MD用DBはバックフィルで正しく同定できないためスキップする ( 誤った shot_order の書き込みを防ぐ )
        if is_md_database(conn):
            logger.warning(
                f"  -> {db_path.name} は MD ( 混合ダブルス ) 用DBのため、バックフィルをスキップします "
                f"( Prepositioned stone 座標がDBに無く正しく同定できないため )。"
            )
            return 0
        # label_database 内で shot_order カラムの存在保証とコミットまで行われる
        updated = label_database(conn, only_unlabeled=only_unlabeled)
        return updated
    finally:
        conn.close()


def main() -> None:
    """コマンドライン引数を解釈し、指定された各DBにバックフィルを実行する。

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="既存DBにストーン同定(shot_order)を後付けする"
    )
    parser.add_argument(
        "--db", type=str, nargs="+", required=True,
        help="対象のSQLiteファイルパス（複数指定可）",
    )
    parser.add_argument(
        "--only-unlabeled", action="store_true",
        help="未ラベルのエンドのみ処理する（再実行時の高速化用）",
    )
    args = parser.parse_args()

    total_updated = 0
    for db_str in args.db:
        db_path = Path(db_str)
        if not db_path.exists():
            logger.error(f"DB not found: {db_path}")
            continue
        logger.info(f"Processing {db_path} ...")
        updated = backfill_one(db_path, args.only_unlabeled)
        logger.info(f"  -> {updated} stones labeled in {db_path.name}")
        total_updated += updated

    logger.info(f"Done. Total {total_updated} stones labeled.")


if __name__ == "__main__":
    # 文字化け防止（Windows コンソール対策）
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
