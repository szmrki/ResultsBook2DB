"""
compare_db.py: 2つの SQLite DB の中身が同じかどうかを比較するスクリプト

リファクタリングの前後で同じ PDF から同じ DB ができるかを確かめるために使う ( issue #22 ) 。
テーブルの有無・列の構成・行数・各行の値をすべて比べ、違いがあれば表示する。
浮動小数点数 ( 石の座標など ) は小さな誤差を許容して比べる。

ファイル同士だけでなく、ディレクトリ同士も比較できる。ディレクトリを渡した場合は、
同じファイル名の DB を組にして比べる ( tools/make_golden_db.py の出力をまとめて比べる用途 ) 。

使い方:
    uv run python tools/compare_db.py db/golden/before/WJCC2022Men.db db/golden/after/WJCC2022Men.db
    uv run python tools/compare_db.py db/golden/before db/golden/after

終了コード:
    0 = すべて一致 / 1 = 違いあり / 2 = 引数の誤り
"""
import argparse
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

# 1テーブルあたりに表示する「違う行」の最大数 ( 大量に違う場合に画面が埋まらないようにする )
MAX_SHOWN_DIFFS = 5


def get_tables(conn: sqlite3.Connection) -> list[str]:
    """DB 内のテーブル名を名前順で返す。

    Args:
        conn: SQLite の接続

    Returns:
        list[str]: テーブル名のリスト ( sqlite_sequence などの内部テーブルも含む )
    """
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall()
    return [r[0] for r in rows]


def get_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """テーブルの列名を定義順で返す。

    Args:
        conn: SQLite の接続
        table: テーブル名

    Returns:
        list[str]: 列名のリスト
    """
    # PRAGMA table_info は ( 列番号, 列名, 型, ... ) の行を返す
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]


def values_equal(a: Any, b: Any, tol: float) -> bool:
    """2つの値が等しいかを判定する。浮動小数点数は誤差 tol までを等しいとみなす。

    Args:
        a: 比較する値
        b: 比較する値
        tol: 浮動小数点数の比較で許容する誤差 ( 絶対値 )

    Returns:
        bool: 等しければ True
    """
    if isinstance(a, float) or isinstance(b, float):
        # 片方が None ( NULL ) で片方が数値なら違う
        if a is None or b is None:
            return a is b
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)
    return a == b


def compare_table(conn_a: sqlite3.Connection, conn_b: sqlite3.Connection, table: str, tol: float) -> list[str]:
    """1つのテーブルの中身を比較し、違いの説明を返す。

    行は rowid で対応づけて比べる。リファクタリング後も INSERT の順番を変えない方針なので、
    ID ( rowid ) まで含めて一致することを確かめられる。

    Args:
        conn_a: 比較元 ( 正解 ) の DB の接続
        conn_b: 比較先の DB の接続
        table: テーブル名
        tol: 浮動小数点数の比較で許容する誤差

    Returns:
        list[str]: 違いの説明。一致していれば空のリスト
    """
    diffs: list[str] = []

    cols_a = get_columns(conn_a, table)
    cols_b = get_columns(conn_b, table)
    if cols_a != cols_b:
        diffs.append(f"  列の構成が違う: {cols_a} != {cols_b}")
        return diffs  # 列が違うと行の比較に意味が無いので、ここで打ち切る

    # rowid をキーにした辞書にする。位置で対応づけると、1行の欠けで後ろの行がすべて
    # ずれて「違う」と判定されてしまうため、同じ rowid の行同士を比べる
    query = f'SELECT rowid, * FROM "{table}"'
    rows_a = {r[0]: r[1:] for r in conn_a.execute(query)}
    rows_b = {r[0]: r[1:] for r in conn_b.execute(query)}
    if len(rows_a) != len(rows_b):
        diffs.append(f"  行数が違う: {len(rows_a)} != {len(rows_b)}")

    # 片方にしか無い行
    only_a = sorted(rows_a.keys() - rows_b.keys())
    only_b = sorted(rows_b.keys() - rows_a.keys())
    if only_a:
        diffs.append(f"  比較元にしか無い rowid ({len(only_a)} 行): {only_a[:MAX_SHOWN_DIFFS]}{' ...' if len(only_a) > MAX_SHOWN_DIFFS else ''}")
    if only_b:
        diffs.append(f"  比較先にしか無い rowid ({len(only_b)} 行): {only_b[:MAX_SHOWN_DIFFS]}{' ...' if len(only_b) > MAX_SHOWN_DIFFS else ''}")

    # 両方にある行の値を比べる
    shown = 0
    n_diff_rows = 0
    for rowid in sorted(rows_a.keys() & rows_b.keys()):
        row_a, row_b = rows_a[rowid], rows_b[rowid]
        if all(values_equal(x, y, tol) for x, y in zip(row_a, row_b)):
            continue
        n_diff_rows += 1
        if shown < MAX_SHOWN_DIFFS:
            # 違う列だけを抜き出して表示する
            changed = [f"{n}: {x!r} -> {y!r}" for n, x, y in zip(cols_a, row_a, row_b) if not values_equal(x, y, tol)]
            diffs.append(f"  rowid={rowid}: " + ", ".join(changed))
            shown += 1
    if n_diff_rows > shown:
        diffs.append(f"  ... ほか {n_diff_rows - shown} 行の値が違う")
    return diffs


def compare_db(path_a: Path, path_b: Path, tol: float) -> bool:
    """2つの DB ファイルを比較し、結果を表示する。

    Args:
        path_a: 比較元 ( 正解 ) の DB のパス
        path_b: 比較先の DB のパス
        tol: 浮動小数点数の比較で許容する誤差

    Returns:
        bool: すべて一致していれば True
    """
    # 読み取り専用で開き、比較で DB を書き換えてしまわないようにする
    conn_a = sqlite3.connect(f"file:{path_a}?mode=ro", uri=True)
    conn_b = sqlite3.connect(f"file:{path_b}?mode=ro", uri=True)
    try:
        ok = True
        tables_a = set(get_tables(conn_a))
        tables_b = set(get_tables(conn_b))
        if tables_a != tables_b:
            ok = False
            print(f"[違い] テーブルの構成: 片方にしか無い = {sorted(tables_a ^ tables_b)}")

        for table in sorted(tables_a & tables_b):
            diffs = compare_table(conn_a, conn_b, table, tol)
            if diffs:
                ok = False
                print(f"[違い] {table}")
                print("\n".join(diffs))
        return ok
    finally:
        conn_a.close()
        conn_b.close()


def main() -> None:
    """コマンドライン引数を解釈し、ファイル同士またはディレクトリ同士を比較する。"""
    parser = argparse.ArgumentParser(description="2つの SQLite DB の中身を比較する")
    parser.add_argument("a", type=Path, help="比較元 ( 正解 ) の DB またはディレクトリ")
    parser.add_argument("b", type=Path, help="比較先の DB またはディレクトリ")
    parser.add_argument("--tol", type=float, default=1e-9, help="浮動小数点数の許容誤差 ( 既定: 1e-9 )")
    args = parser.parse_args()

    # 比較する組 ( 正解, 比較先 ) を作る
    if args.a.is_dir() and args.b.is_dir():
        names_a = {p.name for p in args.a.glob("*.db")}
        names_b = {p.name for p in args.b.glob("*.db")}
        for name in sorted(names_a ^ names_b):
            print(f"[注意] 片方のディレクトリにしか無い: {name}")
        pairs = [(args.a / n, args.b / n) for n in sorted(names_a & names_b)]
    elif args.a.is_file() and args.b.is_file():
        pairs = [(args.a, args.b)]
    else:
        print("ファイル同士、またはディレクトリ同士を指定してください")
        sys.exit(2)

    all_ok = True
    for path_a, path_b in pairs:
        print(f"=== {path_a.name} ===")
        ok = compare_db(path_a, path_b, args.tol)
        print("一致" if ok else "→ 違いあり")
        all_ok = all_ok and ok

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
