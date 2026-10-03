"""
compare_db.py: 2つの SQLite DB の中身が同じかどうかを比較するスクリプト

リファクタリングの前後で同じ PDF から同じ DB ができるかを確かめるために使う ( issue #22 ) 。
テーブルの有無・列の構成・行数・各行の値をすべて比べ、違いがあれば表示する。
浮動小数点数 ( 石の座標など ) は小さな誤差を許容して比べる。

ファイル同士だけでなく、ディレクトリ同士も比較できる。ディレクトリを渡した場合は、
同じファイル名の DB を組にして比べる ( tools/make_golden_db.py の出力をまとめて比べる用途 ) 。

行の対応づけ方は2通りある。
    既定           : rowid ( ID ) で対応づける。ID まで含めて完全に一致することを確かめる。
    --by-content   : ID ではなく内容 ( 大会名・試合のページ・エンド番号・投球番号 ) で対応づける。
                     行の追加で ID がずれる変更 ( 例: issue #15 で MD の各エンドに number=0 の
                     shot を足す ) の前後を比べるときに使う。どのエンドが変わったかも一覧で表示する。

使い方:
    uv run python tools/compare_db.py db/golden/before/WJCC2022Men.db db/golden/after/WJCC2022Men.db
    uv run python tools/compare_db.py db/golden/before db/golden/after
    uv run python tools/compare_db.py db/golden/before db/golden/after --by-content

終了コード:
    0 = すべて一致 / 1 = 違いあり / 2 = 引数の誤り
"""
import argparse
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# 1テーブルあたりに表示する「違う行」の最大数 ( 大量に違う場合に画面が埋まらないようにする )
MAX_SHOWN_DIFFS = 5

# ─── 内容で対応づける比較 ( --by-content ) で使う、テーブルの親子関係 ─────────────
# 親テーブルから順に並べる ( 子の行のキーは、親の行のキーを先頭に付けて作るため ) 。
CONTENT_TABLES = ["events", "games", "ends", "shots", "stones", "lsds", "standings", "rosters"]
# テーブル名 → ( 親テーブル名, 親の ID を指す列名 )
PARENTS: dict[str, tuple[str, str]] = {
    "games": ("events", "event_id"),
    "ends": ("games", "game_id"),
    "shots": ("ends", "end_id"),
    "stones": ("shots", "shot_id"),
    "lsds": ("games", "game_id"),
    "standings": ("events", "event_id"),
    "rosters": ("events", "event_id"),
}
# 親の中で行を1つに決める列。ここに無いテーブル ( stones など ) は、親の中での並び順 ( ID 順 ) を使う
OWN_KEY: dict[str, str] = {"events": "name", "games": "page", "ends": "number", "shots": "number"}


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


def load_by_content(conn: sqlite3.Connection, tables: list[str]) -> dict[str, tuple[list[str], dict[tuple, tuple]]]:
    """各テーブルの行を、ID ではなく内容から作ったキーで引ける形に読み込む。

    キーは「親の行のキー + 親の中でその行を決める値」をつなげたタプルにする。
    例: stones の行のキーは ( 大会名, 試合のページ, エンド番号, 投球番号, その投球の中で何個目の石か ) 。
    値からは ID と親の ID の列を除く ( 行の追加で ID がずれても、内容が同じなら一致とみなすため ) 。

    Args:
        conn: SQLite の接続
        tables: 読み込むテーブル名 ( CONTENT_TABLES のうち、この DB にあるもの )

    Returns:
        dict[str, tuple[list[str], dict[tuple, tuple]]]:
            テーブル名 → ( 値に含める列名のリスト, キー → 値のタプル )
    """
    loaded: dict[str, tuple[list[str], dict[tuple, tuple]]] = {}
    # テーブル名 → ( その行の ID → その行のキー ) 。子テーブルのキーを作るときに使う
    id_to_key: dict[str, dict[int, tuple]] = {}

    for table in CONTENT_TABLES:
        if table not in tables:
            continue
        cols = get_columns(conn, table)
        parent = PARENTS.get(table)
        fk_col = parent[1] if parent else None
        # 値に含める列 ( ID と親の ID を除いた残り )
        value_cols = [c for c in cols if c != "id" and c != fk_col]
        own_col = OWN_KEY.get(table)

        rows: dict[tuple, tuple] = {}
        keys_of_ids: dict[int, tuple] = {}
        # 親の中での並び順 ( 0, 1, 2, ... ) を数えるためのカウンタ
        seq: Counter = Counter()
        for row in conn.execute(f'SELECT * FROM "{table}" ORDER BY id'):
            rec = dict(zip(cols, row))
            # 親の行が見つからない場合 ( 通常は起きない ) は、親の ID をそのままキーに使う
            parent_key = id_to_key[parent[0]].get(rec[fk_col], ("?", rec[fk_col])) if parent else ()
            if own_col is not None:
                key = parent_key + (rec[own_col],)
            else:
                key = parent_key + (seq[parent_key],)
                seq[parent_key] += 1
            # 同じキーの行が既にある場合 ( 通常は起きない ) は、番号を付けて区別する
            n_dup = 1
            base_key = key
            while key in rows:
                key = base_key + (f"dup{n_dup}",)
                n_dup += 1
            rows[key] = tuple(rec[c] for c in value_cols)
            keys_of_ids[rec["id"]] = key
        loaded[table] = (value_cols, rows)
        id_to_key[table] = keys_of_ids
    return loaded


def compare_by_content(conn_a: sqlite3.Connection, conn_b: sqlite3.Connection, tol: float, max_ends: int) -> bool:
    """2つの DB を、行の内容から作ったキーで対応づけて比較し、結果を表示する。

    テーブルごとの違いに加えて、shots / stones の違いをエンド単位にまとめ、
    「どのエンドが、どのように変わったか」を表示する。

    Args:
        conn_a: 比較元 ( 正解 ) の DB の接続
        conn_b: 比較先の DB の接続
        tol: 浮動小数点数の比較で許容する誤差
        max_ends: 違い方のパターンごとに表示するエンドの最大数

    Returns:
        bool: すべて一致していれば True
    """
    tables_a = set(get_tables(conn_a))
    tables_b = set(get_tables(conn_b))
    tables = [t for t in CONTENT_TABLES if t in tables_a and t in tables_b]
    data_a = load_by_content(conn_a, tables)
    data_b = load_by_content(conn_b, tables)

    ok = True
    # エンドのキー ( 大会名, 試合のページ, エンド番号 ) → そのエンドでの違いの数
    # ( 種類は "shots-" = 比較元にしか無い shot, "shots+" = 比較先にしか無い shot, "shots~" = 値が違う shot など )
    end_diffs: dict[tuple, Counter] = defaultdict(Counter)

    for table in tables:
        cols_a, rows_a = data_a[table]
        cols_b, rows_b = data_b[table]
        if cols_a != cols_b:
            ok = False
            print(f"[違い] {table}\n  列の構成が違う: {cols_a} != {cols_b}")
            continue

        only_a = sorted(rows_a.keys() - rows_b.keys(), key=str)
        only_b = sorted(rows_b.keys() - rows_a.keys(), key=str)
        changed = [k for k in rows_a.keys() & rows_b.keys()
                   if not all(values_equal(x, y, tol) for x, y in zip(rows_a[k], rows_b[k]))]
        changed.sort(key=str)
        if not (only_a or only_b or changed):
            continue

        ok = False
        print(f"[違い] {table}")
        if only_a:
            print(f"  比較元にしか無い行: {len(only_a)} 行  例: {only_a[:MAX_SHOWN_DIFFS]}")
        if only_b:
            print(f"  比較先にしか無い行: {len(only_b)} 行  例: {only_b[:MAX_SHOWN_DIFFS]}")
        if changed:
            print(f"  値が違う行: {len(changed)} 行")
            for k in changed[:MAX_SHOWN_DIFFS]:
                cols_changed = [f"{n}: {x!r} -> {y!r}" for n, x, y in zip(cols_a, rows_a[k], rows_b[k])
                                if not values_equal(x, y, tol)]
                print(f"    {k}: " + ", ".join(cols_changed))

        # shots / stones の違いは、エンド単位にまとめる ( キーの先頭3つがエンドを表す )
        if table in ("shots", "stones"):
            for k in only_a:
                end_diffs[k[:3]][f"{table}-"] += 1
            for k in only_b:
                end_diffs[k[:3]][f"{table}+"] += 1
            for k in changed:
                end_diffs[k[:3]][f"{table}~"] += 1

    if end_diffs:
        # 同じ違い方をしたエンドをまとめて数える ( 例: 「shot が1つ増え、石が2個増えた」エンドが何件か )
        patterns: dict[tuple, list[tuple]] = defaultdict(list)
        for end_key, counts in end_diffs.items():
            patterns[tuple(sorted(counts.items()))].append(end_key)
        print("[エンド単位の違い] ( - : 比較元にしか無い / + : 比較先にしか無い / ~ : 値が違う )")
        for pattern, end_keys in sorted(patterns.items(), key=lambda kv: -len(kv[1])):
            label = ", ".join(f"{name} x{n}" for name, n in pattern)
            print(f"  {len(end_keys)} エンド: {label}")
            for end_key in sorted(end_keys, key=str)[:max_ends]:
                print(f"    {end_key[0]} 試合p.{end_key[1]} End {end_key[2]}")
            if len(end_keys) > max_ends:
                print(f"    ... ほか {len(end_keys) - max_ends} エンド")
    return ok


def compare_db(path_a: Path, path_b: Path, tol: float, by_content: bool = False, max_ends: int = 10) -> bool:
    """2つの DB ファイルを比較し、結果を表示する。

    Args:
        path_a: 比較元 ( 正解 ) の DB のパス
        path_b: 比較先の DB のパス
        tol: 浮動小数点数の比較で許容する誤差
        by_content: True なら rowid ではなく行の内容で対応づけて比べる
        max_ends: by_content のとき、違い方のパターンごとに表示するエンドの最大数

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

        if by_content:
            # sqlite_sequence ( 採番の記録 ) などの内部テーブルは、行数が変われば当然変わるので比べない
            return compare_by_content(conn_a, conn_b, tol, max_ends) and ok

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
    parser.add_argument("--by-content", action="store_true",
                        help="rowid ではなく行の内容 ( 大会名・試合のページ・エンド番号・投球番号 ) で対応づけて比べる")
    parser.add_argument("--max-ends", type=int, default=10,
                        help="--by-content のとき、違い方のパターンごとに表示するエンドの最大数 ( 既定: 10 )")
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
        ok = compare_db(path_a, path_b, args.tol, args.by_content, args.max_ends)
        print("一致" if ok else "→ 違いあり")
        all_ok = all_ok and ok

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
