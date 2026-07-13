"""
create_db.py: データベース構築モジュール
SQLite3を使用して、解析結果（ストーン座標、ショット情報、ゲーム結果など）を
保存するための各種テーブルを初期化・作成します。
"""
import sqlite3
import os
import sys
from pathlib import Path

def set_tables(dbname: str | Path, is_md: bool = False) -> None:
    conn = sqlite3.connect(dbname, isolation_level=None)  # autocommit モード（DDLをトランザクション内で実行するため）
    cur = conn.cursor()
    try:
        cur.execute("BEGIN")
        cur.execute("PRAGMA foreign_keys = ON;")
        # location: 会場所在地 ( 都市/州/国 )。会場名と分離できない場合は会場込みの生文字列。取れなければ NULL
        # venue: 会場名。所在地と分離できた場合のみ格納し、分離困難・未検出なら NULL
        cur.execute(
            '''CREATE TABLE events(id INTEGER PRIMARY KEY AUTOINCREMENT, name STRING NOT NULL UNIQUE,
            year INTEGER, category STRING, location STRING, venue STRING)'''
        )
        cur.execute(
            '''CREATE TABLE games(id INTEGER PRIMARY KEY AUTOINCREMENT, event_id INTEGER NOT NULL, page INTEGER,
            team_red STRING, team_yellow STRING, final_score_red INTEGER, final_score_yellow INTEGER,
            FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE ON UPDATE CASCADE)'''
        )
        if is_md:
            cur.execute(
                '''CREATE TABLE ends(id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER NOT NULL, page INTEGER,
                number INTEGER, color_hammer STRING, score_red INTEGER, score_yellow INTEGER, is_power_play INTEGER,
                FOREIGN KEY(game_id) REFERENCES games(id) ON DELETE CASCADE ON UPDATE CASCADE)'''
            )
        else:
            cur.execute(
                '''CREATE TABLE ends(id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER NOT NULL, page INTEGER,
                number INTEGER, color_hammer STRING, score_red INTEGER, score_yellow INTEGER,
                FOREIGN KEY(game_id) REFERENCES games(id) ON DELETE CASCADE ON UPDATE CASCADE)'''
            )
        cur.execute(
            '''CREATE TABLE shots(id INTEGER PRIMARY KEY AUTOINCREMENT, end_id INTEGER NOT NULL, number INTEGER,
            color STRING, team STRING, player_name STRING, type STRING, turn STRING, percent_score INTEGER,
            FOREIGN KEY(end_id) REFERENCES ends(id) ON DELETE CASCADE ON UPDATE CASCADE)'''
        )
        cur.execute(
            '''CREATE TABLE stones(id INTEGER PRIMARY KEY AUTOINCREMENT, shot_id INTEGER NOT NULL, color STRING,
            x FLOAT, y FLOAT, distance_from_center FLOAT, inhouse INTEGER, insheet INTEGER, shot_order INTEGER,
            FOREIGN KEY(shot_id) REFERENCES shots(id) ON DELETE CASCADE ON UPDATE CASCADE)'''
        )
        # ストーンマッチングは shot_id での絞り込みを多数回実行するため、インデックスを張る
        cur.execute("CREATE INDEX IF NOT EXISTS idx_stones_shot_id ON stones(shot_id)")
        cur.execute(
            '''CREATE TABLE lsds (id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER NOT NULL,
            team STRING, player_name STRING, distance_cm FLOAT,
            FOREIGN KEY(game_id) REFERENCES games(id) ON DELETE CASCADE ON UPDATE CASCADE)'''
        )
        # standings: 大会の最終順位 ( 1大会×複数チーム )。events に直接ぶら下がる独立した枝。
        # rank は同順位・番号飛びを許容する。team は3文字コード。MD/4人制で同一定義。
        cur.execute(
            '''CREATE TABLE standings (id INTEGER PRIMARY KEY AUTOINCREMENT, event_id INTEGER NOT NULL,
            rank INTEGER, team STRING,
            FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE ON UPDATE CASCADE)'''
        )
        # 大会ごとに順位を引く用途が主なため、event_id にインデックスを張る
        cur.execute("CREATE INDEX IF NOT EXISTS idx_standings_event_id ON standings(event_id)")
        cur.execute("COMMIT")
    except Exception:
        cur.execute("ROLLBACK")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    filename = sys.argv[1]
    dbname = os.getcwd()+f'/db/{filename}.db'
    set_tables(dbname)