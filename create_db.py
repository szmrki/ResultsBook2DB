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
        
        # rosters: 大会ごとの選手ロースター ( チーム構成・ポジション・役割 )。events に直接ぶら下がる枝。
        # Final Standings ページの選手行から抽出する。4人制と MD で記載フォーマットが構造的に異なる
        # ( 4人制 = Position-Function 、MD = Gender ) ため、ends と同様に is_md でスキーマを分岐する。
        # 1つの DB は4人制か MD のどちらか専用 ( 混在しない ) ので、両者を同名 rosters にしても衝突しない。
        if is_md:
            # MD 版: MD の選手行は Position-Function を持たず Gender ( F/M/C ) で記載されるため、
            # 4人制とは別の列構成にする ( 詳細は docs/event_metadata_design.md 9節 )。
            #   role   : 'player' / 'coach'。4人制と同じ値集合に揃える ( 補欠概念は MD に無い )。
            #   gender : 'Female' / 'Male' / NULL ( coach は性別記載なし )。role とは直交する別軸。
            #            PDF 記載は F/M だが、既存の color='red'/'yellow' 等と表記を揃え単語で持つ。
            cur.execute(
                '''CREATE TABLE rosters (id INTEGER PRIMARY KEY AUTOINCREMENT, event_id INTEGER NOT NULL,
                team STRING, player_name STRING,
                role STRING, gender STRING,
                FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE ON UPDATE CASCADE)'''
            )
        else:
            # 4人制版: Position-Function 記号 ( 1/2/3/4, S, V, A, C ) を解釈して列に分解する。
            # 列は「粗→細」で並べる ( role が最も粗い分類、そこから position・フラグへ絞る )。
            #   role       : 'player' ( 氷上でプレーする選手・補欠含む ) / 'coach' ( C )。最も粗い分類。
            #                補欠 ( Alternate ) も氷上選手なので player とし、role では区別しない。
            #   position   : 投球順。正選手は 1-4、補欠 ( Alternate = フィフス ) は 5 とする。
            #                Coach ( C ) は投球順を持たないため NULL。ORDER BY 用途で INTEGER のまま持つ。
            #   is_skip    : Skip ( 記号 S ) なら 1、それ以外 0。position とは独立
            #   is_vice    : Vice-skip ( 記号 V ) なら 1、それ以外 0。position とは独立
            cur.execute(
                '''CREATE TABLE rosters (id INTEGER PRIMARY KEY AUTOINCREMENT, event_id INTEGER NOT NULL,
                team STRING, player_name STRING,
                role STRING, position INTEGER, is_skip INTEGER, is_vice INTEGER,
                FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE ON UPDATE CASCADE)'''
            )
        # 大会ごとにロースターを引く用途が主なため、event_id にインデックスを張る
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rosters_event_id ON rosters(event_id)")
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