import sqlite3
import os
import sys

def set_tables(dbname: str, is_md: bool = False):
    conn = sqlite3.connect(dbname)
    cur = conn.cursor()
    cur.execute(
        '''CREATE TABLE events(id INTEGER PRIMARY KEY AUTOINCREMENT, name STRING NOT NULL UNIQUE, 
        year INTEGER, category STRING)'''
    )
    cur.execute(
        '''CREATE TABLE games(id INTEGER PRIMARY KEY AUTOINCREMENT, event_id INTEGER NOT NULL, page INTEGER, 
        team_red STRING, team_yellow STRING, final_score_red INTEGER, final_score_yellow INTEGER)'''
    )
    if is_md:
        cur.execute(
            '''CREATE TABLE ends(id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER NOT NULL, page INTEGER, 
            number INTEGER, color_hammer STRING, score_red INTEGER, score_yellow INTEGER, is_power_play INTEGER)'''
        )
    else:
        cur.execute(
            '''CREATE TABLE ends(id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER NOT NULL, page INTEGER, 
            number INTEGER, color_hammer STRING, score_red INTEGER, score_yellow INTEGER)'''
        )
    cur.execute(
        '''CREATE TABLE shots(id INTEGER PRIMARY KEY AUTOINCREMENT, end_id INTEGER NOT NULL, number INTEGER, 
        color STRING, team STRING, player_name STRING, type STRING, turn STRING, percent_score INTEGER)'''
    )
    cur.execute(
        '''CREATE TABLE stones(id INTEGER PRIMARY KEY AUTOINCREMENT, shot_id NOT NULL, color STRING, 
        x FLOAT, y FLOAT, distance_from_center FLOAT, inhouse INTEGER, insheet INTEGER)'''
    )
    conn.commit()
    conn.close()

if __name__ == "__main__":
    filename = sys.argv[1]
    dbname = os.getcwd()+f'/db/{filename}.db'
    set_tables(dbname)