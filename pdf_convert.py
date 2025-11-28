from tools import *
from ultralytics import YOLO
import sqlite3
import glob
import os

if __name__ == "__main__":
    IS_MD = False  #MD版かそうでないか

    color2num = {"red": 0, "yellow": 1}
    num2color = {0: "red", 1: "yellow"}

    work_dir = os.getcwd()
    #print(work_dir)
    dbname = work_dir +'/db/curling_stones_md_v5.db' if IS_MD else work_dir + '/db/curling_stones_4p_v7.db'
    dbname = work_dir + '/db/fordebug.db'
    #print(dbname)
    conn = sqlite3.connect(dbname)
    cur = conn.cursor()

    #Where we want to look for the data (nominally the data/ directory).
    current_dir = work_dir + '/rb_data/data_md' if IS_MD else work_dir + '/rb_data/data_4p'
    #Change directory to the starting directory.
    os.chdir(current_dir)
    games = glob.glob("*")
    #print(games)
    #games = ["WMCC2022", "WMCC2023", "WMCC2024", "WMCC2025", "WWCC2022", "WWCC2023", "WWCC2024", "WWCC2025"]
    games = ["WJCC2022Women"]
    for game in games:
        #game = "WMCC2022"
        #モデルの定義
        if game == "WMCC2025":
            model = YOLO("../../../complete_model/game11_retrain.pt")
        elif game == "WWCC2025":
            model = YOLO("../../../complete_model/game12.pt")
        elif game in ["WMCC2022", "WMCC2023", "WMCC2024", "WWCC2023", 
                        "WWCC2024", "PCCC2024Men", "PCCC2024Women",
                        "PCCC2025Men", "PCCC2025Women"]:
            model = YOLO("../../../complete_model/game13.pt")
        else:
            model = YOLO("../../../complete_model/game10_2_retrain.pt")
        
        cur.execute('INSERT INTO event(name) VALUES (?)', (game,)) #eventテーブルに大会の名前を記述
        event_id = cur.lastrowid #event_idを取得

        file_path = glob.glob(f"{game}/*.pdf")[0]
        doc = fitz.open(file_path)
        print(file_path)
        with pdfplumber.open(file_path) as pdf:
            for pn in range(doc.page_count):
                page_num = pn + 1
                page_plumber = pdf.pages[pn]
                page_mu = doc[pn]
                text = page_mu.get_text()
                if "Game Results" in text: #新たな試合
                    print(f"Game Results page: {page_num}")
                    """
                    try:
                        scores = extract_game_result(page)   #得点表のdf
                        print(scores)
                        #break
                    except Exception as e:
                        print(e)
                        #break
                    """
                    scores = extract_game_result(page_plumber) #得点表のdf
                    print(scores)
                    hammers = get_hammer(scores, IS_MD)  #各エンドのハンマー情報
                    print(hammers)
                    team_red = scores.at[0, "team"]
                    team_yellow = scores.at[1, "team"]
                    try:
                        fin_red = int(scores.at[0, "Total"])
                        fin_yellow = int(scores.at[1, "Total"])
                    except ValueError:
                        fin_red = None
                        fin_yellow = None
                    
                    cur.execute("""INSERT INTO games(event_id, page, team_red, team_yellow, 
                                    final_score_red, final_score_yellow) VALUES (?, ?, ?, ?, ?, ?)""", 
                                    (event_id, page_num, team_red, team_yellow, fin_red, fin_yellow))
                    game_id = cur.lastrowid #game_idを取得
                    num_end = 1

                elif "Shot by Shot" in text: #新たなエンド
                    """           
                    if num_end <= MAX_END:
                        str_end = str(num_end)
                    else: #EEの場合は列名が数値に変換できなくなるので、10(8)の右隣の列名を取得
                        extra_num = num_end - MAX_END
                        col_idx = scores.columns.get_loc(str(MAX_END)) 
                        str_end = scores.columns[col_idx + extra_num]     # 右隣の列名
                    """
                    #print(num_end)
                    str_end = str(num_end)
                    try:
                        score_red = int(scores.at[0, str_end])
                        score_yellow = int(scores.at[1, str_end])
                    except Exception:
                        score_red = None
                        score_yellow = None
                    
                    try:
                        color_hammer = num2color[hammers[num_end - 1]]
                    except Exception:
                        color_hammer = None
                    cur.execute("""INSERT INTO ends(game_id, page, number, color_hammer, 
                                    score_red, score_yellow) VALUES (?, ?, ?, ?, ?, ?)""", 
                                    (game_id, page_num, num_end, color_hammer, 
                                    score_red, score_yellow))
                    end_id = cur.lastrowid #end_idを取得
                    print(f"Shot-by-Shot page: {page_num}")
                    stones_end, shot_info = extract_shotbyshot(doc, page_mu, model, IS_MD)
                    #print(stones_end[0])
                    print("num shots: ", len(shot_info))
                    print(shot_info)
                    """
                    if stones_end.shape[0] != len(shot_info):
                        print("num images: ", stones_end.shape[0])
                        print(shot_info)
                        break
                    """
                    #if page_num == 146: break

                    for shot_num, (stones, info) in enumerate(zip(stones_end, shot_info), start=1):
                        team, player_name, shot_type, turn, percent_score = info
                        shot_color = num2color[(hammers[num_end - 1] + (shot_num % 2)) % 2] #現在のショットの色を指定
                        cur.execute("""INSERT INTO shots(end_id, number, color, team, player_name, 
                                        type, turn, percent_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", 
                                        (end_id, shot_num, shot_color, team, player_name, 
                                        shot_type, turn, percent_score))
                        shot_id = cur.lastrowid #shot_idを取得

                        rows = [(shot_id, num2color[int(row[0])], *row[1:]) for row in stones if row[5] == 1]
                        if len(rows) == 0:
                            rows = [(shot_id, None, None, None, None, None, None)]
                        cur.executemany("""INSERT INTO stones (shot_id, color, x, y, dist, 
                                        inhouse, insheet) VALUES (?, ?, ?, ?, ?, ?, ?)""", rows)
                    num_end += 1
                    #break
                else:
                    continue
        doc.close()
        conn.commit()
        #if game == "WJCC2023Women":
        #    break
    conn.close()
