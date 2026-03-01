import pandas as pd
from typing import Any
from pathlib import Path

def get_hammer(scores: pd.DataFrame, is_md: bool = False) -> list[int | None]: 
    """
        スコア表ベースでエンドごとのハンマーのindexを取得する
        Args:
            score : スコア表のデータフレーム
            is_md : MDのときはハンマーの取得方法が変わる
        Returns:
            list[int | None] : 0 or 1のリスト、長さはエンド数
    """
    #LSFE列に*があるチームがラストストーンエンド
    hammer_list = []
    try:
        start = int(scores[scores["LSFE"].astype(str).str.contains(r"\*")].index[0])
    except Exception:
        start = None
    hammer_list.append(start)

    exclude_cols = ["team", "LSFE", "Total"]
    # 対象列（エンド列）を抽出
    end_cols = [col for col in scores.columns if col not in exclude_cols]
    # いずれかの行について中身が空でないセルを True とする
    non_empty = scores[end_cols].astype(str).apply(
                            lambda col: col.str.strip().ne("").any())
    # NaN でないものだけ数える
    total_ends = non_empty.sum()

    for end in range(1, total_ends):
        str_end = str(end)

        #先に数値かどうか判定してから、ハンマーの処理を行う
        val0 = scores.at[0, str_end]
        val1 = scores.at[1, str_end]
        if str(val0).isdigit() and str(val1).isdigit():
            if int(val0) > int(val1): #team0が得点した場合
                hammer_list.append(1)
            elif int(val0) < int(val1): #team1が得点した場合
                hammer_list.append(0)
            else: #ブランクの場合
                if is_md:
                    hammer_list.append(1-hammer_list[-1])  #前のエンドから交代
                else:
                    hammer_list.append(hammer_list[-1])    #前のエンドと同じ
        else:  #コンシード等で数値が入力されていない場合
            hammer_list.append(None)

    return hammer_list

def delete_files(dir: str | Path) -> None:
    """
        指定されたディレクトリ内のすべてのファイルを削除する
        Args: 
            dir : 削除対象のディレクトリ
    """
    # フォルダ内の全ファイルを削除（サブフォルダは無視）
    for file in Path(dir).iterdir():
        if file.is_file():
            file.unlink()

# 数値に変換できるものは int、できないものはそのまま
def __try_int(x: Any) -> int | str:
    try:
        return int(x)
    except ValueError:
        return x
