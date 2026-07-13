"""
utils.py: 汎用ユーティリティモジュール
スコア表からのハンマー権（後攻）の判定や、ディレクトリの削除、データ型の安全な変換など、
アプリケーション全体で使われる雑多で便利な機能を提供します。
"""
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

    if scores["LSFE"].isnull().all(): #LSFEがすべてNoneの場合
        return [None]
    
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

def to_team_code(team: str | None) -> str | None:
    """
        チーム名から3文字の国コードのみを取り出す
        リザルトブック上のチーム名は "CAN - Canada" のように
        「3文字コード + 空白 + ハイフン + 空白 + 国名」の形式になっている。
        DB には3文字コードのみを保存して表記揺れ ( 国名部分が大会ごとに異なる ) を防ぐ。
        Args:
            team : チーム名 ( 例: 'CAN - Canada' )。None の場合はそのまま返す
        Returns:
            str | None : 3文字コード ( 例: 'CAN' )。team が None ならば None
    """
    if not team:
        return team
    # " - " で分割した先頭要素がコード部分。
    # 万一ハイフンが無い ( 既に3文字化済み等 ) 場合も split の先頭は文字列全体になるため安全。
    return team.split(" - ")[0].strip()

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
