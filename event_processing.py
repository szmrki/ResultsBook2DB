"""
event_processing.py: 1大会分の解析処理の本体 ( GUI に依存しない共通の処理 )

PDF 1ファイル ( = 1大会 ) を DB に取り込む処理を、次の3つに分けて提供する。

1. extract_event     : PDF を読み、解析結果 ( EventResult ) を作る。DB には一切触れない。
                       YOLO モデルの準備 ( 必要ならファインチューニング ) と推論を含む。
2. write_event       : 解析結果を SQLite に書き込む ( INSERT のみ ) 。
3. postprocess_event : DB 上の後処理 ( MD のハンマー補正とストーン同定 ) を行う。

GUI ( worker.py の Worker ) はこの3つを続けて呼ぶ。「PDF を読む処理」と「DB に書き込む処理」を
分けてあるので、PDF の解析を GPU のあるマシンで行い、DB への書き込みを別のマシンで行う、
といった使い方もできる ( issue #21 / #22 ) 。

進捗の通知と中断の確認は、Qt のシグナルではなくコールバック関数で受け取る。
    progress_cb(percent, message) : 進捗率 ( 0-100 ) とメッセージを通知する
    should_stop()                 : 中断が要求されていれば True を返す
"""
import json
import logging
import os
import shutil
import sqlite3
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable

import fitz  # PyMuPDF
import pdfplumber
from ultralytics import YOLO

from detection import create_pseudo_label
from pdf_tools import (
    extract_game_result, extract_lsd_from_text, extract_rosters, extract_rosters_md,
    extract_shotbyshot, extract_standings, extract_venue, extract_year_and_category,
    is_standings_page, save_images,
)
from stone_matching import correct_equidistant_blank_hammer, label_event_ends
from utils import delete_files, get_hammer, to_team_code
from yolo_tools import create_yaml, split_train_val

logger = logging.getLogger(__name__)

# PyInstaller で固めた場合は展開先 ( _MEIPASS ) 、通常実行ではこのファイルのあるディレクトリを基準にする
resource_path = lambda p: Path(getattr(
    sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__))
    )) / p

# YOLO のクラス番号と石の色の対応
NUM2COLOR = {0: "red", 1: "yellow"}

ProgressCallback = Callable[[int, str], None]
StopCallback = Callable[[], bool]

# 解析結果ファイル ( JSON ) の形式のバージョン。形式を変えたら上げる
RESULT_FORMAT_VERSION = 1


# ─── 解析結果の入れ物 ─────────────────────────────────────────────────────────
# DB のテーブル構造 ( events → games → ends → shots → stones ) に対応する入れ子の構造。
# DB の ID は持たない ( ID は write_event で INSERT したときに決まる ) 。

@dataclass
class StoneResult:
    """盤面上の石1個。座標は DigitalCurling3 座標系 ( 単位: m ) 。"""
    color: str
    x: float
    y: float
    distance_from_center: float
    inhouse: int
    insheet: int


@dataclass
class ShotResult:
    """1投分の情報と、その投球後の盤面 ( stones ) 。"""
    number: int
    color: str | None
    team: str | None
    player_name: str | None
    type: str | None
    turn: str | None
    percent_score: Any
    stones: list[StoneResult] = field(default_factory=list)


@dataclass
class EndResult:
    """1エンド分の情報。Shot by Shot のページが無いエンドは page が None で shots が空になる。"""
    number: int
    color_hammer: str | None
    score_red: int | None
    score_yellow: int | None
    is_power_play: int | None = None  # MD のみ使う ( 4人制では None )
    page: int | None = None           # Shot by Shot のページ番号
    shots: list[ShotResult] = field(default_factory=list)
    # MD で Shot by Shot のページを処理したエンドのみ True。このとき prepositioned に
    # 事前配置石 ( 取れなかった場合は None ) が入る。ストーン同定に渡すマップの元になる。
    has_prepositioned_info: bool = False
    prepositioned: list[dict[str, Any]] | None = None


@dataclass
class LsdResult:
    """LSD ( Last Stone Draw ) 1件。"""
    team: str | None
    player_name: str
    distance_cm: float


@dataclass
class GameResult:
    """1試合分の情報。チーム名は3文字コード。"""
    page: int
    team_red: str | None
    team_yellow: str | None
    final_score_red: int | None
    final_score_yellow: int | None
    lsds: list[LsdResult] = field(default_factory=list)
    ends: list[EndResult] = field(default_factory=list)


@dataclass
class EventResult:
    """1大会分の解析結果。"""
    name: str
    year: int | None
    category: str | None
    is_md: bool
    location: str | None = None
    venue: str | None = None
    standings: list[tuple[int, str]] = field(default_factory=list)   # ( 順位, チームコード )
    rosters: list[dict[str, Any]] = field(default_factory=list)      # extract_rosters / extract_rosters_md の出力
    games: list[GameResult] = field(default_factory=list)


# ─── 解析結果ファイル ( JSON ) の書き出し・読み込み ───────────────────────────
# PDF の解析 ( extract_event ) と DB への書き込み ( write_event ) を別のマシンで行う場合に、
# 解析結果をファイルとして受け渡すために使う。浮動小数点数は Python の json が
# 値を変えずに書き出し・読み込みするため、ファイルを経由しても DB の内容は変わらない。

def _json_default(value: Any) -> Any:
    """
        json が標準では書き出せない値を変換する。
        numpy の数値 ( np.int64 など ) が混ざっていた場合に、Python の数値に直す。

        Args:
            value : json が書き出せなかった値

        Returns:
            Any : 書き出せる形に直した値
    """
    if hasattr(value, "item"):  # numpy のスカラー
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_event_result(result: EventResult, path: str | Path) -> None:
    """
        解析結果を JSON ファイルに書き出す。

        Args:
            result : extract_event の解析結果
            path : 書き出し先のファイルパス
    """
    data = {"format_version": RESULT_FORMAT_VERSION, "event": asdict(result)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=_json_default)


def load_event_result(path: str | Path) -> EventResult:
    """
        JSON ファイルから解析結果を読み込む。
        JSON にはタプルが無くリストになるため、タプルだった箇所 ( 順位、事前配置石の座標 ) は
        読み込み時にタプルへ戻す。

        Args:
            path : 解析結果ファイルのパス

        Returns:
            EventResult : 読み込んだ解析結果

        Raises:
            ValueError : ファイルの形式のバージョンが、このプログラムの対応するものと違う場合
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format_version") != RESULT_FORMAT_VERSION:
        raise ValueError(f"Unsupported result format version: {data.get('format_version')} "
                         f"(expected {RESULT_FORMAT_VERSION})")
    ev = data["event"]

    games: list[GameResult] = []
    for g in ev.pop("games"):
        ends: list[EndResult] = []
        for e in g.pop("ends"):
            shots = [ShotResult(**{**sh, "stones": [StoneResult(**st) for st in sh["stones"]]})
                     for sh in e.pop("shots")]
            pre = e.pop("prepositioned")
            if pre is not None:
                pre = [{**p, "pos": tuple(p["pos"])} for p in pre]
            ends.append(EndResult(**e, shots=shots, prepositioned=pre))
        lsds = [LsdResult(**lsd) for lsd in g.pop("lsds")]
        games.append(GameResult(**g, lsds=lsds, ends=ends))
    standings = [tuple(row) for row in ev.pop("standings")]
    return EventResult(**ev, standings=standings, games=games)


# ─── 1. PDF → 解析結果 ────────────────────────────────────────────────────────

def extract_event(pdf_path: str | Path, event_name: str, is_md: bool,
                  progress_cb: ProgressCallback | None = None,
                  should_stop: StopCallback | None = None) -> EventResult | None:
    """
        PDF を解析し、1大会分の解析結果を作る。DB には触れない。
        解析には YOLO モデルを使用し、必要に応じてファインチューニングも行う。

        Args:
            pdf_path : PDF ファイルのパス
            event_name : 大会名
            is_md : MD ( 混合ダブルス ) なら True
            progress_cb : 進捗通知用のコールバック ( 省略可 )
            should_stop : 中断要求を判定するコールバック ( 省略可 )

        Returns:
            EventResult | None : 解析結果。途中で中断された場合は None
    """
    game = event_name
    stopped = should_stop if should_stop is not None else (lambda: False)
    notify = progress_cb if progress_cb is not None else (lambda percent, message: None)

    year, category = extract_year_and_category(game, is_md)
    result = EventResult(name=game, year=year, category=category, is_md=is_md)

    doc = fitz.open(pdf_path)
    logger.info(f"Processing PDF: {pdf_path}")

    model = prepare_model(game, doc, progress_cb, should_stop)

    # 会場情報は複数の順位ページに同一のものが載るため、最初の順位ページで1回だけ取得する
    venue_saved = False
    # 「いま処理中の試合」の情報。Shot by Shot のページは直前の Game Results の試合に属する
    current_game: GameResult | None = None
    hammers: list[int | None] = []
    game_context = ""
    num_end = 1
    start_time_det = time.time()
    with pdfplumber.open(pdf_path) as pdf:
        for pn in range(doc.page_count):
            # 中断チェック (ページごとのループ先頭)
            if stopped():
                break
            notify(int(pn/doc.page_count*100), "Extracting data...")
            page_num = pn + 1
            page_plumber = pdf.pages[pn]
            page_mu = doc[pn]
            text = page_mu.get_text()
            # 順位ページ ( is_standings_page ) の判定を最初に置く。
            # is_standings_page は「単独行 Final Standings + 列見出し行」という
            # 最も厳格な条件のため試合ページを誤って奪う心配が小さく、逆に順位ページの
            # テキストに "Game Results" 等が紛れても正しく順位側に振り分けられる
            # ( 分岐の順序依存による偽陰性を避ける )。
            if is_standings_page(page_mu): #最終順位表のページ ( 複数ページにわたる )
                # 順位 ( standings ) を抽出する。順位ページは複数あるため都度追加する。
                result.standings.extend(extract_standings(page_mu))
                # 選手ロースター ( rosters ) を抽出する。順位ページ ( 複数 ) に選手行が
                # 分かれて載るため standings と同様に都度追加する。4人制と MD で記載フォーマットが
                # 異なるため、抽出関数を切り替える。
                if is_md:
                    result.rosters.extend(extract_rosters_md(page_mu))
                else:
                    result.rosters.extend(extract_rosters(page_mu))
                # 会場情報 ( location / venue ) は最初の順位ページで1回だけ取得する。
                # 2ページ目以降には同じ会場情報が載るため再取得はしない。
                if not venue_saved:
                    result.location, result.venue = extract_venue(page_mu, game)
                    venue_saved = True
                    logger.info(f"[{game}] Standings page: {page_num} - location: {result.location}, venue: {result.venue}")

            elif "Game Results" in text: #新たな試合
                if is_md:
                    scores, power_play_ends = extract_game_result(page_plumber, is_md) #得点表のdfとPPエンドのリスト
                else:
                    scores = extract_game_result(page_plumber) #得点表のdf
                    power_play_ends = []

                hammers = get_hammer(scores, is_md)  #各エンドのハンマー情報
                team_red = scores.at[0, "team"]
                team_yellow = scores.at[1, "team"]
                game_context = f"{team_red} vs {team_yellow}"
                logger.debug(f"Scores:\n{scores}")
                logger.debug(f"Hammers: {hammers}")
                logger.info(f"[{game_context}] - Game Results page: {page_num}")
                try:
                    fin_red = int(scores.at[0, "Total"]) #得点表のdfから最終得点を記録
                    fin_yellow = int(scores.at[1, "Total"])
                except (ValueError, TypeError):
                    logger.warning(f"[{game_context}] Could not parse final scores from page {page_num}")
                    fin_red = None
                    fin_yellow = None

                # DB には3文字コードのみを保存する ( 国名部分の表記揺れを防ぐ )。
                # team_red / team_yellow 変数自体は LSFE 照合や LSD 紐付けで
                # フルネームのまま使うため、解析結果に入れる値だけ変換する。
                team_red_code = to_team_code(team_red)
                team_yellow_code = to_team_code(team_yellow)
                current_game = GameResult(page=page_num, team_red=team_red_code, team_yellow=team_yellow_code,
                                          final_score_red=fin_red, final_score_yellow=fin_yellow)
                result.games.append(current_game)

                # --- LSDデータを抽出 ---
                plumber_text = page_plumber.extract_text()
                if plumber_text:
                    lsd_results = extract_lsd_from_text(plumber_text)
                    for lsd in lsd_results:
                        if lsd["player_red"] and lsd["lsd_red"] is not None:
                            current_game.lsds.append(LsdResult(team_red_code, lsd["player_red"], lsd["lsd_red"]))
                        if lsd["player_yellow"] and lsd["lsd_yellow"] is not None:
                            current_game.lsds.append(LsdResult(team_yellow_code, lsd["player_yellow"], lsd["lsd_yellow"]))

                # ---------------------------
                # ここでエンドの情報をまとめて作る ( この時点では page は None )
                for i in range(len(hammers)):
                    if hammers[i] == None: break #コンシード済みのため
                    num_end_val = i + 1
                    str_end = str(num_end_val)
                    try:
                        score_red = int(scores.at[0, str_end]) #得点表のdfから得点を取得
                        score_yellow = int(scores.at[1, str_end])
                    except Exception:
                        score_red = None #存在しない場合はNULL
                        score_yellow = None

                    try:
                        color_hammer = NUM2COLOR[hammers[i]]
                    except Exception:
                        color_hammer = None

                    end = EndResult(number=num_end_val, color_hammer=color_hammer,
                                    score_red=score_red, score_yellow=score_yellow)
                    if is_md:
                        # Power Play情報の抽出ロジック
                        end.is_power_play = 1 if num_end_val in power_play_ends else 0
                    current_game.ends.append(end)

                num_end = 1

            elif "Shot by Shot" in text: #新たなエンド
                # 該当するエンドを探し、ページ情報を記録する
                if current_game is None:
                    raise RuntimeError(f"Shot by Shot page {page_num} appeared before any Game Results page")
                end = next((e for e in current_game.ends if e.number == num_end), None)
                if end is None:
                    raise RuntimeError(f"[{game_context}] End {num_end} not found for Shot by Shot page {page_num}")
                end.page = page_num

                stones_end, shot_info, pre_stones_np = extract_shotbyshot(doc, page_mu, model, is_md)
                logger.info(f"[{game_context}] End {num_end} - Shot-by-Shot page: {page_num} - Number of shots: {max(len(stones_end), len(shot_info))}")

                # MD版: Prepositioned stone座標をマッチング用の辞書形式に変換して保持する
                if is_md:
                    end.has_prepositioned_info = True
                    if pre_stones_np is not None:
                        pre_stone_dicts: list[dict[str, Any]] = []
                        for row in pre_stones_np:
                            if row[5] == 1:  # insheet フラグが立っている行のみ
                                pre_stone_dicts.append({
                                    'color': NUM2COLOR[int(row[0])],
                                    'pos': (float(row[1]), float(row[2])),
                                    'label': 0,  # Prepositioned stone は shot_order=0
                                })
                        # 有効なストーンが取れた場合のみ保持、取れなかった場合はNone
                        end.prepositioned = pre_stone_dicts if pre_stone_dicts else None
                    else:
                        # MD版だがPrepositioned stone画像がなかった → スキップ対象
                        end.prepositioned = None

                for shot_num, (stones, info) in enumerate(zip_longest(stones_end, shot_info), start=1):
                    if info is not None: #正常時
                        shot_type = info["type"]; percent_score = info["score"]
                        turn = info["turn"]; team = info["team"]; player_name = info["player"]
                    else: #ショット情報が取れない場合はNULLとし、ストーン配置のみ保存する
                        shot_type = None; percent_score = None
                        turn = None; team = None; player_name = None
                        logger.warning(f"[{game_context}] End {num_end} - Shot {shot_num} - Shot info not found")

                    try:
                        shot_color = NUM2COLOR[(hammers[num_end - 1] + (shot_num % 2)) % 2] #現在のショットの色を指定
                    except (TypeError, IndexError):
                        logger.warning(f"[{game_context}] End {num_end} - Shot {shot_num} - Shot color not found")
                        shot_color = None
                    shot = ShotResult(number=shot_num, color=shot_color, team=team, player_name=player_name,
                                      type=shot_type, turn=turn, percent_score=percent_score)

                    if stones is not None: #正常時
                        # insheet フラグが立っている行のみ石として保持する
                        shot.stones = [
                            StoneResult(color=NUM2COLOR[int(row[0])], x=float(row[1]), y=float(row[2]),
                                        distance_from_center=float(row[3]), inhouse=int(row[4]), insheet=int(row[5]))
                            for row in stones if row[5] == 1
                        ]
                    end.shots.append(shot)
                num_end += 1
            else:
                continue
    doc.close()
    # 検出途中で中断された場合は結果を返さない ( 呼び出し側で大会ごと破棄する )
    if stopped():
        return None
    elapsed_det = time.time() - start_time_det
    logger.info(f"[{game}] Detection complete (took {elapsed_det:.2f}s).")
    return result


def prepare_model(game: str, doc: fitz.Document,
                  progress_cb: ProgressCallback | None = None,
                  should_stop: StopCallback | None = None) -> YOLO:
    """
        ファインチューニング済みモデルが存在すればロード、なければ疑似ラベルでFTして保存する。

        Args:
            game : 大会名 ( モデルのファイル名 complete_model/{game}.pt に使う )
            doc : PyMuPDF のファイルオブジェクト ( 学習用の画像を取り出す )
            progress_cb : 進捗通知用のコールバック ( 省略可 )
            should_stop : 中断要求を判定するコールバック ( 省略可 )

        Returns:
            YOLO : 大会用のモデル
    """
    stopped = should_stop if should_stop is not None else (lambda: False)
    notify = progress_cb if progress_cb is not None else (lambda percent, message: None)

    work_dir = Path.cwd()
    model_dir = resource_path(Path("complete_model"))
    game_pt = model_dir / f"{game}.pt"

    if not game_pt.exists():
        start_time_ft = time.time()
        notify(0, "Preparing fine-tuning...")

        base_model_path = resource_path(model_dir / "base.pt")
        if not base_model_path.exists():
            raise FileNotFoundError(f"Base model not found at {base_model_path}. Please ensure 'complete_model/base.pt' exists.")

        model = YOLO(base_model_path)

        dataset_dir = work_dir / "yolo_dataset"
        image_dir = dataset_dir / "images"
        label_dir = dataset_dir / "labels"
        yaml_path = work_dir / "yaml" / "data.yaml"

        try:
            num_images = save_images(doc, output_dir=image_dir, save_num=400)
            num_labels = create_pseudo_label(model, image_dir=image_dir, output_dir=label_dir, threshold=0.75)
            logger.info(f"Dataset prepared: {num_labels} pseudo labels from {num_images} images.")
            split_train_val(image_dir, label_dir, train_ratio=0.8)
            create_yaml(yaml_path, dataset_dir)
        except Exception as e:
            logger.error(f"Failed to prepare dataset for fine-tuning: {e}")
            raise

        def on_train_epoch_end(trainer):
            curr = trainer.epoch + 1
            total = trainer.epochs
            notify(int(curr / total * 100), "Fine-tuning...")
            if stopped():
                trainer.stop = True

        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        try:
            logger.info(f"Starting fine-tuning for event: {game}")
            results = model.train(
                data=resource_path(yaml_path),
                epochs=50,
                imgsz=600,
                iou=0.3,
                conf=0.5,
                save=True,
                name=game,
                exist_ok=False,
                workers=0,
                patience=10,
            )
            final_epoch = model.trainer.epoch + 1
            if not stopped():
                if results and hasattr(results, 'results_dict'):
                    map50 = results.results_dict.get('metrics/mAP50(B)', 'N/A')
                    map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 'N/A')
                    precision = results.results_dict.get('metrics/precision(B)', 'N/A')
                    recall = results.results_dict.get('metrics/recall(B)', 'N/A')
                    logger.info(f"Fine-tuning complete. Results: mAP50={map50:.6f}, mAP50-95={map50_95:.6f}, Precision={precision:.6f}, Recall={recall:.6f}")
                else:
                    logger.info("Fine-tuning complete. Accuracy metrics not available.")
        except Exception as e:
            logger.error(f"Fine-tuning failed for event '{game}': {e}")
            logger.error(traceback.format_exc())
            model.clear_callback("on_train_epoch_end")
            raise

        if not stopped():
            Path(game_pt).unlink(missing_ok=True)
            try:
                save_dir = Path(model.trainer.save_dir)
                best_pt = save_dir / "weights" / "best.pt"
                shutil.copy2(best_pt, game_pt)
                logger.info(f"Successfully saved fine-tuned model from {best_pt} as {game_pt.name}")
            except Exception as e:
                logger.warning(f"Could not copy best.pt to {game_pt.name}: {e}. Attempting direct save.")
                try:
                    model.save(game_pt)
                except Exception as save_e:
                    logger.error(f"Failed to save model directly: {save_e}")
                    raise

        try:
            delete_files(image_dir / "train")
            delete_files(label_dir / "train")
            delete_files(image_dir / "val")
            delete_files(label_dir / "val")
        except Exception as e:
            logger.warning(f"Failed to clean up dataset directories: {e}")

        model.clear_callback("on_train_epoch_end")
        if not stopped():
            elapsed_ft = time.time() - start_time_ft
            logger.info(f"[{game}] Fine-tuning complete ({final_epoch} epochs) (took {elapsed_ft:.2f}s).")
            notify(100, "Fine-tuning complete.")

    return YOLO(game_pt)


# ─── 2. 解析結果 → SQLite ─────────────────────────────────────────────────────

def event_exists(conn: sqlite3.Connection, event_name: str) -> bool:
    """
        同じ名前の大会が既に DB にあるかを調べる。
        events.name は UNIQUE のため、同名の大会は取り込めない。PDF の解析は時間がかかるので、
        解析を始める前にこの関数で確認する。

        Args:
            conn : SQLite の接続
            event_name : 大会名

        Returns:
            bool : 既に存在すれば True
    """
    row = conn.execute("SELECT 1 FROM events WHERE name = ?", (event_name,)).fetchone()
    return row is not None


def write_event(conn: sqlite3.Connection, result: EventResult) -> tuple[int, dict[int, list[dict[str, Any]] | None]]:
    """
        解析結果を SQLite に書き込む。コミットはしない ( 呼び出し側で行う ) 。

        各テーブルへの INSERT の順番は、PDF のページ順 ( 試合 → エンド → ショット ) を保つ。
        ID ( AUTOINCREMENT ) はテーブルごとに INSERT した順に振られるため、順番を保つことで
        常に同じ ID になる。

        Args:
            conn : SQLite の接続
            result : extract_event の解析結果

        Returns:
            tuple[int, dict] : 大会の ID ( event_id ) と、MD の事前配置石のマップ
                ( end_id → 事前配置石のリスト。取れなかったエンドは None ) 。
                マップには Shot by Shot のページを処理した MD のエンドだけが入る。4人制では空。

        Raises:
            sqlite3.IntegrityError : 同じ名前の大会が既に存在する場合
    """
    is_md = result.is_md
    cur = conn.cursor()

    #eventテーブルに大会名、年、カテゴリ、会場情報を記述
    cur.execute('INSERT INTO events(name, year, category, location, venue) VALUES (?, ?, ?, ?, ?)',
                (result.name, result.year, result.category, result.location, result.venue))
    event_id = cur.lastrowid #event_idを取得

    # 順位 ( standings )
    cur.executemany("INSERT INTO standings(event_id, rank, team) VALUES (?, ?, ?)",
                    [(event_id, rank, team) for rank, team in result.standings])

    # 選手ロースター ( rosters ) 。rosters のスキーマは is_md で分かれるため、挿入列を切り替える。
    if is_md:
        # MD 版: role / gender を持つ ( Position-Function は無い )。
        cur.executemany(
            "INSERT INTO rosters(event_id, team, player_name, role, gender) VALUES (?, ?, ?, ?, ?)",
            [(event_id, r["team"], r["player_name"], r["role"], r["gender"]) for r in result.rosters])
    else:
        # 4人制版: role / position / is_skip / is_vice を持つ。
        cur.executemany(
            """INSERT INTO rosters(event_id, team, player_name, role, position, is_skip, is_vice)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [(event_id, r["team"], r["player_name"], r["role"], r["position"], r["is_skip"], r["is_vice"])
             for r in result.rosters])

    # MD版: Prepositioned stoneの座標を end_id → list[dict] のマップにまとめる ( ストーン同定で使う )
    prepositioned_map: dict[int, list[dict[str, Any]] | None] = {}

    for g in result.games:
        cur.execute("""INSERT INTO games(event_id, page, team_red, team_yellow,
                        final_score_red, final_score_yellow) VALUES (?, ?, ?, ?, ?, ?)""",
                        (event_id, g.page, g.team_red, g.team_yellow, g.final_score_red, g.final_score_yellow))
        game_id = cur.lastrowid #game_idを取得

        cur.executemany("INSERT INTO lsds (game_id, team, player_name, distance_cm) VALUES (?, ?, ?, ?)",
                        [(game_id, lsd.team, lsd.player_name, lsd.distance_cm) for lsd in g.lsds])

        for end in g.ends:
            if is_md:
                cur.execute("""INSERT INTO ends(game_id, page, number, color_hammer,
                                score_red, score_yellow, is_power_play) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                                (game_id, end.page, end.number, end.color_hammer,
                                 end.score_red, end.score_yellow, end.is_power_play))
            else:
                cur.execute("""INSERT INTO ends(game_id, page, number, color_hammer,
                                score_red, score_yellow) VALUES (?, ?, ?, ?, ?, ?)""",
                                (game_id, end.page, end.number, end.color_hammer, end.score_red, end.score_yellow))
            end_id = cur.lastrowid #end_idを取得

            if end.has_prepositioned_info:
                prepositioned_map[end_id] = end.prepositioned

            for shot in end.shots:
                cur.execute("""INSERT INTO shots(end_id, number, color, team, player_name,
                                    type, turn, percent_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                                    (end_id, shot.number, shot.color, shot.team, shot.player_name,
                                    shot.type, shot.turn, shot.percent_score))
                shot_id = cur.lastrowid #shot_idを取得

                rows = [(shot_id, s.color, s.x, s.y, s.distance_from_center, s.inhouse, s.insheet)
                        for s in shot.stones]
                if len(rows) == 0: #ストーンが存在しない場合はidのみ
                    rows = [(shot_id, None, None, None, None, None, None)]
                #ストーンはまとめてinsert
                cur.executemany("""INSERT INTO stones (shot_id, color, x, y, distance_from_center,
                                inhouse, insheet) VALUES (?, ?, ?, ?, ?, ?, ?)""", rows)

    return event_id, prepositioned_map


# ─── 3. DB 上の後処理 ─────────────────────────────────────────────────────────

def postprocess_event(conn: sqlite3.Connection, event_id: int, is_md: bool,
                      prepositioned_map: dict[int, list[dict[str, Any]] | None] | None = None,
                      progress_cb: ProgressCallback | None = None,
                      should_stop: StopCallback | None = None) -> int:
    """
        書き込み済みの1大会に対して、DB 上の後処理を行う。コミットはしない ( 呼び出し側で行う ) 。
        MD ではハンマー補正を行い、その後、全エンドのストーン同定 ( shot_order の決定 ) を行う。

        Args:
            conn : SQLite の接続
            event_id : 対象の大会の ID
            is_md : MD ( 混合ダブルス ) なら True
            prepositioned_map : write_event が返した事前配置石のマップ ( MD のみ使う )
            progress_cb : 進捗通知用のコールバック ( 省略可 )
            should_stop : 中断要求を判定するコールバック ( 省略可 ) 。同定の途中で True を返すと打ち切る

        Returns:
            int : shot_order を更新した石の行数
    """
    notify = progress_cb if progress_cb is not None else (lambda percent, message: None)

    # MD版: 同距離ブランクエンド ( ハウス内に両チームの石が残った膠着ブランク ) では
    # 先攻後攻が交代しないため、get_hammer が入れた誤った交代を打ち消す。
    # 同定は color_hammer を色制約に使うため、補正は同定より前に行う。
    if is_md:
        corrected = correct_equidistant_blank_hammer(conn, event_id)
        if corrected:
            name = conn.execute("SELECT name FROM events WHERE id = ?", (event_id,)).fetchone()[0]
            logger.info(f"[{name}] Equidistant-blank hammer correction: {corrected} ends updated.")

    # ストーン同定: この大会の全エンドについて各石の投球元(shot_order)を特定する
    notify(0, "Matching stones...")
    return label_event_ends(
        conn, event_id,
        progress_cb=lambda done, total: notify(
            int(done / total * 100) if total else 100,
            f"Matching stones... ({done}/{total})"
        ),
        should_stop=should_stop,  # 同定中も中止を受け付けて打ち切る
        prepositioned_map=prepositioned_map if is_md else None,
    )
