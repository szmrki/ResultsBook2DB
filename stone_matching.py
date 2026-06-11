"""
stone_matching.py: ストーン同定（投球元の特定）モジュール

DBに保存された各ショット後のストーン座標を用いて、
「盤面にある各ストーンが、そのエンドの何投目に投げられたものか」を同定する。

アルゴリズムの考え方:
    あるエンドはショット1→2→…と進み、ショットごとに「その時点の盤面に
    存在するストーン座標」が記録されている。隣り合うショット間で
    「前の局面のどのストーンが、今の局面のどのストーンに対応するか」を
    二部マッチング（線形割当問題, scipy の linear_sum_assignment）で解く。
    どの既存ストーンにも対応しなかった新しいストーン = 今まさに投げられた石、
    と判定し、そのショット番号(shot_order)を割り当てる。

同定結果は stones テーブルの shot_order カラムに書き戻す。
    - 正の値(1〜16): 何投目に投げられたか（= shots.number と同じドメイン）
    - 負の値        : ハンマールール等の制約に反した要確認のケース（-shot_num）
    - NULL          : 座標が無い行（x/y が NULL のプレースホルダ行）は対象外

注意: shot_order は「その石を投げたショット」の番号であり、
      stones.shot_id（その石が写っている盤面の所属ショット）とは別のショットを指す。
"""
import sqlite3
import logging
from typing import Any, Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# --- マッチングの閾値（DigitalCurling3 座標系, 単位: m） ---
EXIT_THRESHOLD = 2.0       # 既存ストーンが盤面から退出したとみなすコスト
ARRIVAL_THRESHOLD = 1.5    # 新しいストーンが新規参入（シューター）とみなすコスト
DIRECTION_TOLERANCE = 0.07  # y方向の逆走を許容する量（これ以内なら逆走とみなさない）


def match_sequential(
    active_stones: list[dict[str, Any]],
    current_stones: list[dict[str, Any]],
    shot_num: int,
    color_hammer: str | None = None,
    exit_threshold: float = EXIT_THRESHOLD,
    arrival_threshold: float = ARRIVAL_THRESHOLD,
    direction_tolerance: float = DIRECTION_TOLERANCE,
    cost_matrix_override: np.ndarray | None = None,
    log_context: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    1ショット分の局面遷移を二部マッチングで解き、現局面の各ストーンにラベルを付与する。

    Args:
        active_stones: 直前の局面で盤面にあったストーンのリスト。
            各要素は {'color': str, 'pos': (x, y), 'label': int | None, ...} の辞書。
        current_stones: 今回のショット後の盤面にあるストーンのリスト（同じ形式）。
        shot_num: 今回のショット番号（エンド内の投球順, 1〜16）。
        color_hammer: このエンドのハンマー（後攻）の色 'red' or 'yellow'。None なら色制約を使わない。
        exit_threshold: 既存ストーン退出のコスト。
        arrival_threshold: 新規参入（シューター）のコスト。
        direction_tolerance: 逆走とみなさない y 差の許容量(m)。
        cost_matrix_override: 距離コスト部分を外部行列で上書きする場合に指定（学習用）。
            指定時はハード制約（色・方向）はオーバーライド側で適用済みとみなす。
        log_context: 警告ログの先頭に付与する文脈文字列（例: "[大会 | 対戦 | End n (end_id=X)] "）。

    Returns:
        tuple[list[dict], list[dict]]:
            (現局面のストーンリスト（各要素に 'label' が設定済み）, 退出したストーンのリスト)
    """
    # 投球順から、今回シューターとなるべき色を決める
    # ルール: ハンマー（後攻）の色は偶数投目(2,4,...)、先攻の色は奇数投目(1,3,...)
    expected_color = None
    if color_hammer:
        expected_color = color_hammer if shot_num % 2 == 0 else (
            'yellow' if color_hammer == 'red' else 'red'
        )

    n_prev, n_curr = len(active_stones), len(current_stones)

    # エンドの最初のショット: 比較対象が無いので、全て今回の投球扱い
    if n_prev == 0:
        for s in current_stones:
            if expected_color and s['color'] != expected_color:
                logger.warning(
                    f"{log_context}Shot {shot_num}: Initial stone color mismatch "
                    f"(Expected {expected_color}, Got {s['color']})"
                )
            s['label'] = shot_num
        return current_stones, []

    # (n_prev + n_curr) x (n_curr + n_prev) の正方コスト行列を作る。
    # 行: [既存ストーン0..n_prev-1] + [新規参入ダミー0..n_curr-1]
    # 列: [現ストーン0..n_curr-1]   + [退出ダミー0..n_prev-1]
    cost_matrix = np.full((n_prev + n_curr, n_curr + n_prev), 1e9)

    # 1. 既存ストーン → 現ストーン（移動）のコスト = 距離
    #    色違いの対応づけ、およびハウス方向(y増加)への逆走は禁止する
    for i in range(n_prev):
        for j in range(n_curr):
            if cost_matrix_override is not None:
                # 学習済みコスト行列を使用（ハード制約はオーバーライド側で適用済み）
                cost_matrix[i, j] = cost_matrix_override[i, j]
                continue
            # 色が違うストーン同士は対応づけない
            if active_stones[i]['color'] != current_stones[j]['color']:
                continue
            # ストーンはハウス方向（y増加方向）にしか進まない。逆走は許容量以内のみOK
            if current_stones[j]['pos'][1] < active_stones[i]['pos'][1] - direction_tolerance:
                continue
            dist = np.sqrt(
                (active_stones[i]['pos'][0] - current_stones[j]['pos'][0]) ** 2 +
                (active_stones[i]['pos'][1] - current_stones[j]['pos'][1]) ** 2
            )
            cost_matrix[i, j] = dist

    # 2. 既存ストーン → 退出ダミー（盤外に出た）のコスト
    for i in range(n_prev):
        cost_matrix[i, n_curr + i] = exit_threshold

    # 3. 新規参入ダミー → 現ストーン（今投げられたシューター）のコスト
    for j in range(n_curr):
        if expected_color and current_stones[j]['color'] != expected_color:
            # シューター色と異なる石は新規参入になれない（事実上禁止）
            cost_matrix[n_prev + j, j] = 1e9
        else:
            cost_matrix[n_prev + j, j] = arrival_threshold

    # 4. 新規参入ダミー → 退出ダミー（バランス調整用, コスト0）
    for j in range(n_curr):
        for i in range(n_prev):
            cost_matrix[n_prev + j, n_curr + i] = 0.0

    # 線形割当問題を解く
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 既存ストーン→現ストーンの対応と、退出したストーンを集計
    matched_curr_indices = {}
    exited = []
    for r, c in zip(row_ind, col_ind):
        if r < n_prev:
            if c < n_curr:
                matched_curr_indices[c] = r  # 現ストーンc は既存ストーンr の続き
            else:
                exited.append(active_stones[r])  # 既存ストーンr は退出

    # 現ストーンにラベルを引き継ぎ、対応が無かったものを抽出
    final_active, unmatched_indices = [], []
    for j in range(n_curr):
        stone = current_stones[j]
        if j in matched_curr_indices:
            # 既存ストーンのラベル（=最初に投げられたショット番号）を引き継ぐ
            stone['label'] = active_stones[matched_curr_indices[j]]['label']
        else:
            unmatched_indices.append(j)
        final_active.append(stone)

    # 誰とも対応しなかったストーン = 今回(Shot N)投げられた石
    for idx in unmatched_indices:
        if expected_color and final_active[idx]['color'] != expected_color:
            # ハンマールールに反する: 負値ラベルで「要確認」を示す
            logger.warning(
                f"{log_context}Shot {shot_num}: Unmatched stone color is "
                f"'{final_active[idx]['color']}', but expected '{expected_color}'."
            )
            final_active[idx]['label'] = -shot_num
        else:
            final_active[idx]['label'] = shot_num

    return final_active, exited


# ============================================================================
# DB入出力（同定結果を stones.shot_order へ書き戻す）
# ============================================================================

def ensure_shot_order_column(conn: sqlite3.Connection) -> None:
    """
    stones テーブルに shot_order カラムが無ければ追加する（既存DB向けマイグレーション）。

    Args:
        conn: SQLite 接続。

    Returns:
        None
    """
    cur = conn.cursor()
    # stones テーブルが存在しない（未初期化）の場合は何もしない
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stones'")
    if cur.fetchone() is None:
        return
    # 既存カラムを調べ、shot_order が無ければ追加
    cur.execute("PRAGMA table_info(stones)")
    columns = [row[1] for row in cur.fetchall()]
    if "shot_order" not in columns:
        cur.execute("ALTER TABLE stones ADD COLUMN shot_order INTEGER")
        conn.commit()
        logger.info("Added 'shot_order' column to stones table.")


def _fetch_end_metadata(
    conn: sqlite3.Connection, end_id: int
) -> tuple[str, str, str, int, str | None] | None:
    """エンドの大会名・対戦カード・エンド番号・ハンマー色を取得する（ログ用文脈情報）。

    Args:
        conn: SQLite 接続。
        end_id: 対象エンドのID。

    Returns:
        tuple | None: (event_name, team_red, team_yellow, number, color_hammer)。
            エンドが見つからない場合は None。
    """
    cur = conn.cursor()
    cur.execute(
        """SELECT e.name, g.team_red, g.team_yellow, en.number, en.color_hammer
           FROM ends en
           JOIN games g ON en.game_id = g.id
           JOIN events e ON g.event_id = e.id
           WHERE en.id = ?""",
        (end_id,),
    )
    return cur.fetchone()


def _fetch_shots_for_end(conn: sqlite3.Connection, end_id: int) -> list[tuple[int, int, str]]:
    """エンド内のショットを投球順に取得する。

    Args:
        conn: SQLite 接続。
        end_id: 対象エンドのID。

    Returns:
        list[tuple[int, int, str]]: (shot_id, number, color) のリスト。
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT id, number, color FROM shots WHERE end_id = ? ORDER BY number ASC",
        (end_id,),
    )
    return cur.fetchall()


def _fetch_stones(conn: sqlite3.Connection, shot_id: int) -> list[dict[str, Any]]:
    """ショット後の盤面ストーン（座標あり）を取得する。

    Args:
        conn: SQLite 接続。
        shot_id: 対象ショットのID。

    Returns:
        list[dict]: {'db_id', 'color', 'pos', 'label'} の辞書のリスト。
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT id, color, x, y FROM stones WHERE shot_id = ? AND x IS NOT NULL AND y IS NOT NULL",
        (shot_id,),
    )
    rows = cur.fetchall()
    return [{'db_id': r[0], 'color': r[1], 'pos': (r[2], r[3]), 'label': None} for r in rows]


def label_end(conn: sqlite3.Connection, end_id: int) -> int:
    """
    1エンド分のストーンを同定し、stones.shot_order を更新する。

    Args:
        conn: SQLite 接続。
        end_id: 対象エンドのID。

    Returns:
        int: 更新したストーン行数。
    """
    meta = _fetch_end_metadata(conn, end_id)
    if meta is None:
        logger.warning(f"end_id={end_id} not found; skipping.")
        return 0
    event_name, team_red, team_yellow, end_number, color_hammer = meta
    # 対戦カードはチーム名コード部分のみ（例: "CAN - Canada" -> "CAN"）に短縮してログを見やすくする
    red = (team_red or "?").split(" - ")[0]
    yellow = (team_yellow or "?").split(" - ")[0]
    log_context = f"[{event_name} | {red} vs {yellow} | End {end_number} (end_id={end_id})] "

    shots = _fetch_shots_for_end(conn, end_id)

    cur = conn.cursor()
    active_stones: list[dict[str, Any]] = []
    updated = 0
    for shot_id, number, _color in shots:
        current_stones = _fetch_stones(conn, shot_id)
        active_stones, _exited = match_sequential(
            active_stones, current_stones, number,
            color_hammer=color_hammer, log_context=log_context,
        )
        # 今回の局面に存在する各ストーンに、確定したラベルを書き込む
        for s in current_stones:
            cur.execute(
                "UPDATE stones SET shot_order = ? WHERE id = ?",
                (s['label'], s['db_id']),
            )
            updated += 1
    logger.info(f"{log_context}{len(shots)} shots, {updated} stones labeled.")
    return updated


def label_event_ends(
    conn: sqlite3.Connection,
    event_id: int,
    progress_cb: Callable[[int, int], None] | None = None,
) -> int:
    """
    1イベント（大会）配下の全エンドを同定する。worker からの呼び出し用。

    Args:
        conn: SQLite 接続。
        event_id: 対象イベントのID。
        progress_cb: 進捗通知用コールバック。1エンド処理するたびに (done, total) で呼ばれる。
            None の場合は通知しない（同定結果・挙動には影響しない）。

    Returns:
        int: 更新したストーン行数の合計。
    """
    ensure_shot_order_column(conn)
    cur = conn.cursor()
    cur.execute(
        """SELECT en.id FROM ends en
           JOIN games g ON en.game_id = g.id
           WHERE g.event_id = ?
           ORDER BY g.id, en.number""",
        (event_id,),
    )
    end_ids = [row[0] for row in cur.fetchall()]

    total = 0
    n_ends = len(end_ids)
    for i, end_id in enumerate(end_ids, start=1):
        # 1エンドの同定失敗で大会全体を巻き込まないよう、エンド単位で保護する
        try:
            total += label_end(conn, end_id)
        except Exception:
            logger.exception(f"Stone matching failed for end_id={end_id}")
        if progress_cb is not None:
            progress_cb(i, n_ends)
    return total


def label_database(conn: sqlite3.Connection, only_unlabeled: bool = False) -> int:
    """
    DB内の全エンドを同定する。バックフィル（既存DBの一括更新）用。

    Args:
        conn: SQLite 接続。
        only_unlabeled: True の場合、未ラベル（shot_order が全て NULL）のエンドのみ対象。

    Returns:
        int: 更新したストーン行数の合計。
    """
    ensure_shot_order_column(conn)
    cur = conn.cursor()
    cur.execute("SELECT id FROM ends ORDER BY id")
    end_ids = [row[0] for row in cur.fetchall()]

    total = 0
    for end_id in end_ids:
        if only_unlabeled and _end_already_labeled(conn, end_id):
            continue
        try:
            total += label_end(conn, end_id)
        except Exception:
            logger.exception(f"Stone matching failed for end_id={end_id}")
    conn.commit()
    return total


def _end_already_labeled(conn: sqlite3.Connection, end_id: int) -> bool:
    """エンド内に shot_order が設定済みのストーンが1つでもあるか判定する。

    Args:
        conn: SQLite 接続。
        end_id: 対象エンドのID。

    Returns:
        bool: 1つでも shot_order が非NULLなら True。
    """
    cur = conn.cursor()
    cur.execute(
        """SELECT 1 FROM stones st
           JOIN shots sh ON st.shot_id = sh.id
           WHERE sh.end_id = ? AND st.shot_order IS NOT NULL
           LIMIT 1""",
        (end_id,),
    )
    return cur.fetchone() is not None
