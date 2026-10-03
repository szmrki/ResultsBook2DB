"""
prepositioned.py: MD ( 混合ダブルス ) の事前配置石 ( prepositioned stones ) の補正

MD では各エンドの開始時に、両チームの石が1個ずつ決まった位置に置かれる。
    ガード石 : 先攻の石。ハウスの手前 ( ガードゾーン ) に置く
    ハウス石 : 後攻の石。ハウスの中に置く
Results Book の Shot by Shot ページには、この2個を描いた「Prepositioned stones」の図がある。

この図は、まれに図そのものが誤っている。抽出処理や YOLO の誤検出ではなく、
元の PDF の誤りなので、同じ試合の他のエンドや1投目の図と照らし合わせて補正する。
対象にする誤りは次の2種類。

    (a) 1投目に投げた石が混入している ( 石が3個描かれている )
    (b) 2個の石の色が逆になっている
"""
from __future__ import annotations

import logging
import math
import statistics
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 型ヒントのためだけの import 。実行時に import すると event_processing と循環してしまう
    from event_processing import EndResult, GameResult, StoneResult

logger = logging.getLogger(__name__)

# 動いていない石は、別の図でも 0.01〜0.02 m の精度で同じ位置に描かれる。
# 下の2つの上限は、この精度に対して十分な余裕を持たせた値。

# ガード石の前後位置 ( y ) を「同じ試合の他のエンドと同じ」とみなす差の上限 ( m )
GUARD_Y_TOLERANCE = 0.1

# 別の図に描かれた石を「同じ位置にある」とみなす距離の上限 ( m )
SAME_POSITION_TOLERANCE = 0.05


def _is_normal_pair(stones: list[StoneResult]) -> bool:
    """事前配置石が「正常な2個」かどうかを判定する。

    正常な2個とは、色が違い、片方だけがハウスの外にある ( = ガード石とハウス石が1個ずつ ) こと。

    Args:
        stones: 1エンド分の事前配置石

    Returns:
        bool: 正常な2個なら True
    """
    if len(stones) != 2 or stones[0].color == stones[1].color:
        return False
    # inhouse は 1 = ハウス内 / 0 = ハウス外。合計が 1 なら「片方だけハウス内」
    return stones[0].inhouse + stones[1].inhouse == 1


def _reference_guard_y(game: GameResult) -> float | None:
    """同じ試合の正常なエンドから、ガード石の前後位置 ( y ) の基準を求める。

    ガード石の位置は同じ大会でも試合 ( シート ) によって違うので、大会単位の基準や定数は使わず、
    試合ごとに求める。パワープレイのエンドでも前後位置は通常のエンドと同じなので、区別せずに使う。
    外れ値 ( 図の誤り ) に引っ張られないよう、平均ではなく中央値を取る。

    Args:
        game: 1試合分の解析結果

    Returns:
        float | None: ガード石の y の基準。正常なエンドが1つも無ければ None
    """
    guard_ys: list[float] = []
    for end in game.ends:
        if end.prepositioned and _is_normal_pair(end.prepositioned):
            # ハウスの外にあるほうがガード石
            guard = next(s for s in end.prepositioned if s.inhouse == 0)
            guard_ys.append(guard.y)
    return statistics.median(guard_ys) if guard_ys else None


def _remove_thrown_stone(end: EndResult, ref_y: float | None, context: str) -> None:
    """(a) 事前配置の図に混入した「1投目に投げた石」を取り除く。

    石が3個で、色が「2個 + 1個」に分かれているときだけ対象にする。
    1投目を投げるのは先攻で、ガード石も先攻の石なので、2個ある色が「ガード石と投げた石」、
    1個だけの色がハウス石になる。2個のうち、前後位置が基準 ( 同じ試合の他のエンドのガード石 ) に
    合うほうをガード石として残し、もう一方を取り除く。

    Args:
        end: 対象のエンド ( prepositioned を書き換える )
        ref_y: 同じ試合のガード石の y の基準 ( 求められなかった場合は None )
        context: ログの先頭に付ける文脈 ( 大会名・対戦カード )
    """
    stones = end.prepositioned
    if not stones or len(stones) != 3:
        return
    counts = Counter(s.color for s in stones)
    pair_color = next((color for color, n in counts.items() if n == 2), None)
    if pair_color is None:
        return  # 3個とも同じ色。別の種類の異常なので、ここでは扱わない

    where = f"[{context}] End {end.number} (page {end.page})"
    if ref_y is None:
        logger.warning(f"{where}: Prepositioned figure has 3 stones, but no reference guard position "
                       f"is available in this game. Left as detected.")
        return

    pair = [s for s in stones if s.color == pair_color]
    # 基準と同じ前後位置にある石 ( = ガード石の候補 )
    near = [s for s in pair if abs(s.y - ref_y) <= GUARD_Y_TOLERANCE]
    if len(near) != 1:
        # 2個とも基準に近い、または2個とも遠い場合は、どちらがガード石か決められない
        logger.warning(f"{where}: Prepositioned figure has 3 stones, but the guard stone could not be "
                       f"determined (reference y={ref_y:.2f}, candidates y="
                       f"{[round(s.y, 2) for s in pair]}). Left as detected.")
        return

    thrown = next(s for s in pair if s is not near[0])
    end.prepositioned = [s for s in stones if s is not thrown]
    logger.warning(f"{where}: Prepositioned figure contained the first thrown stone. "
                   f"Removed {thrown.color} stone at ({thrown.x:.2f}, {thrown.y:.2f}).")


def _fix_swapped_colors(end: EndResult, context: str) -> None:
    """(b) 事前配置の図で逆になっている石の色を入れ替える。

    事前配置石と同じ位置に、1投目の図で「逆の色の石」があれば、事前配置の図の色が誤っているとみなす。

    Args:
        end: 対象のエンド ( prepositioned の石の色を書き換える )
        context: ログの先頭に付ける文脈 ( 大会名・対戦カード )
    """
    stones = end.prepositioned
    if not stones or len(stones) != 2 or stones[0].color == stones[1].color:
        return
    # 1投目の盤面 ( 無ければ照合できない )
    shot1 = next((shot for shot in end.shots if shot.number == 1), None)
    if shot1 is None or not shot1.stones:
        return

    n_same = 0      # 同じ位置に、同じ色の石があった事前配置石の数 ( 色が正しい証拠 )
    n_opposite = 0  # 同じ位置に、逆の色の石しか無かった事前配置石の数 ( 色が逆の証拠 )
    for pre in stones:
        nearby = [s for s in shot1.stones
                  if math.hypot(s.x - pre.x, s.y - pre.y) <= SAME_POSITION_TOLERANCE]
        if any(s.color == pre.color for s in nearby):
            n_same += 1
        elif nearby:
            n_opposite += 1

    if n_opposite == 0:
        return
    where = f"[{context}] End {end.number} (page {end.page})"
    if n_same > 0:
        logger.warning(f"{where}: Prepositioned stone colors are inconsistent with the first shot figure "
                       f"(matched: {n_same}, opposite: {n_opposite}). Left as detected.")
        return

    # 2個の色を入れ替える
    stones[0].color, stones[1].color = stones[1].color, stones[0].color
    logger.warning(f"{where}: Prepositioned stone colors were swapped in the figure. Corrected to "
                   f"{[(s.color, round(s.x, 2), round(s.y, 2)) for s in stones]}.")


def correct_prepositioned_stones(game: GameResult, context: str) -> None:
    """1試合分の事前配置石を補正する ( 解析結果を書き換える ) 。

    エンドごとに (a) 投げた石の混入 → (b) 色が逆 の順に確認する。
    補正の後も「色の違う2個」になっていないエンドは、検出のまま残して警告ログを出す。

    Args:
        game: 1試合分の解析結果 ( 各エンドの prepositioned を書き換える )
        context: ログの先頭に付ける文脈 ( 大会名・対戦カード )
    """
    # 基準は補正の前に求める
    ref_y = _reference_guard_y(game)
    for end in game.ends:
        if not end.prepositioned:
            continue  # 事前配置石が取れなかったエンド ( 図が無い、または検出が0個 )
        _remove_thrown_stone(end, ref_y, context)
        _fix_swapped_colors(end, context)

        stones = end.prepositioned
        if len(stones) != 2 or stones[0].color == stones[1].color:
            logger.warning(f"[{context}] End {end.number} (page {end.page}): Unexpected prepositioned stones "
                           f"{[(s.color, round(s.x, 2), round(s.y, 2)) for s in stones]}. Left as detected.")
