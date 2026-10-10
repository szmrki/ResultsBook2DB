"""
detection.py: ストーン検出・座標変換に関するコアアルゴリズムモジュール
YOLOモデルを用いた推論を実行し、画像上の境界ボックス(bbox)を
カーリングシートの物理的なDC座標（ハウス中心からの距離など）に変換します。
"""
from ultralytics import YOLO
import numpy as np
import cv2
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

WIDTH = 299
TEE_LINE = 159.5
BACKLINE = 40
CENTER_X = 149
DIAMETER = 239
DC3_WIDTH = 4.75
DC3_TEE_LINE = 38.405
DC3_BACKLINE = 40.234
DC3_CENTER_X = 0
DC3_RADIUS = 1.829
DC3_STONE_RADIUS = 0.145

XA = (DC3_WIDTH / WIDTH); XB = (DC3_WIDTH / 2)   
YA = ((DC3_TEE_LINE - DC3_BACKLINE) / (TEE_LINE - BACKLINE))
YB = DC3_TEE_LINE - YA * TEE_LINE

#class_names = ["red", "yellow"]

#上下反転の判定に使う値
FLIP_BAND = 25           #上下の帯 ( 盤面の外に出た石が並ぶ部分 ) として判定から除く行数
FLIP_WHITE_MIN = 240     #この値以上なら白とみなす ( JPEGのノイズで255にならない場合があるため )
FLIP_WHITE_MARGIN = 0.15 #上下の白い画素の数の差が、全体のこの割合未満なら判定しない
FLIP_DARK_MAX = 80       #この値未満なら黒とみなす ( JPEGのノイズで0にならない場合があるため )
FLIP_LINE_RATIO = 0.8    #1行のうちこの割合以上が黒なら横線とみなす ( 線に石が少し重なる場合があるため )
FLIP_HOG_FROM = 18       #ホッグラインを探す範囲の始まり ( 上端または下端から数えた行数 )
FLIP_HOG_TO = 22         #ホッグラインを探す範囲の終わり ( この行は含まない )

#同じストーンが赤と黄の両方で検出されたときの判定に使う値
#中心間の距離 [px] がこの値未満なら、同じストーンを2色で検出したとみなす。
#実測では、同じストーンの二重検出は 0.9px 未満、図の上で重ねて描かれた別々のストーンは 4.3px 以上だった
SAME_STONE_MAX_DIST = 2
STONE_SAT_MIN = 150      #ストーンの塗りとみなす彩度の下限 ( ハウスの淡い色を除く )
STONE_VAL_MIN = 100      #ストーンの塗りとみなす明度の下限 ( 輪郭線や印の黒を除く )
RED_HUE_MAX = 10         #赤の色相の範囲。OpenCV の H は 0〜179 で、赤は両端にまたがるため「この値以下」または
RED_HUE_MIN = 170        #「この値以上」を赤とする
YELLOW_HUE_MIN = 15      #黄の色相の範囲の始まり
YELLOW_HUE_MAX = 40      #黄の色相の範囲の終わり
CLS_RED = 0              #モデルのクラス番号 ( 赤 )
CLS_YELLOW = 1           #モデルのクラス番号 ( 黄 )

def __flip_by_white(img: np.ndarray) -> bool | None:
    """
        上下の白い画素の数を比べて、上下反転が必要かを判定する
        ハウスは色付きの大きな円なので、ハウスのある側は白い画素が少ない
        Args:
            img : シート画像のnumpy配列 ( 600 x 300 x 3 )
        Returns:
            bool | None : 反転が必要 ( ハウスが下にある ) なら True、不要なら False、
                          上下の差が小さく判定できなければ None
    """
    white = np.all(img[FLIP_BAND:-FLIP_BAND] >= FLIP_WHITE_MIN, axis=-1)
    half = white.shape[0] // 2
    top_white = int(white[:half].sum())
    bottom_white = int(white[half:].sum())
    ratio = (top_white - bottom_white) / max(top_white + bottom_white, 1)
    if abs(ratio) < FLIP_WHITE_MARGIN:
        return None
    return ratio > 0  #上の方が白い = ハウスが下にある

def __flip_by_hogline(img: np.ndarray) -> bool | None:
    """
        ホッグライン ( 黒い横線 ) の位置から、上下反転が必要かを判定する
        正しい向きではホッグラインは下端の付近にある
        Args:
            img : シート画像のnumpy配列 ( 600 x 300 x 3 )
        Returns:
            bool | None : 反転が必要 ( ホッグラインが上にある ) なら True、不要なら False、
                          線が見つからない、または上下の両方にあって判定できなければ None
    """
    #左右1ピクセルが余白の可能性があるため除く
    dark_rows = (img[:, 1:WIDTH].max(axis=2) < FLIP_DARK_MAX).mean(axis=1) > FLIP_LINE_RATIO
    #ホッグラインは端から19〜20行目にある ( 図の高さが600と601の2種類あり、1行ずれる ) 。
    #図の枠線 ( 端から0〜1行目 ) と、反対側の向きのときのバックライン ( 端から39〜40行目 ) は拾わない
    top = bool(dark_rows[FLIP_HOG_FROM:FLIP_HOG_TO].any())
    bottom = bool(dark_rows[-FLIP_HOG_TO:-FLIP_HOG_FROM].any())
    if top == bottom:
        return None
    return top

def needs_flip(img: np.ndarray) -> bool | None:
    """
        シート画像を上下反転 ( 180°回転 ) する必要があるかを判定する
        上下の白い画素の数で判定し、決まらない場合のみホッグラインの位置で判定する
        Args:
            img : シート画像のnumpy配列 ( 600 x 300 x 3 )
        Returns:
            bool | None : 反転が必要なら True、不要なら False、判定できなければ None
    """
    flip = __flip_by_white(img)
    if flip is None:
        flip = __flip_by_hogline(img)
    return flip

def __count_stone_pixels(img: np.ndarray, xyxy: np.ndarray) -> tuple[int, int]:
    """
        バウンディングボックスの中にある、赤と黄のストーンの塗りの画素数を数える
        Args:
            img : シート画像のnumpy配列 ( BGR )
            xyxy : バウンディングボックス ( 左上x, 左上y, 右下x, 右下y )
        Returns:
            tuple[int, int] : ( 赤の画素数, 黄の画素数 )
    """
    height, width = img.shape[:2]
    #箱が画像の外にはみ出す場合があるため、画像の範囲に収める
    x1 = max(int(np.floor(xyxy[0])), 0)
    y1 = max(int(np.floor(xyxy[1])), 0)
    x2 = min(int(np.ceil(xyxy[2])), width)
    y2 = min(int(np.ceil(xyxy[3])), height)
    if x1 >= x2 or y1 >= y2:
        return 0, 0

    #色相 ( H ) で赤と黄を分ける。彩度と明度の下限で、ハウスの淡い色・白地・黒い線を除く
    hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    fill = (sat >= STONE_SAT_MIN) & (val >= STONE_VAL_MIN)
    red = fill & ((hue <= RED_HUE_MAX) | (hue >= RED_HUE_MIN))
    yellow = fill & (hue >= YELLOW_HUE_MIN) & (hue <= YELLOW_HUE_MAX)
    return int(red.sum()), int(yellow.sum())

def __find_wrong_color_boxes(img: np.ndarray, xyxy: np.ndarray, cls: np.ndarray, conf: np.ndarray,
                             context: str = "") -> set[int]:
    """
        同じストーンが赤と黄の両方で検出された組を探し、誤った色の方の番号を返す
        YOLO の NMS はクラスごとに行われるため、同じ位置でも色が違う検出は両方残る。
        投げた直後のストーンに印が描かれる大会などで、赤いストーンに黄の検出が重なることがある。
        確信度は誤った色の方が高い場合があるため、図の塗りの色で正しい方を決める
        Args:
            img : 検出に使ったシート画像のnumpy配列 ( BGR )
            xyxy : 検出ごとのバウンディングボックス ( N x 4 )
            cls : 検出ごとのクラス番号 ( N ) 。赤が0、黄が1
            conf : 検出ごとの確信度 ( N )
            context : 警告ログに付ける、画像の出どころの説明 ( ページ番号など )
        Returns:
            set[int] : 取り除く検出の番号 ( xyxy の行番号 ) の集合
    """
    centers = np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2])
    drop: set[int] = set()
    for r in np.flatnonzero(cls == CLS_RED):
        for y in np.flatnonzero(cls == CLS_YELLOW):
            if np.hypot(*(centers[r] - centers[y])) >= SAME_STONE_MAX_DIST:
                continue

            #2つの箱はほぼ同じ位置にあるので、赤の箱の範囲で塗りの色を数える
            n_red, n_yellow = __count_stone_pixels(img, xyxy[r])
            logger.debug(f"[{context}] 赤と黄が重なった検出があります "
                         f"( 塗りの画素数 赤: {n_red}, 黄: {n_yellow} / 確信度 赤: {conf[r]:.2f}, 黄: {conf[y]:.2f} )")
            if n_red > n_yellow:
                drop.add(int(y))
            elif n_yellow > n_red:
                drop.add(int(r))
            else:
                #塗りの色で決まらない場合は、確信度の低い方を取り除く
                loser = int(r) if conf[r] < conf[y] else int(y)
                drop.add(loser)
                logger.warning(f"[{context}] 赤と黄が重なった検出の色を図から判定できませんでした "
                               f"( 確信度の高い方を残します。赤: {conf[r]:.2f}, 黄: {conf[y]:.2f} )")
    return drop

def get_stones_pos(imgs: list[np.ndarray], model: YOLO, context: str = "") -> list[np.ndarray]:
    """
        ストーン座標をモデルを用いて取得する（バッチ推論）
        Args:
            imgs : シート画像のnumpy配列リスト
            model : YOLOのモデル
            context : 警告ログに付ける、画像の出どころの説明 ( ページ番号など )
        Returns:
            list[np.ndarray] : 各画像について (16 x 6) のストーン情報配列のリスト
    """
    if not imgs:
        return []

    # 画像ごとの前処理（反転判定・上下マスク）
    preprocessed = []
    for i, img in enumerate(imgs, start=1):
        flip = needs_flip(img)
        if flip is None:
            logger.warning(f"[{context}] {i}枚目のシート画像の上下の向きを判定できませんでした ( 反転せずに検出します )")
        elif flip:
            img = cv2.flip(img, -1)

        #誤検出を防ぐため上下に白でマスク
        img[:20,1:-2] = 255
        img[-19:,1:-2] = 255
        preprocessed.append(img)

    # バッチで一括推論
    results = model(preprocessed,
                    iou=0.3,
                    conf=0.5,
                    save=False,
                    exist_ok=True,
                )

    stones_list = []
    # 結果は入力順に並ぶ
    for i, (img, result) in enumerate(zip(preprocessed, results), start=1):
        # 中心座標リスト
        centers = []

        # 同じストーンが赤と黄の両方で検出された場合、誤った色の方を取り除く
        wrong = __find_wrong_color_boxes(
            img,
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int),
            result.boxes.conf.cpu().numpy(),
            f"{context} {i}枚目",
        )

        # バウンディングボックスから中心座標を計算
        for j, box in enumerate(result.boxes):
            if j in wrong:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()  # 左上(x1, y1), 右下(x2, y2)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            #座標をDCに変換
            cx = XA * cx - XB
            cy = YA * cy + YB
            dist = np.sqrt((cx - DC3_CENTER_X)**2 + (cy - DC3_TEE_LINE)**2)
            is_inhouse = int(dist <= DC3_RADIUS + DC3_STONE_RADIUS)
            is_insheet = 1
            cls_id = int(box.cls[0]) #赤が0, 黄色が1

            centers.append([cls_id, cx, cy, dist, is_inhouse, is_insheet])

        if not centers: #空リストのときには[0, 0, 0, 0, 0, 0]を追加しておく
            centers.append([0]*6)

        stones = np.array(centers)

        row = stones.shape[0]
        if row < 16:
            padding = np.zeros((16 - row, 6))
            stones = np.vstack([stones, padding]) #(16,6)

        stones_list.append(stones)

    return stones_list

def create_pseudo_label(model: YOLO, image_dir: Path, output_dir: Path, threshold: float = 0.8) -> int:
    """
        既存のモデルを用いて予測を行い、疑似ラベルを生成する
        Args:
            model : YOLOのモデル
            image_dir : 予測したい画像が格納されているディレクトリ名
            output_dir : ラベルの保存先
        Returns:
            int : 生成されたラベル数
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    imgs = [img for img in os.listdir(image_dir) if img.endswith(".png")]
    num_labels = 0
    for img_path in imgs:
        logger.debug(f"Generating pseudo label for: {img_path}")
        img_path = image_dir / img_path

        # 推論（OpenCV画像データを直接渡す）
        results = model.predict(
            source=img_path,                 
            iou=0.3,                    # NMS IoUしきい値
            conf=0.5,                   #信頼度しきい値
            save=False,                 # 結果画像を保存しない
            save_txt=False,
            exist_ok=True,
            verbose=False,    
        )
        boxes = results[0].boxes
        txt_data = []
        skip_flag = False
        for box in boxes:
            cls = int(box.cls[0])
            x, y, w, h = box.xywhn[0]
            conf = float(box.conf[0])
            if conf < threshold:   #全検出物体の確信度が閾値以上の画像を疑似ラベルとする
                skip_flag = True
                break
            txt_data.append((cls, 
                             round(x.item(), 6), 
                             round(y.item(), 6),
                             round(w.item(), 7), 
                             round(h.item(), 7),
                             ))
            
        if skip_flag: 
            img_path.unlink(missing_ok=True)
            continue
        else:
            # ファイル名を保存用に使う
            img_file = img_path.name
            txt_file = Path(img_file).with_suffix(".txt").name
            with open(output_dir / txt_file, "w") as f:
                for txt in txt_data:
                    line = " ".join(map(str, txt))
                    f.write(line + "\n")
            num_labels += 1
    
    return num_labels
        
if __name__ == "__main__":
    model = YOLO("complete_model/base.pt")
    create_pseudo_label(model, image_dir=Path("tmp/tmp2"), output_dir=Path("yolo_dataset"), threshold=0.75)