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

def get_stones_pos(img: np.ndarray, model: YOLO) -> np.ndarray:
    """
        ストーン座標をモデルを用いて取得する
        Args:
            img : シート画像のnumpy配列
            model : YOLOのモデル
        Returns:
            np.ndarray : (16 x 6)のストーン情報の配列
    """
    #必要であれば反転
    row20 = img[20,:,:]
    black_pixels = np.all(row20==0, axis=1) 
    if np.all(black_pixels[1:WIDTH]):  #左右1ピクセルが余白の可能性があるため
          img = cv2.flip(img, -1)

    #誤検出を防ぐため上下に白でマスク
    img[:20,1:-2] = 255
    img[-19:,1:-2] = 255  

    results = model(img, 
                    iou=0.3, 
                    conf=0.5, 
                    save=False,
                    exist_ok=True,
                ) #modelに通す
                
    # 中心座標リスト
    centers = []

    # バウンディングボックスから中心座標を計算
    for box in results[0].boxes:
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

    return stones

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