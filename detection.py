from ultralytics import YOLO
import numpy as np
import cv2

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

def get_stones_pos(img, model) -> np.ndarray:
    """
        ストーン座標をモデルを用いて取得する
        Args:
            img_path : シート画像のファイルパス
        Returns:
            np.ndarray : (16 x 6)のストーン情報の配列
    """
    #img = cv2.imread(img_path)
    #必要であれば反転
    row20 = img[20,:,:]
    black_pixels = np.all(row20==0, axis=1) 
    if np.all(black_pixels[1:WIDTH]):  #左右1ピクセルが余白の可能性があるため
          img = cv2.flip(img, -1)

    #誤検出を防ぐため上下に白でマスク
    img[:20,1:-2] = 255
    img[-19:,1:-2] = 255  

    results = model(img, iou=0.3, conf=0.5) #modelに通す
                
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

def get_hammer_img(img_path, is_md=False, game=None) -> str: 
    """
        画像ベースで各エンドの1投目終了時にハンマーの色を取得するメソッド
        Args:
            img_path : シート画像のファイルパス
            is_md : MDのときはハンマーの取得方法が変わる
            year : MDのときは年によっても異なる
        Returns:
            str : "red" or "yellow"
    """
    img = cv2.imread(img_path)
    #必要であれば反転
    row20 = img[20,:,:]
    black_pixels = np.all(row20==0, axis=1) 
    if np.all(black_pixels[1:WIDTH]):  #左右1ピクセルが余白の可能性があるため
        img = cv2.flip(img, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #RGBに変換
    roi = img[-20:]

    red_bin = cv2.inRange(roi, (240,0,0), (255,0,0)) #赤を取得
    #red_area = np.count_nonzero(red_bin)
    red_area = cv2.countNonZero(red_bin)

    black_bin = cv2.inRange(roi, (0,0,0), (10,10,10)) #黒を取得
    white_bin =cv2.inRange(roi, (240,240,240), (255,255,255)) #白を取得

    mask_any = red_bin | black_bin | white_bin  # 赤・黒・白のピクセルを1にする
    mask_not = cv2.bitwise_not(mask_any)          # 反転して、対象外のピクセルを1に
    yellow_area = cv2.countNonZero(mask_not) #黄色の面積を赤・黒・白以外の面積とする

    #print(f"red: {red_area}, yellow: {yellow_area}")
    #MDの2017年以降は描かれ方が異なる
    if is_md and (game != "WMDCC2016"):
        if red_area >= yellow_area:
            return "yellow"
        else:
            return "red"
    else:        
        if red_area >= yellow_area:
            return "red"
        else:
            return "yellow"
        
if __name__ == "__main__":
    model = YOLO("complete_model/game10_2_retrain.pt")
    get_hammer_img("rb_data/data_md/WMDCC2024/WMDCC2024_ResultsBook-12_5.png", model)