import os
import shutil
import random
import yaml
import logging
from pathlib import Path
from utils import delete_files

logger = logging.getLogger(__name__)

def split_train_val(image_dir: Path, label_dir: Path, train_ratio: float = 0.8, seed: int = 42) -> None:
    """
        画像とラベルを訓練用と検証用に分割する
        Args:
            image_dir : 画像のパス
            label_dir : ラベルのパス
            train_ratio : 訓練データの割合(0~1)
            seed : 乱数用のシード
    """
    random.seed(seed)

    # 出力フォルダ作成
    train_img_dir = image_dir / "train"
    val_img_dir = image_dir / "val"
    train_lbl_dir = label_dir / "train"
    val_lbl_dir = label_dir / "val"

    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    #画像フォルダ初期化
    delete_files(train_img_dir)
    delete_files(train_lbl_dir)
    delete_files(val_img_dir)
    delete_files(val_lbl_dir)

    # 画像一覧取得（.png限定）
    images = [f for f in os.listdir(image_dir) if f.endswith(".png")]

    for img_name in images:
        base = os.path.splitext(img_name)[0]
        lbl_name = base + ".txt"
        img_path = image_dir / img_name
        lbl_path = label_dir / lbl_name

        # ラベルが無い場合はスキップ
        if not lbl_path.exists():
            logger.warning(f"ラベルが無いためスキップ: {img_name}")
            continue

        # train or val に振り分け
        if random.random() < train_ratio:
            dst_img = train_img_dir
            dst_lbl = train_lbl_dir
        else:
            dst_img = val_img_dir
            dst_lbl = val_lbl_dir

        # ファイル移動
        shutil.move(img_path, dst_img / img_name)
        shutil.move(lbl_path, dst_lbl / lbl_name)

def create_yaml(
        save_path: Path,
        dataset_root: Path,
        train_path: Path = Path("images/train"),
        val_path: Path = Path("images/val"),
        ) -> None:
    """
        YOLOの学習用のyamlファイルを作成する
        Args: 
            save_path : 保存ファイル名
            dataset_root : データのパス
            train : 訓練画像のパス
            val : 検証画像のパス
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    names = ["red", "yellow"]

    yaml_dict = {
        "path": str(dataset_root),
        "train": str(train_path),
        "val": str(val_path),
        "names": {i: name for i, name in enumerate(names)}
    }

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, allow_unicode=True)
