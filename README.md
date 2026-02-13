# ResultsBook2DB

[CURLIT社](https://curlit.com/) が提供するカーリングの公式記録集 (Results Book)から試合データを抽出し、SQLite データベースに保存するツールです。

## 特徴

*   **PDF 解析**: Result Book (PDF形式) を読み込み、大会情報、試合結果、スコアを自動抽出します。
*   **ストーン検出**: 各大会に自動適応したYOLO モデルを使用して、Shot by Shot 画像からストーンの位置座標を検出します。
*   **データベース保存**: 抽出したデータを構造化された SQLite データベース (`.db`) に保存します。

## 必要要件

*   Python 3.12+ (動作確認済み)
*   推奨環境: GPU搭載 PC

## ビルド

実行ファイル (`.exe` 等) の作成方法については、[BUILD_GUIDE.md](BUILD_GUIDE.md) を参照してください。

## データ構造 (Database Schema)

生成される SQLite データベース (`.db`) は以下のテーブルで構成されています。

### `events` (大会情報)

| カラム名 | 説明 |
| :--- | :--- |
| `id` | 大会ID (主キー) |
| `name` | 大会名 (ユニーク属性) |

### `games` (試合情報)

| カラム名 | 説明 |
| :--- | :--- |
| `id` | 試合ID (主キー) |
| `event_id` | 大会ID (外部キー: `events.id`) |
| `page` | PDF内の掲載ページ |
| `team_red` | 赤ストーンチーム名 |
| `team_yellow` | 黄ストーンチーム名 |
| `final_score_red` | 赤ストーンチームの最終スコア |
| `final_score_yellow` | 黄ストーンチームの最終スコア |

### `ends` (エンド情報)

| カラム名 | 説明 |
| :--- | :--- |
| `id` | エンドID (主キー) |
| `game_id` | 試合ID (外部キー: `games.id`) |
| `page` | PDF内の掲載ページ |
| `number` | エンド番号 |
| `color_hammer` | ラストストーン（ハンマー）のチームカラー |   
| `score_red` | このエンドの赤ストーンチーム得点 |
| `score_yellow` | このエンドの黄ストーンチーム得点 |

### `shots` (ショット情報)

| カラム名 | 説明 |
| :--- | :--- |
| `id` | ショットID (主キー) |
| `end_id` | エンドID (外部キー: `ends.id`) |
| `number` | ショット番号 (1-16) |
| `color` | ショットを投げたチームカラー |
| `team` | チーム名 |
| `player_name` | 選手名 |
| `type` | ショットの種類 |
| `turn` | 回転方向 (cw, ccw) |
| `percent_score` | ショットスコア(0, 25, 50, 75, 100) |

### `stones` (ストーン配置情報)

| カラム名 | 説明 |
| :--- | :--- |
| `id` | ストーンID (主キー) |
| `shot_id` | ショットID (外部キー: `shots.id`) |
| `color` | ストーンのカラー |
| `x` | X座標 (DigitalCurling3 準拠) |
| `y` | Y座標 (DigitalCurling3 準拠) |
| `distance_from_center` | ハウス中心からの距離 |
| `inhouse` | ハウス内判定 (1: 内, 0: 外) |
| `insheet` | シート内判定 (通常は1) |

## 座標系 (Coordinate System)

本システムは **[DigitalCurling3](https://github.com/digitalcurling/DigitalCurling3)** の座標系に準拠しています。

*   **単位**: メートル (m)
*   **原点**: デリバリー側のハック中心
*   **X軸**: センターラインから左右に ±2.375
*   **Y軸**: センターライン方向 (デリバリー側からハウス方向が正)
    *   ティーライン (Tee Line): y = 38.405
    *   バックライン (Back Line): y = 40.234
*   **サイズ情報**:
    *   ハウス半径: 1.829
    *   ストーン半径: 0.145

## ライセンス

本ソフトウェアは **GNU Affero General Public License v3.0 (AGPL-3.0)** の下で公開されています。
詳細は [LICENSE](LICENSE) ファイルを参照してください。

### 使用ライブラリ

本システムは以下の主要ライブラリを使用して開発されました。

*   **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)**
*   **[PyMuPDF (fitz)](https://github.com/pymupdf/PyMuPDF)**
*   **[PySide6](https://pypi.org/project/PySide6/)**
*   **[pdfplumber](https://github.com/jsvine/pdfplumber)**
*   **[OpenCV](https://opencv.org/)**

Copyright (C) 2026 szmrki
