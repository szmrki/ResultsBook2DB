# ビルドガイド (Build Guide)

このドキュメントでは、ResultsBook2DB を各 OS (Windows, macOS, Linux) でビルドし、実行ファイルを作成する手順を説明します。

## 共通の前提条件

- Python 3.12 がインストールされていること
- パッケージマネージャー `uv` がインストールされていること
- ソースコードを以下のコマンドでクローンしていること
  ```bash
  git clone https://github.com/szmrki/ResultsBook2DB.git
  cd ResultsBook2DB
  ```

## PyTorch と GPU について

本プロジェクトでは `pyproject.toml` の `[tool.uv.sources]` 設定により、Windows / Linux 環境では **CUDA 12.8 対応の GPU 版 PyTorch** が `uv sync` 時に自動的にインストールされます。macOS では CPU 版が自動的にインストールされます。

### 動作要件

| OS | インストールされる PyTorch | GPU サポート | 備考 |
|----|------------------------|-------------|------|
| Windows / Linux | CUDA 12.8 版 | NVIDIA GPU (ドライバ **570.x 以上**) | `[tool.uv.sources]` で自動設定 |
| macOS (Apple Silicon) | デフォルト版 (MPS 対応) | Apple GPU (M1/M2/M3/M4) | PyPI 標準ホイールに MPS が内蔵 |

> **注意**: GPU を搭載していない環境でも、CUDA 版 PyTorch は CPU フォールバックで動作するため、ビルド・実行ともに問題ありません（推論速度は低下します）。

### NVIDIA ドライバのバージョン確認方法

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

ドライバが 570.x 未満の場合は、[NVIDIA ドライバダウンロードページ](https://www.nvidia.com/drivers) から最新版に更新してください。

### 異なる CUDA バージョンを使用する場合

古い GPU やドライバの制約で CUDA 12.8 が利用できない場合は、`pyproject.toml` 内のインデックス URL を変更してください：

```toml
# 例: CUDA 12.4 を使用する場合
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu124"  # cu128 → cu124 に変更
explicit = true
```

変更後、`uv sync` を再実行してください。

## 1. Windows でのビルド

Windows では、`.exe` ファイルを作成できます。

### 手順

1.  **プロジェクトの同期と仮想環境の構築**
    uv を用いて依存関係を一括でインストールします。実行すると自動的に `.venv` などの仮想環境が作成・同期されます。
    ```powershell
    uv sync
    ```

2.  **PyInstaller によるビルド**
    ```powershell
    uv run pyinstaller main_production.spec
    ```
    ビルドが完了すると、`dist\ResultsBook2DB` フォルダが生成されます。中の `RB2DB.exe` を実行して動作確認してください。  
    適宜ショートカットを作成してください。

## 2. macOS でのビルド

macOS では、`.app` アプリケーションバンドルを作成できます。

### 手順

1.  **プロジェクトの同期と仮想環境の構築**
    ```bash
    uv sync
    ```
    
2.  **PyInstaller によるビルド**
    ```bash
    uv run pyinstaller main_production.spec
    ```
    完了すると `dist` フォルダ内に `ResultsBook2DB.app` が生成されます。

## 3. Linux でのビルド

Ubuntu 等の Linux 環境向けの手順です。

### 手順

1.  **プロジェクトの同期と仮想環境の構築**
    ```bash
    uv sync
    ```

2.  **PyInstaller によるビルド**
    ```bash
    uv run pyinstaller main_production.spec
    ```
    完了すると `dist/ResultsBook2DB` フォルダが生成されます。
    - **本体ファイル**: `dist/ResultsBook2DB/RB2DB`

3.  **実行**
    ```bash
    ./dist/ResultsBook2DB/RB2DB
    ```

## アップデート時の手順

ソースコードや依存関係に更新があった場合は、以下の手順で再ビルドしてください。

### ローカルデータのバックアップ
再ビルド時、出力先の `dist/` フォルダは完全に削除されるため、学習済みの重みや解析結果を引き継ぎたい場合は、ビルド前に以下のフォルダを別の場所へ退避させ、ビルド完了後に元の場所へ戻してください。

- **`dist/ResultsBook2DB/_internal/complete_model/`** （独自の学習モデル）
- **`dist/ResultsBook2DB/runs/`**（FT結果の画像など）
- **`dist/ResultsBook2DB/logs/`** （ログファイル）

### ビルド実行手順

```bash
git pull
uv sync
uv run pyinstaller main_production.spec -y
```

`uv sync` は `uv.lock` に基づいて差分のみを更新するため、毎回フルインストールにはなりません。

