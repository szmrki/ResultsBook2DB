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

## 1. Windows でのビルド

Windows では、`.exe` ファイルを作成できます。

### 手順

1.  **プロジェクトの同期と仮想環境の構築**
    uv を用いて依存関係を一括でインストールします。実行すると自動的に `.venv` などの仮想環境が作成・同期されます。
    ```powershell
    uv sync
    ```

2.  **PyTorch (GPU版) のインストール (推奨)**
    CUDA対応のGPUを使用する場合、PyTorch 公式サイトから適切なバージョンで上書きインストールしてください。
    例 (CUDA 11.8の場合):
    ```powershell
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    ※ CPUのみの場合はこのステップをスキップしても構いませんが、動作が遅くなる可能性があります。

3.  **PyInstaller によるビルド**
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

2.  **PyTorch (GPU版) のインストール (推奨)**
    GPUを使用する場合、事前にインストールしてください。
    例 (CUDA 11.8):
    ```bash
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

3.  **PyInstaller によるビルド**
    ```bash
    uv run pyinstaller main_production.spec
    ```
    完了すると `dist/ResultsBook2DB` フォルダが生成されます。
    - **本体ファイル**: `dist/ResultsBook2DB/RB2DB`

5.  **実行**
    ```bash
    ./dist/ResultsBook2DB/RB2DB
    ```
