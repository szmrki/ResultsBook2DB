# ビルドガイド (Build Guide)

このドキュメントでは、ResultsBook2DB を各 OS (Windows, macOS, Linux) でビルドし、実行ファイルを作成する手順を説明します。

## 共通の前提条件

- Python 3.12 がインストールされていること
- ソースコードを以下のコマンドでクローンしていること
  ```bash
  git clone https://github.com/szmrki/ResultsBook2DB.git
  cd ResultsBook2DB
  ```

## 1. Windows でのビルド

Windows では、`.exe` ファイルを作成できます。

### 手順

1.  **仮想環境の作成と有効化** (PowerShell)
    ```powershell
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    ```

2.  **PyTorch (GPU版) のインストール (推奨)**
    CUDA対応のGPUを使用する場合、`requirements.txt` の前に PyTorch 公式サイトから適切なバージョンをインストールしてください。
    例 (CUDA 11.8の場合):
    ```powershell
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    ※ CPUのみの場合はこのステップをスキップしても構いませんが、動作が遅くなる可能性があります。

3.  **依存ライブラリのインストール**
    ```powershell
    pip install -r requirements.txt
    ```

4.  **PyInstaller によるビルド**
    ```powershell
    pyinstaller main_production.spec
    ```
    ```
    ビルドが完了すると、`dist\ResultsBook2DB` フォルダが生成されます。中の `ResultsBook2DB.exe` を実行して動作確認してください。

## 2. macOS でのビルド

macOS では、`.app` アプリケーションバンドルを作成できます。
※ Apple Silicon (M1/M2/M3) 環境では、ネイティブ対応のターミナルを使用してください。

### 手順

1.  **仮想環境の作成と有効化**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **依存ライブラリのインストール**
    ```bash
    pip install -r requirements.txt
    ```
    ※ macOS (Apple Silicon) では、通常の `pip install` で Metal (MPS) 対応の PyTorch がインストールされます。

3.  **PyInstaller によるビルド**
    ```bash
    pyinstaller main_production.spec
    ```
    完了すると `dist` フォルダ内に `ResultsBook2DB.app` が生成されます。

### 注意点
- セキュリティ設定により、初回起動時に「開発元を検証できない」等の警告が出ることがあります。「システム設定」→「プライバシーとセキュリティ」から許可するか、Ctrlキーを押しながらクリックして「開く」を選択してください。

## 3. Linux でのビルド

Ubuntu 等の Linux 環境向けの手順です。

### 手順

1.  **仮想環境の作成と有効化**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **PyTorch (GPU版) のインストール (推奨)**
    GPUを使用する場合、事前にインストールしてください。
    例 (CUDA 11.8):
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

3.  **依存ライブラリのインストール**
    ```bash
    pip install -r requirements.txt
    ```

4.  **PyInstaller によるビルド**
    ```bash
    pyinstaller main_production.spec
    ```
    完了すると `dist/ResultsBook2DB` フォルダが生成されます。

5.  **実行**
    ```bash
    ./dist/ResultsBook2DB/ResultsBook2DB
    ```
