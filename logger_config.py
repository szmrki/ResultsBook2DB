import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable

def setup_logging(log_dir: str = "logs", log_file: str = "app.log") -> Path:
    """
    ロギングの基本設定を行う。
    - コンソール出力 (DEBUG以上)
    - ファイル出力 (INFO以上, ローテーションあり)
    Args:
        log_dir: ログ保存ディレクトリ
        log_file: ログファイル名
    Returns:
        Path: ログファイルのパス
    """
    # ログ保存ディレクトリの作成
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    log_file_path = log_path / log_file

    # ロガーのルート設定
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 外部ライブラリのログを抑制 (pdfminer, ultralytics など)
    logging.getLogger("pdfminer").setLevel(logging.WARNING)
    #logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # フォーマットの設定 (モジュール名を少し短縮して表示)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)-10s - %(levelname)-8s - %(message)s'
    )

    # Console Handler (出力先: 標準出力)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (出力先: ファイル, 5MB毎にローテーション, 最大5個まで)
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logging.info("Logging initialized. Log file: %s", log_file_path)
    return log_file_path

class QtLogHandler(logging.Handler):
    """
    PythonのloggingメッセージをQtのシグナル経由でGUIに送るためのハンドラ
    """
    def __init__(self, slotfunc: Callable[[str, int], None]) -> None:
        super().__init__()
        self.slotfunc = slotfunc # MainWindow.log_write 相当の関数を受け取る

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        # ログレベルに応じた色情報の付与などを検討できるが、まずはメッセージを直接送る
        self.slotfunc(msg, record.levelno)

def add_qt_handler(slotfunc: Callable[[str, int], None], log_file_path: Path | None = None) -> None:
    """
    既存のロガーにQtハンドラを追加する。
    log_file_path: ログファイルパス（ログ出力用。Noneの場合は出力しない）
    """
    logger = logging.getLogger()
    handler = QtLogHandler(slotfunc)
    
    # GUI用のフォーマット（ファイル用より少しシンプルに）
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    logging.info("Qt Log Handler added.")
    if log_file_path is not None:
        logging.info("Logging initialized. Log file: %s", log_file_path)

if __name__ == "__main__":
    # テスト用
    setup_logging()
    logging.debug("This is a debug message")
    logging.info("This is an info message")
    logging.warning("This is a warning message")
    logging.error("This is an error message")
    logging.critical("This is a critical message")
