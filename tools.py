"""
tools.py は肥大化を防ぐため、以下の3つのモジュールに分割されました。
このファイルは、他のスクリプトからの既存の import 呼び出し（後方互換性）を維持するための「窓口（ファサード）」として機能します。

1. pdf_tools.py: PDFの解析・画像抽出処理
2. yolo_tools.py: YOLOの学習データ管理・YAML作成処理
3. utils.py: 汎用機能・スコア表整形処理
"""

from pdf_tools import *
from yolo_tools import *
from utils import *
