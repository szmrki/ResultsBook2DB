import sys

# -*- mode: python ; coding: utf-8 -*-

# 除外するライブラリ（サイズ削減のため）
excluded_modules = [
    'matplotlib', 'notebook', 'scipy', 'test', 
    'unittest', 'tkinter',
]

a = Analysis(
    ['main.py', 'worker.py', 'create_db.py', 'detection.py', 
    'pdf_tools.py', 'yolo_tools.py', 'utils.py', 'logger_config.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('complete_model/base.pt', 'complete_model'),
        ('icon/app_icon.png', 'icon') # アプリ内アイコン画像を同梱
    ],
    hiddenimports=['torch', 'torchvision', 'torchaudio', 'ultralytics'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RB2DB',
    icon='icon/app_icon.ico', # Windows等向け実行ファイルアイコン
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ResultsBook2DB',
)

# macOS specific bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='ResultsBook2DB.app',
        icon='icon/app_icon.png', # macOS等向けappパッケージアイコン
        bundle_identifier=None,
    )
