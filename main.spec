# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py', 'worker.py', 'create_db.py', 'tools.py', 'detection.py'],
    pathex=[],
    binaries=[],
    datas=[('complete_model/base.pt', 'complete_model')],
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
    name='ResultsBook2DB',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    name='',
)
