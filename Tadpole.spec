# PyInstaller recipe for Tadpole (one-folder build, like Lillypad.spec).
#
#   .venv\Scripts\python scripts\build_julia_bundle.py      # Julia runtime + depot + env
#   julia --startup-file=no scripts\build_sysimage.jl        # optional, but see below
#   .venv\Scripts\pyinstaller Tadpole.spec --noconfirm
#
# → dist\Tadpole\Tadpole.exe, with the Julia bundle at _internal\julia_bundle.
#
# Without the sysimage the exe still works, but on a new machine the first
# launch recompiles FFTW and TadpoleRetrieval (~3–4 min, once, into
# %LOCALAPPDATA%\Tadpole\julia_depot): precompile caches do not survive the
# bundle being moved. With the sysimage it starts in seconds.
#
# The Julia bundle is ~1.3 GB, almost all of it the Julia runtime itself.

import os

from PyInstaller.utils.hooks import collect_data_files

BUNDLE = os.path.join('build', 'julia_bundle')
if not os.path.isdir(BUNDLE):
    raise SystemExit('No Julia bundle: run  .venv\\Scripts\\python '
                     'scripts\\build_julia_bundle.py  first.')
if not os.path.isfile(os.path.join(BUNDLE, 'sysimage', 'tadpole.dll')):
    print('WARNING: no sysimage in the bundle; the first launch on a new '
          'machine will recompile packages for several minutes.')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('icons', 'icons'),
           ('tadpole/design/qss', 'tadpole/design/qss'),
           (BUNDLE, 'julia_bundle')] + collect_data_files('juliacall')
          # the refractiveindex.info database ships inside the package
          + collect_data_files('refractiveindex'),
    # The worker is started by multiprocessing ("spawn") and imported by
    # name, so PyInstaller's import scan has to be told about it.
    hiddenimports=['tadpole.engine.julia_worker', 'juliacall'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6', 'PySide2', 'pytest', 'tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Tadpole',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icons/Tadpole.ico'] if os.path.exists('icons/Tadpole.ico') else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Tadpole',
)
