
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT      = Path(__file__).parent.parent.resolve()
DIST_DIR  = ROOT / "dist"
BUILD_DIR = ROOT / "build"
SPEC_FILE = ROOT / "packaging" / "build.spec"


def _header(msg: str) -> None:
    print(f"\n{'─' * 60}\n  {msg}\n{'─' * 60}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build AInsights .exe")
    p.add_argument("--skip-model", action="store_true",
                   help="Skip HuggingFace model download if already cached")
    p.add_argument("--no-upx",     action="store_true",
                   help="Skip UPX compression")
    p.add_argument("--debug",      action="store_true",
                   help="Build with console=True to see crash tracebacks")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def clean() -> None:
    _header("Cleaning previous build artefacts")
    for d in (DIST_DIR, BUILD_DIR):
        if d.exists():
            shutil.rmtree(d)
            print(f"  Removed: {d}")
    print("  Done.")


# ─────────────────────────────────────────────────────────────────────────────
def check_prerequisites() -> None:
    _header("Checking prerequisites")

    # PyInstaller
    r = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--version"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("  ✗  PyInstaller not found. Run: pip install pyinstaller")
        sys.exit(1)
    print(f"  ✓  PyInstaller {r.stdout.strip()}")

    v = sys.version_info
    if v < (3, 11):
        print(f"  ✗  Python {v.major}.{v.minor} is too old — use 3.11")
        sys.exit(1)
    if v >= (3, 13):
        print(f"  ⚠  Python {v.major}.{v.minor} — some wheels may be missing for 3.13")
        print("     If the build fails, try Python 3.11")
    else:
        print(f"  ✓  Python {v.major}.{v.minor}.{v.micro}")

    if shutil.which("upx"):
        print("  ✓  UPX found")
    else:
        print("  ⚠  UPX not found — https://upx.github.io/")

    try:
        import transformers
        import sentence_transformers
        print(f"  ✓  transformers {transformers.__version__}")
        print(f"  ✓  sentence_transformers {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"  ✗  {e}")
        print("     Run: pip install -r requirements.txt")
        sys.exit(1)

def pre_download_model() -> None:
    _header("Pre-downloading HuggingFace embedding model")
    print("  Model : all-MiniLM-L6-v2")
    print("  Downloading …")
    subprocess.run(
        [
            sys.executable, "-c",
            "from sentence_transformers import SentenceTransformer; "
            "m = SentenceTransformer('all-MiniLM-L6-v2'); "
            "print('  ✓  Model ready:', m)"
        ],
        check=True,
    )


def run_pyinstaller(no_upx: bool, debug: bool) -> None:
    _header("Running PyInstaller")

    spec_content = SPEC_FILE.read_text(encoding="utf-8")

    if debug:
        BUILD_DIR.mkdir(exist_ok=True)
        debug_spec = BUILD_DIR / "debug.spec"
        debug_spec.write_text(
            spec_content.replace("console=False", "console=True"),
            encoding="utf-8",
        )
        spec_to_use = debug_spec
        print("  ⚠  DEBUG BUILD — console window will be visible")
        print("     Crash tracebacks will appear in the CMD window")
    else:
        spec_to_use = SPEC_FILE

    cmd = [sys.executable, "-m", "PyInstaller", str(spec_to_use), "--clean"]
    if no_upx:
        cmd.append("--noupx")

    print(f"  {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print("\n  ✗  PyInstaller failed — see output above.")
        sys.exit(1)


def post_build() -> None:
    _header("Build report")

    exe    = DIST_DIR / "AInsights" / "AInsights.exe"
    folder = DIST_DIR / "AInsights"

    if not exe.exists():
        print("  ✗  AInsights.exe not found.")
        return

    exe_mb    = round(exe.stat().st_size / 1024 / 1024, 1)
    total_mb  = round(
        sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
        / 1024 / 1024, 1,
    )

    print(f"  ✓  AInsights.exe    {exe_mb} MB")
    print(f"  ✓  Total folder     {total_mb} MB")
    print(f"  ✓  Output: {folder}")
    print()
    print(textwrap.dedent("""\
        Distribution:
          1. Zip the entire dist/AInsights/ folder
          2. User unzips anywhere (e.g. C:\\Apps\\AInsights\\)
          3. User installs Ollama: https://ollama.com/download
          4. Double-click AInsights.exe

        First launch: model downloads automatically (~4 GB, once only)
        Subsequent launches: ~10-15 seconds to open

        If startup fails:
          %TEMP%\\ainsights_launcher.log  — full error traceback
          Rebuild with --debug flag to see errors in a console window
    """))


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  AInsights — Desktop Build Pipeline")
    print("=" * 60)

    clean()
    check_prerequisites()

    if args.skip_model:
        print("\n  ⚠  Skipping model download (--skip-model).")
    else:
        pre_download_model()

    run_pyinstaller(no_upx=args.no_upx, debug=args.debug)
    post_build()


if __name__ == "__main__":
    main()