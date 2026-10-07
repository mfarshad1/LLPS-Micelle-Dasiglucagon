from pathlib import Path
import subprocess
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "analysis" / "charge_mapping"
OUT = ROOT / "reproduced_figures" / "figure3_charge_mapping.pdf"

result = SRC / "Charge_pH.pdf"
if result.exists():
    result.unlink()

subprocess.run(
    [sys.executable, "charge5-new.py"],
    cwd=SRC,
    check=True
)

if not result.exists():
    raise RuntimeError("charge5-new.py did not create Charge_pH.pdf")

shutil.copy2(result, OUT)
print(f"Saved {OUT}")
