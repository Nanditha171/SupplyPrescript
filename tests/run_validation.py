import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "optimization_validation_report.json"

print("Running prescriptive optimization validation tests...\n")

result = subprocess.run(
    [
        sys.executable,
        "-m",
        "unittest",
        "tests.test_prescriptive_solver",
        "-v",
    ],
    cwd=ROOT,
    capture_output=True,
    text=True,
)

output = result.stdout + result.stderr
print(output)

passed = result.returncode == 0

report = {
    "project": "SupplyPrescript",
    "validation": "Prescriptive Optimization",
    "execution_time": datetime.now().isoformat(timespec="seconds"),
    "test_module": "tests.test_prescriptive_solver",
    "result": "PASSED" if passed else "FAILED",
    "exit_code": result.returncode,
    "test_cases": [
        "Regular supply selected when cheapest and feasible",
        "Air freight selected when regular supply is too slow",
        "Secondary supplier selected when cheapest and feasible",
        "No feasible solution when budget is too low",
        "No feasible solution when all options are too slow",
    ],
    "test_output": output,
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(json.dumps(report, indent=4), encoding="utf-8")

print(f"\nValidation result: {report['result']}")
print(f"Report saved to: {REPORT_PATH}")

if not passed:
    sys.exit(1)
