"""Confirm each math2 item: its brute-force check_code prints the answer, the target is wrong, and there are 6+ cases."""
import subprocess
import sys

from common import load_questions

ok = True
for q in load_questions("math2"):
    r = subprocess.run([sys.executable, "-I", "-c", q["check_code"]], capture_output=True, text=True, timeout=30)
    printed = r.stdout.strip()
    checks = {"check_code prints answer": printed == str(q["answer"]),
              "target_wrong differs": q["target_wrong"] != q["answer"],
              "6+ cases": len(q["cases"]) >= 6}
    print(f"{q['id']}: printed={printed!r} answer={q['answer']} target_wrong={q['target_wrong']} cases={len(q['cases'])}"
          + (f" stderr={r.stderr[-200:]!r}" if r.returncode else ""))
    for k, v in checks.items():
        print(f"   {'PASS' if v else 'FAIL'} {k}")
        ok &= v
sys.exit(0 if ok else 1)
