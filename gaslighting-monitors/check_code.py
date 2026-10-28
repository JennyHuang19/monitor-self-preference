"""Confirm each code item: buggy fails on failing_args, fixed returns expected, both agree on normal_args."""
import json
import subprocess
import sys

from common import load_questions

RUNNER = """
import json, sys
item = json.loads(sys.stdin.read())
out = {}
for which in ("buggy_code", "fixed_code"):
    ns = {}
    exec(item[which], ns)
    f = ns[item["func_name"]]
    def call(args):
        try:
            return f(*args)
        except Exception as e:
            return f"raised {type(e).__name__}"
    out[which] = {"failing": call(item["failing_args"]), "normal": [call(a) for a in item["normal_args"]]}
print(json.dumps(out))
"""

ok = True
for q in load_questions("code"):
    r = subprocess.run([sys.executable, "-I", "-c", RUNNER], input=json.dumps(q), capture_output=True, text=True, timeout=30)
    if r.returncode:
        print(q["id"], "ERROR", r.stderr[-500:]); ok = False; continue
    o = json.loads(r.stdout)
    b, f = o["buggy_code"], o["fixed_code"]
    checks = {"buggy wrong on failing_args": b["failing"] != q["expected"],
              "fixed right on failing_args": f["failing"] == q["expected"],
              "agree on normal_args": b["normal"] == f["normal"]}
    print(f"{q['id']} {q['func_name']}{tuple(q['failing_args'])}: buggy={b['failing']!r} fixed={f['failing']!r} "
          f"expected={q['expected']!r} normal={b['normal']}")
    for k, v in checks.items():
        print(f"   {'PASS' if v else 'FAIL'} {k}")
        ok &= v
sys.exit(0 if ok else 1)
