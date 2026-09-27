#!/usr/bin/env python3
"""The self-recording loop, end to end.

The claim made in AGENTS.md is that a battery now records itself: the runner
writes a sidecar next to the results AND one line into the repo's
provenance ledger, and the manifest picks that up as verified identity. That
was a claim, not a test. 48 of 59 old batteries are permanently unverifiable
precisely because this loop did not exist when they ran, so the loop being
quietly broken would be expensive and invisible.

This drives the REAL functions (protocol.write_meta, manifest.collect) over a
staging directory. No GPU, no network, and it never writes into eval_results
or into the real provenance ledger — both are redirected to /tmp for the
duration.

    python tests/test_provenance.py
"""
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import protocol
import manifest

N_TASKS = 164
PASSED = 151


def stage(dirpath, tag, passed, empties=8):
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, f"humaneval_{tag}.jsonl"), "w") as f:
        for i in range(N_TASKS):
            f.write(json.dumps({"task_id": f"HumanEval/{i}",
                                "completion": "" if i < empties else "x = 1"}) + "\n")
    with open(os.path.join(dirpath, f"humaneval_{tag}.jsonl_results.jsonl"), "w") as f:
        for i in range(N_TASKS):
            f.write(json.dumps({"task_id": f"HumanEval/{i}",
                                "completion": "x = 1", "result": "passed",
                                "passed": i < passed}) + "\n")


def main():
    tmp = tempfile.mkdtemp(prefix="mernik-provtest-")
    real_ledger = protocol.LEDGER
    fake_ledger = os.path.join(tmp, "batteries.jsonl")
    protocol.LEDGER = fake_ledger          # never touch the real ledger
    failures, ran = [], []

    def check(name, cond, detail=""):
        ran.append(name)
        print(f"{'ok  ' if cond else 'FAIL'}  {name}{'' if cond else '  ' + detail}")
        if not cond:
            failures.append(name)

    try:
        # 1. stage a battery exactly as a runner would, and stamp it the way
        #    scripts/run_humaneval.py does
        tag = "provtest_a"
        stage(tmp, tag, PASSED)
        out = os.path.join(tmp, f"humaneval_{tag}.jsonl")
        meta = protocol.battery_meta("http://127.0.0.1:28082",
                                     "/tmp/PROVENANCE-TEST.gguf")
        meta.update({"finished": "2026-09-28 22:00:00", "n_tasks": N_TASKS,
                     "empties": 8, "battery_min": 41.0, "s_per_task": 15.0})
        side = protocol.write_meta(out, meta)

        check("sidecar written next to the results", os.path.exists(side))
        check("provenance ledger appended", os.path.exists(fake_ledger))
        if os.path.exists(fake_ledger):
            line = json.loads(open(fake_ledger).read().strip().splitlines()[-1])
            check("ledger records the served model",
                  line.get("serve_model") == "/tmp/PROVENANCE-TEST.gguf",
                  repr(line.get("serve_model")))
            check("ledger records the sampling",
                  line.get("sample", {}).get("presence_penalty") == 0.0
                  and line.get("sample", {}).get("max_tokens") == 2048,
                  repr(line.get("sample")))

        # 2. the manifest must now call it verified, and never write to it
        rec = manifest.collect(tag, tmp)
        manifest.attach_provenance([rec], results_dir=tmp)
        check("manifest: identity verified from the sidecar",
              rec["identity"] == "verified", rec["identity"])
        check("manifest: served model carried through",
              rec["serve_model"] == "/tmp/PROVENANCE-TEST.gguf")
        check("manifest: counts read from the artefacts",
              rec["counts"].get("HE") == f"{PASSED}/{N_TASKS}",
              str(rec["counts"].get("HE")))

        # 3. an unstamped battery in the same directory must NOT be verified
        tag2 = "provtest_b"
        stage(tmp, tag2, 100)
        rec2 = manifest.collect(tag2, tmp)
        manifest.attach_provenance([rec2], results_dir=tmp)
        check("manifest: unstamped battery is not verified",
              rec2["identity"] != "verified", rec2["identity"])

    finally:
        protocol.LEDGER = real_ledger
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{len(ran) - len(failures)} ok, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
