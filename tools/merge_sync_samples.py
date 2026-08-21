#!/usr/bin/env python3
"""合并三个 GPU 评测实例的逐条 JSONL 明细，汇总 talkvid vs vivi 的 Sync-C/D 对比。"""

import json
import sys

import numpy as np


def main():
    files = [
        "/tmp/cmp_sync_gpu_report.txt",
        "/tmp/cmp_sync_gpu_b_report.txt",
        "/tmp/cmp_sync_gpu_c_report.txt",
    ]
    recs = []
    for fp in files:
        try:
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith("{"):
                        continue
                    recs.append(json.loads(line))
        except FileNotFoundError:
            pass

    by_tag = {"talkvid": [], "vivi": []}
    for r in recs:
        if r["ok"]:
            by_tag[r["tag"]].append(r)

    out = sys.stdout
    out.write(f"合并明细 {len(recs)} 条; 有效: talkvid {len(by_tag['talkvid'])}, "
              f"vivi {len(by_tag['vivi'])}\n\n")
    for tag in ("talkvid", "vivi"):
        ok = by_tag[tag]
        c = np.array([r["c"] for r in ok])
        d = np.array([r["d"] for r in ok])
        o = np.array([r["o"] for r in ok])
        out.write(f"[{tag}] n={len(c)}\n")
        out.write(f"  Sync-C (↑): mean {c.mean():.3f}  median {np.median(c):.3f}  "
                  f"std {c.std():.3f}  min {c.min():.3f}  max {c.max():.3f}\n")
        out.write(f"  Sync-D (↓): mean {d.mean():.3f}  median {np.median(d):.3f}  "
                  f"std {d.std():.3f}  min {d.min():.3f}  max {d.max():.3f}\n")
        out.write(f"  |offset| mean {np.abs(o).mean():.2f} 帧  "
                  f"(offset mean {o.mean():+.2f})\n\n")


if __name__ == "__main__":
    main()
