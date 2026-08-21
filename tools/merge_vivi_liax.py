"""合并 16 个分片 json -> data/vivi/train_liax.json, 并核对覆盖率。

用法: python merge_vivi_liax.py
每条以 sample_id 去重; 校验每条 motion_pt_path 存在。报告缺失/未覆盖数量。
"""
import glob
import json
import os

OUT_ROOT = "/nfs-speech-cfs/wangzhou/data/tts/VividHead/lia-x"
JSON_DIR = os.path.join(OUT_ROOT, "json_shards")
LIST = "/nfs-speech-cfs/wangzhou/s2s/vibehead/data/vivi/train.list"
FINAL = "/nfs-speech-cfs/wangzhou/s2s/vibehead/data/vivi/train_liax.json"


def main():
    shard_files = sorted(glob.glob(os.path.join(JSON_DIR, "train_liax.shard*.json")))
    print(f"[merge] 发现 {len(shard_files)} 个分片 json")

    by_id = {}
    for sf in shard_files:
        n = 0
        with open(sf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                by_id[r["sample_id"]] = r
                n += 1
        print(f"    {os.path.basename(sf)}: {n} 行")

    # 校验 .pt 存在
    missing_pt = [r for r in by_id.values() if not os.path.exists(r["motion_pt_path"])]

    # 与 train.list 全集对比覆盖率
    with open(LIST) as f:
        all_keys = {f"vivi_{json.loads(l)['key']}" for l in f if l.strip()}
    covered = set(by_id.keys())
    uncovered = all_keys - covered

    records = sorted(by_id.values(), key=lambda r: r["sample_id"])
    with open(FINAL, "w") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[merge] 写出 {len(records)} 行 -> {FINAL}")
    print(f"[merge] train.list 全集 {len(all_keys)} 条, 已覆盖 {len(covered)}, "
          f"未覆盖 {len(uncovered)}")
    print(f"[merge] motion .pt 缺失(json 有但文件不在): {len(missing_pt)}")
    if uncovered:
        print("        未覆盖示例:", list(uncovered)[:10])
    if missing_pt:
        print("        缺失 .pt 示例:", [r["sample_id"] for r in missing_pt][:10])


if __name__ == "__main__":
    main()
