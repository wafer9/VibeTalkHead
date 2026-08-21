"""阶段一 (syncnet 环境): 对 gen/GT 视频逐帧跑 s3fd, 缓存每条视频的人脸框。

FID/FVD 前要把 gen/GT 都裁到人脸区域消除构图差异, 且必须同一套检测器。s3fd 在
syncnet 环境, I3D/torch_fidelity 在 vibe 环境, 故拆两阶段: 这里只出框缓存 json,
eval_fid_fvd.py --face_crop 读它裁人脸。

对每条视频: 逐帧 s3fd 检测, 每帧取置信度最高的一个框 (talking-head 单人);
无框的帧记 null (裁剪时回退到相邻帧框或整帧中心)。为省时可 --stride 隔帧检测,
中间帧用最近的检测框 (人脸位置帧间近乎不变)。

输出 json: { "<视频绝对路径>": [[x1,y1,x2,y2] 或 null, ... 每帧一个], ... }

用法 (conda activate syncnet):
    CUDA_VISIBLE_DEVICES=6 python extract_faceboxes.py --limit 5
    CUDA_VISIBLE_DEVICES=6 python extract_faceboxes.py            # 全量
"""
import argparse
import glob
import json
import os

os.environ.setdefault("TMPDIR", "/dev/shm/facebox_tmp")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

import cv2
import numpy as np
import torch
from syncnet_python.detectors.s3fd import S3FD
from syncnet_python.detectors.s3fd.nets import S3FDNet

GEN_DIR = "/nfs-speech-cfs/wangzhou/s2s/vibehead/exp/s4_1p7_vivid_5e4/infer_hdtf100_14_3.0"
GEN_DIR = "/nfs-speech-cfs/wangzhou/s2s/vibehead/exp/s5_1p7_all/infer_hdtf_6_3.0"
GT_DIR = "/nfs-speech-cfs/wangzhou/s2s/vibehead/data/hdtf/clips"
S3FD_W = "/nfs-speech-cfs/wangzhou/s2s/data/syncnet_python/detectors/s3fd/weights/sfd_face.pth"
OUT = "/dev/shm/fidfvd/faceboxes.json"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_dir", default=GEN_DIR)
    p.add_argument("--gt_dir", default=GT_DIR)
    p.add_argument("--s3fd_weights", default=S3FD_W)
    p.add_argument("--out", default=OUT)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--stride", type=int, default=3,
                   help="每隔几帧检一次, 中间帧沿用最近检测框 (人脸帧间几乎不动)")
    p.add_argument("--scale", type=float, default=0.25, help="s3fd 检测缩放 (同 pipeline)")
    p.add_argument("--conf", type=float, default=0.9)
    return p.parse_args()


def load_s3fd(weights, device="cuda"):
    net = S3FDNet(device=device)
    net.load_state_dict(torch.load(weights, map_location=device))
    net.eval()
    return S3FD(net=net, device=device)


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def biggest_box(boxes):
    """s3fd 返回 (N,5) [x1,y1,x2,y2,conf]; 取面积最大的框 (主说话人)。"""
    if boxes is None or len(boxes) == 0:
        return None
    b = np.asarray(boxes)
    areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    x1, y1, x2, y2 = b[int(np.argmax(areas)), :4]
    return [float(x1), float(y1), float(x2), float(y2)]


@torch.no_grad()
def video_boxes(s3fd, path, stride, scale, conf):
    frames = read_frames(path)
    n = len(frames)
    boxes = [None] * n
    last = None
    for i in range(0, n, stride):
        b = biggest_box(s3fd.detect_faces(frames[i], conf_th=conf, scales=[scale]))
        if b is not None:
            last = b
        boxes[i] = b if b is not None else last
    # 用最近检测框补齐未检测的帧
    last = None
    for i in range(n):
        if boxes[i] is not None:
            last = boxes[i]
        else:
            boxes[i] = last
    # 开头就 None 的, 用后面第一个有效框回填
    if boxes and boxes[0] is None:
        nxt = next((b for b in boxes if b is not None), None)
        boxes = [b if b is not None else nxt for b in boxes]
    return boxes


def list_pairs(gen_dir, gt_dir, limit):
    gens = sorted(v for v in glob.glob(os.path.join(gen_dir, "*.mp4"))
                  if not v.endswith(".noaudio.mp4"))
    pairs = []
    for gv in gens:
        gt = os.path.join(gt_dir, os.path.basename(gv))
        if os.path.exists(gt):
            pairs.append((gv, gt))
    return pairs[:limit] if limit > 0 else pairs


def main():
    args = get_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    s3fd = load_s3fd(args.s3fd_weights, "cuda")
    pairs = list_pairs(args.gen_dir, args.gt_dir, args.limit)
    print(f"[box] {len(pairs)} 对视频, stride={args.stride}", flush=True)

    cache = {}
    for k, (gv, gt) in enumerate(pairs):
        for path in (gv, gt):
            cache[path] = video_boxes(s3fd, path, args.stride, args.scale, args.conf)
        n_none = sum(1 for b in cache[gv] if b is None) + sum(1 for b in cache[gt] if b is None)
        print(f"[{k+1:3d}/{len(pairs)}] {os.path.basename(gv):45s} "
              f"gen帧{len(cache[gv])} gt帧{len(cache[gt])} 未检出{n_none}", flush=True)

    with open(args.out, "w") as f:
        json.dump(cache, f)
    print(f"\n[box] 写出 {len(cache)} 条视频的框 -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
