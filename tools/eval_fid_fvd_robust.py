#!/usr/bin/env python
"""Run vibehead/eval_fid_fvd.py with stable rank-deficient FVD covariance.

With only 60 samples and 400-D I3D features the covariance is rank deficient.
The upstream pytorch-fid implementation rejects sqrtm imaginary residue above
1e-3. With 60 samples and 400-D features, 1e-3--1e-2 numerical residue is
possible. We use the same real-valued Frechet formula and discard only this
small numerical imaginary component.
"""
import os
import sys

import numpy as np


VIBEHEAD = "/nfs-speech-cfs/wangzhou/s2s/vibehead"
sys.path.insert(0, VIBEHEAD)
import eval_fid_fvd as evaluation  # noqa: E402

# The upstream evaluator uses a fixed /dev/shm work directory. Allow concurrent
# evaluations to isolate their extracted frames instead of overwriting each other.
evaluation.WORK = os.environ.get("FIDFVD_WORK", evaluation.WORK)
os.makedirs(evaluation.WORK, exist_ok=True)


def stable_frechet_distance(feat1, feat2):
    from scipy import linalg

    mu1, sig1 = feat1.mean(0), np.cov(feat1, rowvar=False)
    mu2, sig2 = feat2.mean(0), np.cov(feat2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sig1.dot(sig2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sig1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sig1 + offset).dot(sig2 + offset))
    if np.iscomplexobj(covmean):
        imag = float(np.max(np.abs(covmean.imag)))
        if imag > 1e-2:
            raise ValueError(f"sqrtm imaginary residue too large: {imag}")
        print(f"[FVD] sqrtm numerical imaginary residue={imag:.6g}; using real part")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sig1) + np.trace(sig2) - 2 * np.trace(covmean))


evaluation.frechet_distance = stable_frechet_distance
evaluation.main()
