"""zlib-lossless (de)serialization for WAN-VAE latents.

VAE latent 接近去相关高斯,通用无损压缩省不了太多,但字节平面里高 8 位
(符号+指数)熵很低——实测 zlib 能压到 ~0.685 (2.37TB → 1.62TB),且解压快
(~200MB/s, 一条 <0.03s),不会拖慢 dataloader。lzma 更省 (1.32TB) 但解压
0.3s/条会成为新瓶颈,故用 zlib。

磁盘格式 (.ptz), 全部小端:
    magic  b"LATZ"                      4 bytes
    ndim   uint8                        1 byte
    shape  ndim × uint32                4*ndim bytes
    payload zlib.compress(bf16.view(uint16).tobytes())
latent 恒为 bf16;还原时按 shape reshape 回 [16, Tlat, 64, 64]。
"""
import struct
import zlib

import torch

_MAGIC = b"LATZ"


def save_latent_zlib(latent: torch.Tensor, path: str, level: int = 6):
    """latent: bf16 tensor [16, Tlat, 64, 64]. 原子写 (tmp + rename)."""
    assert latent.dtype == torch.bfloat16, f"expected bf16, got {latent.dtype}"
    latent = latent.cpu().contiguous()
    raw = latent.view(torch.uint16).numpy().tobytes()
    payload = zlib.compress(raw, level)
    header = _MAGIC + struct.pack("<B", latent.ndim)
    header += struct.pack("<%dI" % latent.ndim, *latent.shape)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(header)
        f.write(payload)
    import os
    os.replace(tmp, path)


def load_latent_zlib(path: str) -> torch.Tensor:
    """.ptz -> bf16 tensor [16, Tlat, 64, 64]."""
    with open(path, "rb") as f:
        blob = f.read()
    assert blob[:4] == _MAGIC, f"bad magic in {path}"
    ndim = blob[4]
    off = 5
    shape = struct.unpack_from("<%dI" % ndim, blob, off)
    off += 4 * ndim
    raw = zlib.decompress(blob[off:])
    flat = torch.frombuffer(bytearray(raw), dtype=torch.uint16)
    return flat.view(torch.bfloat16).reshape(shape)
