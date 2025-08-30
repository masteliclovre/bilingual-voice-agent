import numpy as np
from scipy.signal import resample_poly




def resample_pcm16(pcm16: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
    if src_hz == dst_hz:
        return pcm16
    # Use polyphase for good quality and speed (e.g., 24000 -> 8000: up=1, down=3)
    gcd = np.gcd(src_hz, dst_hz)
    up = dst_hz // gcd
    down = src_hz // gcd
    out = resample_poly(pcm16.astype(np.float32), up, down)
    return np.clip(out, -32768, 32767).astype(np.int16)