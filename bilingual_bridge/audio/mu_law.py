import numpy as np

MU = 255.0

# PCM16 [-32768, 32767] <-> mu-law 8-bit (0-255)


def pcm16_to_mulaw(pcm16: np.ndarray) -> np.ndarray:
    x = np.clip(pcm16.astype(np.float32) / 32768.0, -1.0, 1.0)
    s = np.sign(x)
    y = s * np.log1p(MU * np.abs(x)) / np.log1p(MU)
    mulaw = ((y + 1) / 2 * MU + 0.5).astype(np.uint8)
    return mulaw




def mulaw_to_pcm16(mulaw: np.ndarray) -> np.ndarray:
    y = (mulaw.astype(np.float32) / MU) * 2 - 1
    x = np.sign(y) * (1 / MU) * ((1 + MU) ** np.abs(y) - 1)
    pcm = np.clip(x * 32768.0, -32768, 32767).astype(np.int16)
    return pcm