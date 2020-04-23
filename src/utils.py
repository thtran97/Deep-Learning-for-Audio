import torch
import librosa
import numpy as np
from itertools import groupby
from scipy.ndimage import gaussian_filter1d


def zcr_vad(y, shift=0.025, win_len=2048, hop_len=1024, threshold=0.005):
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if y.ndim == 2:
        y = y[0]
#     print(y.shape)
    zcr = librosa.feature.zero_crossing_rate(y + shift, win_len, hop_len)[0]
#     print("zcr : ", zcr.shape)
    activity = gaussian_filter1d(zcr, 1) > threshold
#     print(activity)
    activity = np.repeat(activity, len(y) // len(activity) + 1)
#     print(activity)
    activity = activity[:len(y)]
    return activity


def get_timestamp(activity):
#     print("Activity:", activity)
#     print(len(activity))
    mask = [k for k, _ in groupby(activity)]
#     print("Mask : ", mask)
#     print(len(mask))
    change = np.argwhere(activity[:-1] != activity[1:]).flatten()
#     print("Change:", change)
#     print(change.shape)
    span = np.concatenate([[0], change, [len(activity)]])
    span = list(zip(span[:-1], span[1:]))
#     print("Span: ", span)
    span = np.array(span)[mask]
#     print(span)
    print("Nb of detected activities :", len(span))
    return span