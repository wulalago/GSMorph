import numpy as np


def intensity_scale(img):
    """
    Rescale the image to [0, 1]
    ----------------------------
    Parameters:
        img: [numpy.array; float; NxN] now is only support the grey image
    ----------------------------
    """
    return (img - np.min(img))/(np.max(img) - np.min(img))


def clip_intensity(img, interval):
    """
    Clip the intensity of an image
    ----------------------------
    Parameters:
        img: [numpy.array; float; NxN] now is only support the grey image
        interval: [list or tuple; ] the interval for clipping
    ----------------------------
    """
    cutoff = np.percentile(img, interval)
    img = np.clip(img, a_min=round(cutoff[0]), a_max=round(cutoff[1]))
    return img
