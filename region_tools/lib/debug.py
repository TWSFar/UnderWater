import numpy as np

width = 3840
height = 2160
stride = 600
side = 800

w_num = max(round((width - (side - stride)) / stride), 1)
w_len = np.ceil(width / w_num) + (side - stride)
h_num = max(round((height - (side - stride)) / stride), 1)
h_len = np.ceil(height / h_num) + (side - stride)

shift_w = np.arange(0, w_num) * stride
shift_h = np.arange(0, h_num) * stride

shift_w, shift_h = np.meshgrid(shift_w, shift_h)
shifts = np.vstack((
    shift_w.ravel(), shift_h.ravel(),
    shift_w.ravel(), shift_h.ravel()
)).transpose()
shifts[:, 2] = (shifts[:, 2] + w_len).clip(0, width)
shifts[:, 3] = (shifts[:, 3] + h_len).clip(0, height)
pass