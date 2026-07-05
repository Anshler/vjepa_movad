""" This file is modified from:
https://raw.githubusercontent.com/piergiaj/pytorch-i3d/master/videotransforms.py
"""
import numpy as np
import cv2
import torch

from torchvision import transforms as T


def pad_frames(input_shape):
    def f(imgs):
        t, h, w, c = imgs.shape
        shape_r, shape_c = input_shape

        ims = np.zeros((t, shape_r, shape_c, c), dtype=np.float32)
        for i, im in enumerate(imgs):
            padded_image = padding(im, shape_r, shape_c, c)
            if c == 1:
                padded_image = np.expand_dims(padded_image, axis=-1)
            ims[i] = padded_image.astype(np.float32)
        return ims
    return f


def padding(img, shape_r=480, shape_c=640, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def pad_collate_videos(batch):
    """Collate function that pads variable-length videos to the max in the batch.

    Each sample is ``(video_data, data_info)`` where ``video_data`` has shape
    ``[T_i, C, H, W]``.  Videos are zero-padded to ``max(T_i)`` so they can be
    stacked into a single ``[B, max_T, C, H, W]`` tensor.
    """
    videos, infos = zip(*batch)
    max_len = max(v.shape[0] for v in videos)
    padded = []
    for v in videos:
        if v.shape[0] < max_len:
            pad = torch.zeros(max_len - v.shape[0], *v.shape[1:], dtype=v.dtype)
            v = torch.cat([v, pad], dim=0)
        padded.append(v)
    # data_info arrives as numpy arrays from Dota.__getitem__ — convert to tensors
    infos = [torch.as_tensor(info) for info in infos]
    return torch.stack(padded, dim=0), torch.stack(infos, dim=0)


class RandomVerticalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        coins = torch.rand(img.shape[0], device=img.device)
        indexes = (coins > self.p).nonzero().squeeze(1).tolist()
        img[indexes] = T.functional.vflip(img[indexes])
        return indexes, img


class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        coins = torch.rand(img.shape[0], device=img.device)
        indexes = (coins > self.p).nonzero().squeeze(1).tolist()
        img[indexes] = T.functional.hflip(img[indexes])
        return indexes, img
