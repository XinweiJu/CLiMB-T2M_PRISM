# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image, ImageEnhance  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
import re
import torch.nn.functional as F


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 load_depth=False,
                 is_train=False,
                 img_ext='.png',
                 data = 'c3vd',
                 motion_aux='none',
                 motion_root=None,
                 shading_root=None,
                 edge_root=None):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.LANCZOS
        self.data = data
        self.motion_aux = motion_aux
        self.motion_root = motion_root
        self.shading_root = shading_root
        self.edge_root = edge_root
        if self.motion_aux != 'none' and not self.motion_root:
            raise ValueError("motion_root is required when motion_aux is enabled")

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)

        self.resize_color = {}
        self.resize_edge = {}

        for i in range(self.num_scales):
            s = 2 ** i
            size = (self.height // s, self.width // s)

            # color_aug 用 LANCZOS（PIL only）
            self.resize_color[i] = size
            self.resize_edge[i] = size


        self.load_depth = load_depth #self.check_depth()
        print("self.data:", self.data)

    def preprocess(self, inputs, color_aug):
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    height, width = self.resize_color[i]
                    inputs[(n, im, i)] = inputs[(n, im, i - 1)].resize(
                        (width, height), Image.LANCZOS)
            if "edge" in k or "lum" in k or "motion" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    height, width = self.resize_edge[i]
                    source = inputs[(n, im, i - 1)]
                    inputs[(n, im, i)] = F.interpolate(
                        source.unsqueeze(0),
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.pil_to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.pil_to_tensor(color_aug(f))

    @staticmethod
    def pil_to_tensor(image):
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(array)).div_(255.0)

    def color_jitter(self, image):
        image = ImageEnhance.Brightness(image).enhance(
            random.uniform(*self.brightness))
        image = ImageEnhance.Contrast(image).enhance(
            random.uniform(*self.contrast))
        return ImageEnhance.Color(image).enhance(
            random.uniform(*self.saturation))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        if len(line) == 1:
            #/Datasets/C3VD_Undistorted/Dataset/cecum_t1_a/0021_color.png
            # Extract the filename from the path
            full_image_path = line[0]
            folder_full = os.path.dirname(full_image_path)                      # /Datasets/C3VD_Undistorted/Dataset/cecum_t1_a
            folder_name = os.path.basename(folder_full)                         # cecum_t1_a
            filename = os.path.basename(full_image_path).split(".")[0]         # 0021_color
            # Use regular expression to find the first sequence of digits
            match = re.search(r'\d+', filename)

            # Convert the found sequence to an integer
            if match:
                frame_index = int(match.group())
            else:
                raise ValueError(f"No digits found in filename: {filename}")

            side = None
        else:
            folder = line[0]

            if len(line) == 3:
                frame_index = int(line[1])
            else:
                frame_index = 0

            if len(line) == 3:
                side = line[2]
            else:
                side = None

        for i in self.frame_idxs:
            # if i == "s":
            #     other_side = {"r": "l", "l": "r"}[side]
            #     inputs[("color", i, -1)] = self.get_color(folder_full, frame_index, other_side, do_flip)
            # else:
            #
            inputs[("color", i, -1)] = self.get_color(folder_full, frame_index + i, side, do_flip)

            if self.data == 'c3vd':
                inputs[("edge", i, -1)] = self.get_edge(folder_name, frame_index + i)
                inputs[("lum", i, -1)] = self.get_lum(folder_name, frame_index + i)
            elif self.data == 'hk':
                inputs[("edge", i, -1)] = self.get_edge_hk(folder_name, frame_index + i)
                inputs[("lum", i, -1)] = self.get_lum_hk(folder_name, frame_index + i)
            elif self.data == 'endomapper':
                inputs[("edge", i, -1)] = self.get_edge_endomapper(folder_name, frame_index + i)
                inputs[("lum", i, -1)] = self.get_lum_endomapper(folder_name, frame_index + i)
            if self.motion_aux != 'none':
                digits = 4 if self.data == 'c3vd' else (6 if self.data == 'endomapper' else 5)
                motion_path = os.path.join(
                    self.motion_root, self.motion_aux, folder_name,
                    ("{:0" + str(digits) + "d}.png").format(frame_index + i))
                if os.path.exists(motion_path):
                    motion = np.asarray(Image.open(motion_path).convert("L"), dtype=np.float32) / 255.0
                else:
                    motion = np.zeros(inputs[("color", i, -1)].size[::-1], dtype=np.float32)
                inputs[("motion", i, -1)] = torch.from_numpy(motion).unsqueeze(0)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = self.color_jitter
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        # print("✅ color_aug: ",
        #     inputs[("color_aug", 0, 0)].shape,
        #     inputs[("color_aug", 0, 0)].dtype,
        #     torch.min(inputs[("color_aug", 0, 0)]).item(),
        #     torch.max(inputs[("color_aug", 0, 0)]).item())

        # print("✅ edge: ",
        #     inputs[("edge", 0, 0)].shape,
        #     inputs[("edge", 0, 0)].dtype,
        #     torch.min(inputs[("edge", 0, 0)]).item(),
        #     torch.max(inputs[("edge", 0, 0)]).item())

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
            del inputs[("edge", i, -1)]
            if self.motion_aux != 'none':
                del inputs[("motion", i, -1)]
            # del inputs[("lum", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder_full, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    # def check_depth(self):
    #     raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    # def get_edge(self, folder, frame_index):
    #     raise NotImplementedError
