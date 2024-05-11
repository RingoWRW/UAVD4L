import argparse
import os
from typing import Union, Optional, Dict, List, Tuple
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
import collections.abc as collections
import torch
import cv2
import numpy as np
import h5py
import imagesize
from .utils.parsers import parse_image_lists

class ImageDataset(torch.utils.data.Dataset):
    default_conf = {
        'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
        'grayscale': True,
        'resize_max': None,
        'resize_force': False,
        'interpolation': 'cv2_area',  # pil_linear is more accurate but slower
    }

    def __init__(self, root, paths=None):
        self.conf = self.default_conf
        self.root = root
        self.globs = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
        if paths is None:
            paths = []
            for g in self.globs:
                paths += list(Path(root).glob('**/'+g))
            if len(paths) == 0:
                raise ValueError(f'Could not find any image in root: {root}.')
            paths = sorted(list(set(paths)))
            self.names = [i.relative_to(root).as_posix() for i in paths]
        else:
            if isinstance(paths, (Path, str)):
                self.names = parse_image_lists(paths)
            elif isinstance(paths, collections.Iterable):
                self.names = [p.as_posix() if isinstance(p, Path) else p
                              for p in paths]
            else:
                raise ValueError(f'Unknown format for path argument {paths}.')

            for name in self.names:
                if not (root / name).exists():
                    raise ValueError(
                        f'Image {name} does not exists in root: {root}.')
        
    def __getitem__(self, idx):
        name = self.names[idx]
        
        image_raw_size = imagesize.get(self.root / name)
        image_size = (image_raw_size[0]//8*8, image_raw_size[1]//8*8)
        wh0_raw = np.array(image_raw_size)
        wh0 = np.array(image_size)

        # image_raw = cv2.imread(str(self.root / name), cv2.IMREAD_GRAYSCALE)
        # image =  cv2.resize(image_raw, (image_raw.shape[1]//8*8, image_raw.shape[0]//8*8))
        # image = torch.from_numpy(image)[None][None].cuda() / 255.
        # image_raw = torch.from_numpy(image_raw)[None][None].cuda()
        # #?np.array 经过loader后变成了tensor
        # wh0_raw = np.array(image_raw.shape[-2:][::-1])
        # wh0 = np.array(image.shape[-2:][::-1])
        # import ipdb; ipdb.set_trace();

        # hw0_raw = np.array([size_raw[0], size_raw[1]])
        # hw0 = np.array([size[0], size[1]])
        scale = (wh0_raw / wh0).astype(np.float32)

        data = {
            'scale': scale,
            'name': name,
            # 'image': image,
            'original_size': wh0,
        }
        return data

    def __len__(self):
        return len(self.names)
def list_h5_names(path):
    names = []
    with h5py.File(str(path), 'r') as fd:
        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip('/'))
        fd.visititems(visit_fn)
    return list(set(names))
    
def main(images_dir:Path, feature_path: Optional[Path] = None):
    images_dir = images_dir
    loader = ImageDataset(images_dir)
    loader = torch.utils.data.DataLoader(loader, num_workers=0)
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    
    # keypoints = []
    # pred = {}
    # uncertainty = 0
    # for data in tqdm(loader):
    #     name = data['name'][0]  # remove batch dimension
    #     scale0 = np.asarray([data['scale'][0][1], data['scale'][0][0]] )  #由wh调换为hw
    #     scale1 = 8.0
    #     scale = scale0 * scale1
    #     size_8 = np.asarray([data['original_size'][0][1], data['original_size'][0][0]] )
    #     for y in range(0, int(size_8[0]/ scale1)) :
    #         for x in range(0, int(size_8[1]/scale1)) :
    #             keypoints.append([x, y])  #[x, y] 列 行
    #     pred['hw'] = size_8
    #     pred['scale'] = scale
    #     pred['keypoints0'] = np.asarray(keypoints)
    #     pred['keypoints'] = (pred['keypoints0'] + .5) * scale[None] - .5
    #     uncertainty = 1.0 * scale.mean()  
    #     break
    
    # (w*h) = 600*600
    pred_600x600 = {}
    keypoints_600x600 = []
    scale0 = np.asarray([1, 1] ) # h*w
    scale1 = 8.0
    scale = scale0 * scale1
    size_8 = np.asarray([600, 600] ) # h*w
    for y in range(0, int(size_8[0]/ scale1)) :
        for x in range(0, int(size_8[1]/scale1)) :
            keypoints_600x600.append([x, y])  #[x, y] 列 行
    pred_600x600['hw'] = size_8
    pred_600x600['scale'] = scale
    pred_600x600['keypoints0'] = np.asarray(keypoints_600x600)
    pred_600x600['keypoints'] = (pred_600x600['keypoints0'] + .5) * scale[None] - .5
    uncertainty_600x600 = 1.0 * scale.mean()  

    # (w*h) = 480*600
    pred_480x640 = {}
    keypoints_480x640 = []
    scale0 = np.asarray([1, 1] ) # h*w
    scale1 = 8.0
    scale = scale0 * scale1
    size_8 = np.asarray([640, 480]) # h*w
    for y in range(0, int(size_8[0]/ scale1)) :
        for x in range(0, int(size_8[1]/scale1)) :
            keypoints_480x640.append([x, y])  #[x, y] 列 行
    pred_480x640['hw'] = size_8
    pred_480x640['scale'] = scale
    pred_480x640['keypoints0'] = np.asarray(keypoints_480x640)
    pred_480x640['keypoints'] = (pred_480x640['keypoints0'] + .5) * scale[None] - .5
    uncertainty_480x600 = 1.0 * scale.mean()  

    for data in tqdm(loader):
        as_half = True
        name = data['name'][0]  # remove batch dimension
        if (data['original_size'][0][0].item() == 600) and (data['original_size'][0][1].item() == 600):
            pred = pred_600x600
            uncertainty = uncertainty_600x600
        elif (data['original_size'][0][0].item() == 480) and (data['original_size'][0][1].item() == 640):
            pred = pred_480x640
            uncertainty = uncertainty_480x600
        else:
            import ipdb; 
            ipdb.set_trace();
        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)  
        with h5py.File(str(feature_path), 'a') as fd:
            try:
                if name in fd:
                    del fd[name]
                grp = fd.create_group(name)
                for k, v in pred.items():
                    grp.create_dataset(k, data=v)
                if 'keypoints' in pred:
                    grp['keypoints'].attrs['uncertainty'] = uncertainty
            except OSError as error:
                if 'No space left on device' in error.args[0]:
                    del grp, fd[name]
                raise error
        
        # name = list_h5_names(feature_path)   
        
        

    