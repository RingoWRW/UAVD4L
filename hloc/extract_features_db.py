import argparse
import torch
from pathlib import Path
from typing import Dict, List, Union, Optional
import h5py
from types import SimpleNamespace
import cv2
import numpy as np
from tqdm import tqdm
import pprint
import collections.abc as collections
import PIL.Image
import time


from . import extractors, logger
from .utils.base_model import dynamic_load
from .utils.parsers import parse_image_lists
from .utils.io import read_image, list_h5_names, get_keypoints, get_Points3D
from .get_3dpoints import Get_Points3D, read_valid_depth, interpolate_depth
from .transform import parse_intrinsic_list, parse_pose_list

'''
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the feature file that will be generated.
    - model: the model configuration, as passed to a feature extractor.
    - preprocessing: how to preprocess the images read from disk.
'''
confs = {
    'superpoint_aachen': {
        'output': 'feats-superpoint-n4096-r1024',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    # Resize images to 1600px even if they are originally smaller.
    # Improves the keypoint localization if the images are of good quality.
    'superpoint_max': {
        'output': 'feats-superpoint-n4096-rmax1600',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
            'resize_force': True,
        },
    },
    'superpoint_inloc': {
        'output': 'feats-superpoint-n4096-r1600',
        'model': {
            'name': 'superpoint',
            'nms_radius': 4,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        },
    },
    'r2d2': {
        'output': 'feats-r2d2-n5000-r1024',
        'model': {
            'name': 'r2d2',
            'max_keypoints': 5000,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    },
    'd2net-ss': {
        'output': 'feats-d2net-ss',
        'model': {
            'name': 'd2net',
            'multiscale': False,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
    },
    'sift': {
        'output': 'feats-sift',
        'model': {'name': 'dog'},
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        },
    },
    'svcnn': {
        'output': 'feats-svcnn',
        'model': {'name': 'dog'},
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        },
    },
    'sosnet': {
        'output': 'feats-sosnet',
        'model': {'name': 'dog', 'descriptor': 'sosnet'},
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        },
    },
    'disk': {
        'output': 'feats-disk',
        'model': {
            'name': 'disk',
            'max_keypoints': 5000,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
    },
    # Global descriptors
    'dir': {
        'output': 'global-feats-dir',
        'model': {'name': 'dir'},
        'preprocessing': {'resize_max': 1024},
    },
    'netvlad': {
        'output': 'global-feats-netvlad',
        'model': {'name': 'netvlad'},
        'preprocessing': {'resize_max': 1024},
    },
    'openibl': {
        'output': 'global-feats-openibl',
        'model': {'name': 'openibl'},
        'preprocessing': {'resize_max': 1024},
    },
}


def resize_image(image, size, interp):
    if interp.startswith('cv2_'):
        interp = getattr(cv2, 'INTER_' + interp[len('cv2_') :].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith('pil_'):
        interp = getattr(PIL.Image, interp[len('pil_') :].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(f'Unknown interpolation {interp}.')
    return resized


class ImageDataset(torch.utils.data.Dataset):
    default_conf = {
        'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
        'grayscale': False,
        'resize_max': None,
        'resize_force': False,
        'interpolation': 'cv2_area',  # pil_linear is more accurate but slower
    }

    def __init__(self, root, conf, paths=None):
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.root = root
        if paths is None:
            paths = []
            for g in conf.globs:
                paths += list(Path(root).glob('**/' + g))
            if len(paths) == 0:
                raise ValueError(f'Could not find any image in root: {root}.')
            paths = sorted(list(set(paths)))
            self.names = [i.relative_to(root).as_posix() for i in paths]
            logger.info(f'Found {len(self.names)} images in root {root}.')
        else:
            if isinstance(paths, (Path, str)):
                self.names = parse_image_lists(paths)
            elif isinstance(paths, collections.Iterable):
                self.names = [p.as_posix() if isinstance(p, Path) else p for p in paths]
            else:
                raise ValueError(f'Unknown format for path argument {paths}.')

            for name in self.names:
                if not (root / name).exists():
                    raise ValueError(f'Image {name} does not exists in root: {root}.')

    def __getitem__(self, idx):
        name = self.names[idx]
        image = read_image(self.root / name, self.conf.grayscale)
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]

        if self.conf.resize_max and (
            self.conf.resize_force or max(size) > self.conf.resize_max
        ):
            scale = self.conf.resize_max / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            image = resize_image(image, size_new, self.conf.interpolation)

        if self.conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.0

        data = {
            'image': image,
            'original_size': np.array(size),
        }
        return data

    def __len__(self):
        return len(self.names)


@torch.no_grad()
def main(
    conf: Dict,
    image_dir: Path,
    depth_path: Path,
    ref_pose: Path,
    ref_intrinsics: Path,
    export_dir: Optional[Path] = None,
    as_half: bool = False,
    image_list: Optional[Union[Path, List[str]]] = None,
    feature_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    logger.info(
        'Extracting local features with configuration:' f'\n{pprint.pformat(conf)}'
    )
    slogen = 'db/'
    dataset = ImageDataset(image_dir, conf['preprocessing'], image_list)

 

    if feature_path is None:
        feature_path = Path(export_dir, conf['output_db'] + '.h5')
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    skip_names = set(
        list_h5_names(feature_path) if feature_path.exists() and not overwrite else ()
    )
    dataset.names = [n for n in dataset.names if n not in skip_names]
    if len(dataset.names) == 0:
        logger.info('Skipping the local feature extraction.')
        return feature_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(extractors, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)
    poses = parse_pose_list(ref_pose)
    K = parse_intrinsic_list(ref_intrinsics)

    loader = torch.utils.data.DataLoader(
        dataset, num_workers=1, shuffle=False, pin_memory=True
    )
    for idx, data in enumerate(tqdm(loader)):
        name = dataset.names[idx]
        pred = model({'image': data['image'].to(device, non_blocking=True)})

        pred['image_size'] = original_size = data['original_size'][0].numpy()
        
        if 'keypoints' in pred:
            pred['keypoints'] = pred['keypoints'][0].cpu().numpy()
            size = np.array(data['image'].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            # import pdb;pdb.set_trace()
            pred['keypoints'] = (pred['keypoints'] + 0.5) * scales[None] - 0.5
            # 
            if 'scales' in pred:
                pred['scales'] *= scales.mean()
            # add keypoint uncertainties scaled to the original resolution
            uncertainty = getattr(model, 'detection_noise', 1) * scales.mean()
        # get 2D-3D correspondences of database images by depth map 
        # import pdb;pdb.set_trace()
        pose_c2w = torch.tensor(poses[name]).float()
        K_w2c = torch.tensor(K[name]).float()
        K_c2w = K_w2c.inverse()

        depth_name = name.split('/')[-1].split('.')[0] + '0000.exr' #! 0000.exr
        
        depth_exr = str(depth_path / depth_name)
        # xiugai
        try:
            depth, valid = read_valid_depth(depth_exr, pred['keypoints'])

            if depth is None:
                # print(depth_exr)
                continue
            # if points.shape != [len(points), 2]:
            # if len(pred['keypoints'][valid]) < 8:
            #     continue

            Points_3D = Get_Points3D(
                depth,
                pose_c2w[:3, :3],
                pose_c2w[:3, 3],
                K_c2w,
                torch.tensor(pred['keypoints'][valid]),
            )
            

            for k, v in pred.items():
                if k in 'descriptors':
                    descriptors = v[0].T
                    descriptors = descriptors[valid].cpu().numpy()
                    pred[k] = descriptors.T
                elif torch.is_tensor(v[0]):
                    pred[k] = (v[0])[valid].cpu().numpy()
                else:
                    pred[k] = v

            pred['depth'] = depth.numpy()
            pred['3D_points'] = Points_3D.numpy().astype(np.float64)
        except:
            continue

        # =======

        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if dt != np.float64:
                    pred[k] = pred[k].astype(np.float64)
    
        # if '55z' in name:
        #     import ipdb; ipdb.set_trace();
        with h5py.File(str(feature_path), 'a', libver='latest') as fd:
            try:
                if name in fd:
                    del fd[name]
                grp = fd.create_group(name)
                # xiugai
                if len(pred) == 0:
                    continue

                for k, v in pred.items():

                    grp.create_dataset(k, data=v)
                if 'keypoints' in pred:
                    grp['keypoints'].attrs['uncertainty'] = uncertainty
            except OSError as error:
                if 'No space left on device' in error.args[0]:
                    logger.error(
                        'Out of disk space: storing features on disk can take '
                        'significant space, did you enable the as_half flag?'
                    )
                    del grp, fd[name]
                raise error
        # print(feature_path)
        # mkpts1r_final = get_keypoints(feature_path, 'db/17h00039.png')
        # point = get_Points3D(feature_path, 'db/17h00039.png')
        # import ipdb; ipdb.set_trace();
        del pred

    logger.info('Finished exporting features.')
    return feature_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path, required=True)
    parser.add_argument(
        '--conf', type=str, default='superpoint_aachen', choices=list(confs.keys())
    )
    parser.add_argument('--as_half', action='store_true')
    parser.add_argument('--image_list', type=Path)
    parser.add_argument('--feature_path', type=Path)
    args = parser.parse_args()
    main(confs[args.conf], args.image_dir, args.export_dir, args.as_half)
