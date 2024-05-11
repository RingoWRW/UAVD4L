import argparse
import torch
from pathlib import Path
from typing import Dict, List, Union, Optional
import h5py
from types import SimpleNamespace
import cv2
import os
import numpy as np
from tqdm import tqdm
import pprint
import collections.abc as collections
import PIL.Image
import time
from . import extractors, logger
from .utils.base_model import dynamic_load
from .utils.parsers import parse_image_lists
from .utils.io import read_image, list_h5_names


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
    export_dir: Optional[Path] = None,
    as_half: bool = True,
    image_list: Optional[Union[Path, List[str]]] = None,
    feature_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    logger.info(
        'Extracting local features with configuration:' f'\n{pprint.pformat(conf)}'
    )
    t_start = time.time()
    slogen = 'query/'
    dataset = ImageDataset(image_dir, conf['preprocessing'], image_list)

    # import pdb;pdb.set_trace()

    if feature_path is None:
        feature_path = Path(export_dir, 'query_' + conf['output'] + '.h5')
        
    if os.path.exists(feature_path):
        os.remove(feature_path)
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    skip_names = set(
        list_h5_names(feature_path) if feature_path.exists() and not overwrite else ()
    )
    
    dataset.names = [n for n in dataset.names if n not in skip_names]
    if len(dataset.names) == 0:
        logger.info('Skipping the query local extraction.')
        return feature_path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(extractors, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    loader = torch.utils.data.DataLoader(
        dataset, num_workers=1, shuffle=False, pin_memory=True
    )
    for idx, data in enumerate(tqdm(loader)):
        name = dataset.names[idx]  #!\
        pred = model({'image': data['image'].to(device, non_blocking=True)})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        pred['image_size'] = original_size = data['original_size'][0].numpy()
        if 'keypoints' in pred:
            size = np.array(data['image'].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred['keypoints'] = (pred['keypoints'] + 0.5) * scales[None] - 0.5
            if 'scales' in pred:
                pred['scales'] *= scales.mean()
            # add keypoint uncertainties scaled to the original resolution
            uncertainty = getattr(model, 'detection_noise', 1) * scales.mean()
        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)

        with h5py.File(str(feature_path), 'a', libver='latest') as fd:
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
                    logger.error(
                        'Out of disk space: storing features on disk can take '
                        'significant space, did you enable the as_half flag?'
                    )
                    del grp, fd[name]
                raise error

        del pred
        t_end = time.time()
    logger.info(f'Local feature extraction uses {t_end-t_start} seconds.')
    logger.info('Finished exporting features.')
    return feature_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path, required=True)

    parser.add_argument('--as_half', action='store_true')
    parser.add_argument('--image_list', type=Path)
    parser.add_argument('--feature_path', type=Path)
    args = parser.parse_args()

