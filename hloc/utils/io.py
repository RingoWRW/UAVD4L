from typing import Tuple, Dict, List, Union, Any
from pathlib import Path
import numpy as np
import cv2
import h5py

from .parsers import names_to_pair, names_to_pair_old


def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f'Cannot read image {path}.')
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def list_h5_names(path):
    names = []
    with h5py.File(str(path), 'r') as fd:
        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip('/'))
        fd.visititems(visit_fn)
    return sorted(list(set(names)))


def load_hdf5(path: Path) -> Dict[str, Any]:
    with h5py.File(path, 'r') as hfile:
        data = {}
        def collect(_, obj):  # noqa
            if isinstance(obj, h5py.Dataset):
                name = obj.parent.name.strip('/')
                if len(obj.__array__()) != 2: # ignore "image_size" items
                    data[name] = obj.__array__()
        hfile.visititems(collect)
    return data


# def get_keypoints(path: Path, name: str) -> np.ndarray:
#     with h5py.File(str(path), 'r') as hfile:
#         p = hfile[name]['keypoints'].__array__()
#     return p

def get_keypoints(path: Path, name: str,
                  return_uncertainty: bool = False) -> np.ndarray:
    with h5py.File(str(path), 'r', libver='latest') as hfile:
        dset = hfile[name]['keypoints']
        p = dset.__array__()
        uncertainty = dset.attrs.get('uncertainty')
    if return_uncertainty:
        return p, uncertainty
    return p


        
def get_Points3D(path: Path, name: str) -> np.ndarray:
    with h5py.File(str(path), 'r', libver='latest') as hfile:
        dset = hfile[name]['3D_points']
        p = dset.__array__()
    return p

def get_correspondences(feature_path: Path, name: str):
    points2D = get_keypoints(feature_path, name)
    points2D += 0.5  # COLMAP coordinates
    
    points3D = get_Points3D(feature_path, name)
    
    return points2D, points3D
def find_pair(hfile: h5py.File, name0: str, name1: str):
    pair = names_to_pair(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair(name1, name0)
    if pair in hfile:
        return pair, True
    # older, less efficient format
    pair = names_to_pair_old(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair_old(name1, name0)
    if pair in hfile:
        return pair, True
    raise ValueError(
        f'Could not find pair {(name0, name1)}... '
        'Maybe you matched with a different list of pairs? ')


def get_matches(path: Path, name0: str, name1: str) -> Tuple[np.ndarray]:
    with h5py.File(str(path), 'r') as hfile:
        pair, reverse = find_pair(hfile, name0, name1)
        matches = hfile[pair]['matches0'].__array__()
        scores = hfile[pair]['matching_scores0'].__array__()
    idx = np.where(matches != -1)[0]
    matches = np.stack([idx, matches[idx]], -1)
    if reverse:
        matches = np.flip(matches, -1)
    scores = scores[idx]
    return matches, scores


def get_matches_loftr(path: Path, name0: str, name1: str) -> Tuple[np.ndarray]:
    with h5py.File(str(path), 'r') as hfile:
        pair, reverse = find_pair(hfile, name0, name1)
        mkpt_db = hfile[pair]['keypoints1'].__array__()
        mkpt_q = hfile[pair]['keypoints0'].__array__()
        # matches = hfile[pair]['matches0'].__array__()
        scores = hfile[pair]['scores'].__array__()
    return mkpt_q, mkpt_db, scores