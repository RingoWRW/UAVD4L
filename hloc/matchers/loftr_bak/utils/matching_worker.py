from .. import LoFTR, default_cfg
import torch
from pathlib import Path
from typing import List, Tuple
from .... import logger
import pprint


from ....utils.parsers import names_to_pair, names_to_pair_old

# from ....hloc.utils.ray_utils import ProgressBar, chunk_index
import ray # version==1.2
from ray.actor import ActorHandle
from typing import ChainMap
from pathlib import Path
import pprint
import collections.abc as collections
from tqdm import tqdm
import h5py
import torch
import numpy as np
import os
import math
import cv2

confs = {

}
cfg_ray = {
        "slurm": False,
        "n_workers": 16,
        # "n_cpus_per_worker": 1,
        "n_cpus_per_worker": 1,
        "n_gpus_per_worker": 0.25,
        "local_mode": False,
    }



def read_image_pairs(name0, name1, root):
    if not (root / name0).exists():
        raise ValueError(
        f'Image {name0} does not exists in root: {root}.')
    image0_raw = cv2.imread(str(root / name0), cv2.IMREAD_GRAYSCALE)
    image0 =  cv2.resize(image0_raw, (image0_raw.shape[1]//8*8, image0_raw.shape[0]//8*8))
    image0 = torch.from_numpy(image0)[None][None].cuda() / 255.
    image0_raw = torch.from_numpy(image0_raw)[None][None].cuda()
    if not (root / name1).exists():
        raise ValueError(
        f'Image {name1} does not exists in root: {root}.')
    image1_raw = cv2.imread(str(root / name1), cv2.IMREAD_GRAYSCALE)
    image1 =  cv2.resize(image1_raw, (image1_raw.shape[1]//8*8, image1_raw.shape[0]//8*8))
    image1 = torch.from_numpy(image1)[None][None].cuda() / 255.
    image1_raw = torch.from_numpy(image1_raw)[None][None].cuda()
    batch = {'name0': name0, 'name1': name1, 'image0': image0, 'image1': image1,'image0_raw': image0_raw, 'image1_raw': image1_raw}

    return batch

def find_unique_new_pairs(pairs_all: List[Tuple[str]], match_path: Path = None):
    '''Avoid to recompute duplicates to save time.'''
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = list(pairs)
    if match_path is not None and match_path.exists():
        with h5py.File(str(match_path), 'r') as fd:
            pairs_filtered = []
            for i, j in pairs:
                if (names_to_pair(i, j) in fd or
                        names_to_pair(j, i) in fd or
                        names_to_pair_old(i, j) in fd or
                        names_to_pair_old(j, i) in fd):
                    continue
                pairs_filtered.append((i, j))
        return pairs_filtered
    return pairs


@torch.no_grad()
def match_worker1(img_pairs, subset_ids, match_model, feature_path, pba=None):
    # match features by superglue
    feature_file = h5py.File(feature_path, "r")
    matches_dict = {}
    subset_ids = tqdm(subset_ids) if pba is None else subset_ids

    for subset_id in subset_ids:
        name0, name1 = img_pairs[subset_id]
        pair = names_to_pair(name0, name1)

        # Construct input data:
        data = {}
        feats0, feats1 = feature_file[name0], feature_file[name1]
        for k in feats0.keys():
            data[k + "0"] = feats0[k].__array__()
        for k in feats1.keys():
            data[k + "1"] = feats1[k].__array__()
        data = {k: torch.from_numpy(v)[None].float().cuda() for k, v in data.items()}

        data["image0"] = torch.empty(
            (
                1,
                1,
            )
            + tuple(feats0["image_size"])[::-1]
        )
        data["image1"] = torch.empty(
            (
                1,
                1,
            )
            + tuple(feats1["image_size"])[::-1]
        )

        pred = match_model(data)
        pair = names_to_pair(name0, name1)

        matches = pred["matches0"][0].cpu().short().numpy()
        if "matching_scores0" in pred:
            scores = pred["matching_scores0"][0].cpu().half().numpy()
            matches = np.c_[matches, scores]
            matches_dict[pair] = matches  # 2*N 
        else:
            matches_dict[pair] = matches  # 1*N    


        if pba is not None:
            pba.update.remote(1)

    return matches_dict
@torch.no_grad()
def match_worker(images_dir:Path,
                 matches_path:Path,
                 img_pairs, subset_ids, match_model, pba=None):
    # match features by superglue
    subset_ids = tqdm(subset_ids) if pba is None else subset_ids

    for subset_id in subset_ids:
        name0, name1 = img_pairs[subset_id]
        batch = read_image_pairs(name0, name1, images_dir)   #图片
        import ipdb; ipdb.set_trace();
        match_model(batch, matches_path)

        if pba is not None:
            pba.update.remote(1)
            


@ray.remote(num_cpus=1, num_gpus=0.25, max_calls=1)  # release gpu after finishing
def match_worker_ray_wrapper(images_dir,match_path, pairs,subset_ids, match_model, pba: ActorHandle):
    return match_worker(images_dir,match_path, pairs,subset_ids, match_model, pba)


@torch.no_grad()
def match_from_loftr(weights,
                     pairs,
                     images_dir:Path,
                     match_path: Path,
                     use_ray: bool = False) -> Path:

    logger.info('Matching local features with configuration:'
                f'\n{pprint.pformat(weights)}')
    
    if len(pairs) == 0:
        logger.info('Skipping the matching.')
        return
#!loftr
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LoFTR(config = default_cfg)  # load loftr
    model.load_state_dict(torch.load(weights)['state_dict'])
    model = model.eval().to(device)
    
    if False : # use_ray:
        # Initial ray:
        if cfg_ray["slurm"]:
            ray.init(address=os.environ["ip_head"])
        else:
            ray.init(
                num_cpus=math.ceil(cfg_ray["n_workers"] * cfg_ray["n_cpus_per_worker"]),
                num_gpus=math.ceil(cfg_ray["n_workers"] * cfg_ray["n_gpus_per_worker"]),
                local_mode=cfg_ray["local_mode"],
                ignore_reinit_error=True,
            ) 
        #?
        pb = ProgressBar(len(pairs), "Matching image pairs...") 
        all_subset_ids = chunk_index(
            len(pairs), math.ceil(len(pairs) / cfg_ray["n_workers"])
        )  
        obj_refs = [
            match_worker_ray_wrapper.remote(
                images_dir,match_path, pairs, subset_ids, model, pb.actor
            )
            for subset_ids in all_subset_ids
        ]
        pb.print_until_done()
        results = ray.get(obj_refs)
        matches_dict = dict(ChainMap(*results))
    else:
        matches_dict = match_worker(images_dir, match_path, pairs, range(len(pairs)), model)
    
            
    logger.info('Finished exporting matches.')