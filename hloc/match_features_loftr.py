import torch
from pathlib import Path
from typing import List, Tuple
from . import logger
import pprint
import time


# from ....utils.parsers import names_to_pair, names_to_pair_old
from .utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval

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
from torch.utils.data import DataLoader, get_worker_info
import numpy as np
import os
import math
import cv2
from kornia.feature import LoFTR

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

def get_feat_index(keypoints, width, height):
    keypoints = torch.round((keypoints/ 2)).int() * 2
    w = width // 8
    h = height // 8
    feat_index = keypoints[..., 1] // 8 * w + keypoints[..., 0] // 8
    return keypoints, feat_index

class ImagePairDataset(torch.utils.data.Dataset):

    def __init__(self, root, image_pairs):
        self.root = root
        self.image_pairs = image_pairs
    
    def read_image_pairs(self, name0, name1, root):
        if not (root / name0).exists():
            raise ValueError(
            f'Image {name0} does not exists in root: {root}.')
        if 'query' in name0:
            tmp_name = name0
            name0 = name1
            name1 = tmp_name
        assert('db' in name0)
        image0_raw = cv2.imread(str(root / name0), cv2.IMREAD_GRAYSCALE)
        image0 =  cv2.resize(image0_raw, (image0_raw.shape[1]//8*8, image0_raw.shape[0]//8*8))
        image0 = torch.from_numpy(image0)[None] / 255.
        image0_raw = torch.from_numpy(image0_raw)[None]
        if not (root / name1).exists():
            raise ValueError(
            f'Image {name1} does not exists in root: {root}.')
        image1_raw = cv2.imread(str(root / name1), cv2.IMREAD_GRAYSCALE)
        image1 =  cv2.resize(image1_raw, (image1_raw.shape[1]//8*8, image1_raw.shape[0]//8*8))
        image1 = torch.from_numpy(image1)[None] / 255.
        image1_raw = torch.from_numpy(image1_raw)[None]
        batch = {'name0': name0, 'name1': name1, 'image0': image0, 'image1': image1,'image0_raw': image0_raw, 'image1_raw': image1_raw}

        return batch

    def __getitem__(self, idx):
        name0, name1 = self.image_pairs[idx]
        data = self.read_image_pairs(name0, name1, self.root)

        return data

    def __len__(self):
        return len(self.image_pairs)

@torch.no_grad()
def match_worker(images_dir: Path,
                 features_path: Path,
                 matches_path: Path,
                 img_pairs, subset_ids, match_model, pba=None):

    match_model.eval()
    dataset = ImagePairDataset(images_dir, img_pairs)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    
    for data in tqdm(dataloader):
        data['image0'] = data['image0'].cuda()
        data['image1'] = data['image1'].cuda()
        correspondences = match_model(data)
        h0, w0 = data['image0'].shape[2:]
        h1, w1 = data['image1'].shape[2:]
        correspondences['keypoints0'], feat_index0 = get_feat_index(correspondences['keypoints0'], w0, h0)
        correspondences['keypoints1'], feat_index1 = get_feat_index(correspondences['keypoints1'], w1, h1)
        batch_indexes = correspondences['batch_indexes']
        for batch_idx, (name0, name1) in enumerate(zip(data['name0'], data['name1'])):
            indexes = torch.where(batch_indexes==batch_idx)
            match = torch.zeros(w0//8*h0//8, dtype=torch.int).cuda()
            match[...] = -1
            match[feat_index0[indexes].long()] = feat_index1[indexes]
            score = torch.zeros(w0//8*h0//8, dtype=torch.float32).cuda()
            score[feat_index0[indexes].long()] = correspondences['confidence'][indexes]

            # feature_data = {}
            # with h5py.File(str(features_path), 'r') as fd:
            #     grp = fd[name0]
            #     for k, v in grp.items():
            #         feature_data[k+'0'] = torch.from_numpy(v.__array__()).float()
            #     grp = fd[name1]
            #     for k, v in grp.items():
            #         feature_data[k+'1'] = torch.from_numpy(v.__array__()).float()
            # img0 = data['image0_raw'][batch_idx][0].cpu().numpy()
            # img1 = data['image1_raw'][batch_idx][0].cpu().numpy()
            # kp0 = feature_data['keypoints0'][feat_index0[indexes].long()].cpu().numpy()
            # kp1 = feature_data['keypoints1'][feat_index1[indexes].long()].cpu().numpy()
            # cv_kp0 = []
            # cv_kp1 = []
            # cv_corr_match = []
            # num_corr = 0
            # for pid, (p0, p1) in enumerate(zip(kp0, kp1)):
            #     if pid % 10 == 0:
            #         cv_kp0.append(cv2.KeyPoint(p0[0], p0[1], 1, 1))
            #         cv_kp1.append(cv2.KeyPoint(p1[0], p1[1], 1, 1))
            #         cv_corr_match.append(cv2.DMatch(num_corr, num_corr, 1))
            #         num_corr += 1
            # display_corr = cv2.drawMatches(img0, cv_kp0, img1, cv_kp1, cv_corr_match, None)
            # cv2.imwrite('test.png', display_corr)
            # import ipdb; ipdb.set_trace();

            pair = names_to_pair(name0, name1)
            with h5py.File(str(matches_path), 'a') as fd:
                if pair in fd:
                   del fd[pair]
                grp = fd.create_group(pair)
                matches = match.cpu().short().numpy()
                grp.create_dataset('matches0', data=matches)
            
                scores = score.cpu().half().numpy()
                grp.create_dataset('matching_scores0', data=scores)

    # subset_ids = tqdm(subset_ids) if pba is None else subset_ids
    # for subset_id in subset_ids:
    #     name0, name1 = img_pairs[subset_id]
    #     batch = read_image_pairs(name0, name1, images_dir)   #图片
    #     correspondences = match_model(batch)
    #     h0, w0 = batch['image0'].shape[2:]
    #     h1, w1 = batch['image1'].shape[2:]
    #     correspondences['keypoints0'], feat_index0 = get_feat_index(correspondences['keypoints0'], w0, h0)
    #     correspondences['keypoints1'], feat_index1 = get_feat_index(correspondences['keypoints1'], w1, h1)
    #     match = torch.zeros(w0//8*h0//8, dtype=torch.int).cuda()
    #     match[...] = -1
    #     match[feat_index0.long()] = feat_index1
    #     score = torch.zeros(w0//8*h0//8, dtype=torch.float32).cuda()
    #     score[feat_index0.long()] = correspondences['confidence']

    #     pair = names_to_pair(name0, name1)
    #     with h5py.File(str(matches_path), 'a') as fd:
    #         if pair in fd:
    #             del fd[pair]
    #         grp = fd.create_group(pair)
    #         matches = match.cpu().short().numpy()
    #         grp.create_dataset('matches0', data=matches)
            
    #         scores = score.cpu().half().numpy()
    #         grp.create_dataset('matching_scores0', data=scores)
        
    #     if pba is not None:
    #         pba.update.remote(1)
            


@ray.remote(num_cpus=1, num_gpus=0.25, max_calls=1)  # release gpu after finishing
def match_worker_ray_wrapper(images_dir,match_path, pairs,subset_ids, match_model, pba: ActorHandle):
    return match_worker(images_dir,match_path, pairs,subset_ids, match_model, pba)


@torch.no_grad()
def match_from_loftr(conf,
                     features_path,
                     pairs,
                     images_dir:Path,
                     match_path: Path,
                     use_ray: bool = False) -> Path:

    logger.info('Matching local features with configuration:'
                f'\n{pprint.pformat(conf)}')
    
    if len(pairs) == 0:
        logger.info('Skipping the matching.')
        return
#!loftr
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LoFTR(pretrained='outdoor').to(device)
    
    t_start = time.time()

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
        matches_dict = match_worker(images_dir, features_path, match_path, pairs, range(len(pairs)), model)
    
    t_end = time.time()
    logger.info(f'Matching uses {t_end-t_start} seconds.')
    logger.info('Finished exporting matches.')

    return match_path