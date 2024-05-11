import argparse
from typing import Union, Optional, Dict, List, Tuple
from pathlib import Path
import pprint
import collections.abc as collections
from tqdm import tqdm
import h5py
import torch
import time

from . import matchers, logger
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval
from .utils.io import list_h5_names


'''
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the match file that will be generated.
    - model: the model configuration, as passed to a feature matcher.
'''
confs = {
    'superglue': {
        'output': 'matches-superglue',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 50,
        },
    },
    'superglue-fast': {
        'output': 'matches-superglue-it5',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 5,
        },
    },
    'NN-superpoint': {
        'output': 'matches-NN-mutual-dist.7',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'distance_threshold': 0.7,
        },
    },
    'NN-ratio': {
        'output': 'matches-NN-mutual-ratio8',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'ratio_threshold': 0.8,
        }
    },
    'NN-mutual': {
        'output': 'matches-NN-mutual',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
        },
    },
    'loftr': {
        'output': 'matches-loftr',
        'model': {
            'name': 'loftr',
            'indoor': '',
            'outdoor': '',
        }
    }
}

def match_from_loftr(conf, features, image_dir, pairs_path, match_path, overwrite: bool = False):
    from . import match_features_loftr
    
    match_path.parent.mkdir(exist_ok=True, parents=True)

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        logger.info('Skipping the matching.')
        return
    
    # pairs = list(pairs)
    # if match_path is not None and match_path.exists():
    #     with h5py.File(str(match_path), 'r') as fd:
    #         for i, j in pairs:
    #             if (names_to_pair(i, j) in fd and names_to_pair(j, i) in fd) or \
    #                (names_to_pair_old(i, j) in fd and names_to_pair_old(j, i) in fd):
    #                 import ipdb; ipdb.set_trace();

    t_start = time.time()
    match_features_loftr.match_from_loftr(conf, features, pairs, image_dir, match_path)
    t_end = time.time()

    logger.info(f'Matching uses {t_end-t_start} seconds.')

    return match_path
    

def main(conf: Dict,
         pairs: Path, features_db: Union[Path, str],features_query: Union[Path, str],
         export_dir: Optional[Path] = None,
         matches: Optional[Path] = None,
         features_ref: Optional[Path] = None,
         overwrite: bool = False, 
         image_dir: Path = None) -> Path:

    features_q = Path(export_dir, features_query+'.h5')
    features_ref = Path(export_dir, features_db+'.h5')
    # Path("/media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/outputs/targetloc/CityofStars_all/", features_db+'.h5')
    # Path(export_dir, features_db+'.h5')
    if matches is None:
        
        matches = Path(
            export_dir, f'{features_query}_{conf["output"]}_{pairs.stem}.h5')

    if isinstance(features_ref, collections.Iterable):
        features_ref = list(features_ref)
    else:
        features_ref = [features_ref]
    
    match_from_paths(conf, pairs, matches, features_q, features_ref, overwrite)
    return matches


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
def match_from_paths(conf: Dict,
                     pairs_path: Path,
                     match_path: Path,
                     feature_path_q: Path,
                     feature_paths_refs: Path,
                     overwrite: bool = False) -> Path:
    logger.info('Matching local features with configuration:'
                f'\n{pprint.pformat(conf)}')
    
    # import pdb;pdb.set_trace()
    if not feature_path_q.exists():
        raise FileNotFoundError(f'Query feature file {feature_path_q}.')
    # xiugai 不可以注释
    for path in feature_paths_refs:
        if not path.exists():
            raise FileNotFoundError(f'Reference feature file {path}.')
 
    name2ref = {n: i for i, p in enumerate(feature_paths_refs)
                for n in list_h5_names(p)}
    match_path.parent.mkdir(exist_ok=True, parents=True)

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        logger.info('Skipping the matching.')
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)
    
    # time statistic
    t_start = time.time()
    for (name0, name1) in tqdm(pairs, smoothing=.1):
        data = {}
        with h5py.File(str(feature_path_q), 'r') as fd:
            grp = fd[name0]
            for k, v in grp.items():
                data[k+'0'] = torch.from_numpy(v.__array__()).float().to(device)
            # some matchers might expect an image but only use its size
            data['image0'] = torch.empty((1,)+tuple(grp['image_size'])[::-1])
        with h5py.File(str(feature_paths_refs[name2ref[name1]]), 'r') as fd:
            grp = fd[name1]
            for k, v in grp.items():
                data[k+'1'] = torch.from_numpy(v.__array__()).float().to(device)
            data['image1'] = torch.empty((1,)+tuple(grp['image_size'])[::-1])
        data = {k: v[None] for k, v in data.items()}
        
        # time1 = time.time()

        pred = model(data)
        
        # time2 = time.time()
        # print("matching:", time2 - time1)
        pair = names_to_pair(name0, name1)
        with h5py.File(str(match_path), 'a') as fd:
            if pair in fd:
                del fd[pair]
            grp = fd.create_group(pair)
            matches = pred['matches0'][0].cpu().short().numpy()
            grp.create_dataset('matches0', data=matches)
            
            if 'matching_scores0' in pred:
                scores = pred['matching_scores0'][0].cpu().half().numpy()
                grp.create_dataset('matching_scores0', data=scores)
    t_end = time.time()

    logger.info(f'Matching uses {t_end-t_start} seconds.')
    logger.info('Finished exporting matches.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path)
    parser.add_argument('--features', type=str,
                        default='feats-superpoint-n4096-r1024')
    parser.add_argument('--matches', type=Path)
    parser.add_argument('--conf', type=str, default='superglue',
                        choices=list(confs.keys()))
    args = parser.parse_args()
    main(confs[args.conf], args.pairs, args.features, args.export_dir)
