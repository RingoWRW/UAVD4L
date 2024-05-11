import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.spatial.transform import Rotation
from typing import Dict, List, Union
from tqdm import tqdm
import pickle
import os
import time
import pycolmap

from . import logger
from .utils.io import get_keypoints, get_matches, load_hdf5
from .utils.parsers import parse_image_lists, parse_retrieval
from .utils.read_write_model import qvec2rotmat, rotmat2qvec
from .utils.model3d import Model3D


def do_covisibility_clustering(frame_ids: List[int],
                               reconstruction: pycolmap.Reconstruction):
    clusters = []
    visited = set()
    for frame_id in frame_ids:
        # Check if already labeled
        if frame_id in visited:
            continue

        # New component
        clusters.append([])
        queue = {frame_id}
        while len(queue):
            exploration_frame = queue.pop()

            # Already part of the component
            if exploration_frame in visited:
                continue
            visited.add(exploration_frame)
            clusters[-1].append(exploration_frame)

            observed = reconstruction.images[exploration_frame].points2D
            connected_frames = {
                obs.image_id
                for p2D in observed if p2D.has_point3D()
                for obs in
                reconstruction.points3D[p2D.point3D_id].track.elements
            }
            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


class QueryLocalizer:
    def __init__(self, reconstruction, config=None):
        self.reconstruction = reconstruction
        self.config = config or {}

    def localize(self, points2D_all, points2D_idxs, points3D_id, query_camera):
        points2D = points2D_all[points2D_idxs]
        points3D = [self.reconstruction.points3D[j].xyz for j in points3D_id]
        ret = pycolmap.absolute_pose_estimation(
            points2D, points3D, query_camera,
            estimation_options=self.config.get('estimation', {}),
            refinement_options=self.config.get('refinement', {}),
        )
        return ret


class RetrievalLocalizer:
    def __init__(self, reconstruction, config):
        self.reconstruction = reconstruction
        self.config = config
    

    def localize(self, db_ids, qname, global_descriptors_dict):
        # localize
        if self.config['do_pose_approximation']:
            if self.config['global_descriptors'] is None:
                raise RuntimeError(
                    'Pose approximation requires global descriptors')
            else:
                global_descriptors = global_descriptors_dict
            Rt_retri = self.reconstruction.pose_approximation(qname, db_ids, global_descriptors)
   
        else:
            id_retri = db_ids[0]
            image_retri = self.reconstruction.dbs[id_retri]
            Rt_retri = (image_retri.qvec2rotmat(), image_retri.tvec)

        return Rt_retri


def pose_from_cluster(
        localizer: QueryLocalizer,
        qname: str,
        query_camera: pycolmap.Camera,
        db_ids: List[int],
        features_path: Path,
        matches_path: Path,
        **kwargs):

    kpq = get_keypoints(features_path, qname)
    kpq += 0.5  # COLMAP coordinates

    kp_idx_to_3D = defaultdict(list)
    kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
    num_matches = 0
    for i, db_id in enumerate(db_ids):
        image = localizer.reconstruction.images[db_id]
        if image.num_points3D() == 0:
            logger.debug(f'No 3D points found for {image.name}.')
            continue
        points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1
                                 for p in image.points2D])

        matches, _ = get_matches(matches_path, qname, image.name)
        matches = matches[points3D_ids[matches[:, 1]] != -1]
        num_matches += len(matches)
        for idx, m in matches:
            id_3D = points3D_ids[m]
            kp_idx_to_3D_to_db[idx][id_3D].append(i)
            # avoid duplicate observations
            if id_3D not in kp_idx_to_3D[idx]:
                kp_idx_to_3D[idx].append(id_3D)

    idxs = list(kp_idx_to_3D.keys())
    mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
    mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
    ret = localizer.localize(kpq, mkp_idxs, mp3d_ids, query_camera, **kwargs)
    ret['camera'] = {
        'model': query_camera.model_name,
        'width': query_camera.width,
        'height': query_camera.height,
        'params': query_camera.params,
    }

    # mostly for logging and post-processing
    mkp_to_3D_to_db = [(j, kp_idx_to_3D_to_db[i][j])
                       for i in idxs for j in kp_idx_to_3D[i]]
    log = {
        'db': db_ids,
        'PnP_ret': ret,
        'keypoints_query': kpq[mkp_idxs],
        'points3D_ids': mp3d_ids,
        'points3D_xyz': None,  # we don't log xyz anymore because of file size
        'num_matches': num_matches,
        'keypoint_index_to_db': (mkp_idxs, mkp_to_3D_to_db),
    }
    return ret, log


def pose_from_cluster_(
        localizer: RetrievalLocalizer,
        qname: str,
        query_camera: pycolmap.Camera,
        db_ids: List[int],
        sensor_dict: Dict,
        gt_dict: Dict,
        global_descriptors_dict: Dict,
        **kwargs):
    
    # 1) recover Rt from db retrieval
    Rt_retri = localizer.localize(db_ids, qname, global_descriptors_dict)

    R_retri = Rt_retri[0]
    t_retri = Rt_retri[1]

    qvec_retri = rotmat2qvec(R_retri)

    return qvec_retri, t_retri



def main(reference_sfm: Union[Path, pycolmap.Reconstruction],
         queries: Path,
         retrieval: Path,
         results: Path,
         prepend_camera_name: bool = False,
         config: Dict = None):

    assert retrieval.exists(), retrieval

    queries = parse_image_lists(queries, with_intrinsics=True)
    retrieval_dict = parse_retrieval(retrieval)

    logger.info('Reading the 3D model...')
    if not isinstance(reference_sfm, pycolmap.Reconstruction):
        reference_sfm = Model3D(reference_sfm)
    db_name_to_id = reference_sfm.name2id
    id_to_db_name = reference_sfm.id2name
    
    logger.info('Reading the sensor records...')
    sensor_dict = {}
    with open (config['sensors'], 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = data[0]
            t, q = np.split(np.array(data[1:], float), [3])

            sensor_dict[name] = (qvec2rotmat(q), t)

    # debug: read the gt pose
    logger.info('Reading the gt poses...')
    test_names = []
    gt_dict = {} 
    with open(config['gts'], 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = data[0]
            t, q = np.split(np.array(data[1:], float), [3])

            gt_dict[name] = (qvec2rotmat(q), t)
            test_names.append(name)

    logger.info('Reading the global descriptors...')
    global_descriptors_dict = load_hdf5(config['global_descriptors'])

    localizer = RetrievalLocalizer(reference_sfm, config)

    poses = {}
    pairs = []
    logs = {
        'retrieval': retrieval,
        'loc': {},
    }
    logger.info('Starting localization...')
    t_start = time.time()
    for qname, qcam in tqdm(queries):
        # 1) recover Rt from db retrieval
        if qname not in retrieval_dict:
            logger.warning(
                f'No images retrieved for query image {qname}. Skipping...')
            continue
        db_names = retrieval_dict[qname]
        db_ids = []
        for n in db_names:
            if n not in db_name_to_id:
                logger.warning(f'Image {n} was retrieved but not in database')
                continue
            db_ids.append(db_name_to_id[n])

        # select top-n reference images
        db_ids = db_ids[:config['num_dbs']]

        if config['do_covisibility_clustering']:
            cluster_ids = reference_sfm.covisbility_filtering(db_ids)
        else:
            cluster_ids = db_ids

        # localize
        qvec, tvec = pose_from_cluster_(
            localizer, qname, qcam, cluster_ids, sensor_dict, gt_dict, global_descriptors_dict)
        poses[qname] = (qvec, tvec)

        logs['loc'][qname] = {
            'db': db_ids,
            'covisibility_clustering': config['do_covisibility_clustering'],
        }

        # cluster pairs
        if config['cluster_loc_pairs'] is not None:
            for cluster_id in cluster_ids:
                cluster_name = id_to_db_name[cluster_id]
                pairs.append((qname, cluster_name))
    t_end = time.time()

    logger.info(f'Localizing uses {t_end-t_start} seconds.')
    logger.info(f'Localized {len(poses)} / {len(queries)} images.')

    logger.info(f'Writing poses to {results}...')
    with open(results, 'w') as f:
        for q in poses:
            qvec, tvec = poses[q]
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            name = q.split('/')[-1]
            if prepend_camera_name:
                name = q.split('/')[-2] + '/' + name
            f.write(f'{name} {qvec} {tvec}\n')    

    cluster_loc_pairs = config['cluster_loc_pairs']
    logger.info(f'Writing cluster pairs to {cluster_loc_pairs}...')
    with open(config['cluster_loc_pairs'], 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_sfm', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)
    parser.add_argument('--retrieval', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--ransac_thresh', type=float, default=12.0)
    parser.add_argument('--covisibility_clustering', action='store_true')
    parser.add_argument('--prepend_camera_name', action='store_true')
    args = parser.parse_args()
    main(**args.__dict__)
