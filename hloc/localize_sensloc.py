import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Union
from tqdm import tqdm
import pickle
import time
import pycolmap

# import poselib

from . import logger
from .utils.io import get_keypoints, get_matches, get_correspondences, get_Points3D
from .utils.parsers import parse_image_lists, parse_retrieval


def do_covisibility_clustering(
    frame_ids: List[int], reconstruction: pycolmap.Reconstruction
):
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
                for p2D in observed
                if p2D.has_point3D()
                for obs in reconstruction.points3D[p2D.point3D_id].track.elements
            }
            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


class QueryLocalizer:
    def __init__(self, config=None):
        self.config = config or {}

    def localize(self, points3D, points2D, query_camera):
        points3D = [points3D[i] for i in range(points3D.shape[0])]
        ret = pycolmap.absolute_pose_estimation(
            points2D,
            points3D,
            query_camera,
            estimation_options=self.config.get('estimation', {}),
            refinement_options=self.config.get('refinement', {}),
        )
        return ret


def main(
    queries: Path,
    retrieval: Path,
    features_db: Path,
    features_query: Path,
    matches_path: Path,
    results: Path,
    ransac_thresh: int = 12,
    covisibility_clustering: bool = False,
    prepend_camera_name: bool = False,
    config: Dict = None,
):

    assert retrieval.exists(), retrieval
    assert features_db.exists(), features_db
    assert features_query.exists(), features_query
    assert matches_path.exists(), matches_path

    queries = parse_image_lists(queries, with_intrinsics=True)
    retrieval_dict = parse_retrieval(retrieval)

    logger.info('Reading the 3D points...')
    # if not isinstance(reference_sfm, pycolmap.Reconstruction):
    #     reference_sfm = pycolmap.Reconstruction(reference_sfm)
    # db_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}

    config = {"estimation": {"ransac": {"max_error": ransac_thresh}}, **(config or {})}
    localizer = QueryLocalizer(config)

    poses = {}
    logs = {
        'features db': features_db,
        'features query': features_query,
        'matches': matches_path,
        'retrieval': retrieval,
        'loc': {},
    }
    logger.info('Starting localization...')
    t_start = time.time()
    for qname, query_camera in tqdm(queries):
        if qname not in retrieval_dict:
            logger.warning(f'No images retrieved for query image {qname}. Skipping...')
            continue
        db_names = retrieval_dict[qname]
        points2D_q = get_keypoints(features_query, qname)
        points2D_q += 0.5  # COLMAP coordinates
        # for n in db_names:
        db_name = db_names[0] 
        points3D_db = get_Points3D(features_db, db_name)

        matches, _ = get_matches(matches_path, qname, db_name)
        # valid = matches > -1
        points2D_q = points2D_q[matches[:, 0]]
        points3D_db = points3D_db[matches[:, 1]]

        num_matches = points2D_q.shape[0]
        
        if len(points3D_db) >= 8:

            ret = localizer.localize(points3D_db, points2D_q, query_camera)
            ret['camera'] = {
                'model': query_camera.model_name,
                'width': query_camera.width,
                'height': query_camera.height,
                'params': query_camera.params,
            }
            log = {
                'PnP_ret': ret,
                'keypoints_query': points2D_q,
                'points3D_ids': None,
                'points3D_xyz': None,  # we don't log xyz anymore because of file size
                'num_matches': num_matches,
                'keypoint_index_to_db': None,
            }
            if ret['success']:
                poses[qname] = (ret['qvec'], ret['tvec'])
        else:
            print(qname, 'len less 8')
    t_end = time.time()
    logger.info(f'Localize uses {t_end-t_start} seconds.')
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

    logs_path = f'{results}_logs.pkl'
    logger.info(f'Writing logs to {logs_path}...')
    with open(logs_path, 'wb') as f:
        pickle.dump(logs, f)
    logger.info('Done!')


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
