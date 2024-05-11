import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Union
from tqdm import tqdm
import pickle
import time
import pycolmap
from .get_3dpoints import Get_Points3D, read_valid_depth
import torch
from .transform import parse_intrinsic_list, parse_pose_list

# import poselib

from . import logger
from .utils.io import get_matches_loftr
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
    matches_path: Path,
    results: Path,
    depth_path: Path,
    ref_pose: Path,
    ref_intrinsics: Path,
    nums_loc: int=3,
    ransac_thresh: int = 12,
    covisibility_clustering: bool = False,
    prepend_camera_name: bool = False,
    config: Dict = None,
):

    assert retrieval.exists(), retrieval
    assert matches_path.exists(), matches_path

    queries = parse_image_lists(queries, with_intrinsics=True)
    retrieval_dict = parse_retrieval(retrieval)

    logger.info('Reading the 3D points...')

    config = {"estimation": {"ransac": {"max_error": ransac_thresh}}, **(config or {})}
    localizer = QueryLocalizer(config)
    poses_db = parse_pose_list(ref_pose)
    K = parse_intrinsic_list(ref_intrinsics)
    poses = {}
    logs = {
        'matches': matches_path,
        'retrieval': retrieval,
        'loc': {},
    }
    logger.info('Starting localization...')
    t_start = time.time()

    # temp_3D_points_db = None
    for qname, query_camera in tqdm(queries):
     
        if qname not in retrieval_dict:
            logger.warning(f'No images retrieved for query image {qname}. Skipping...')
            continue
        db_names = retrieval_dict[qname]
            
        num_loc = 0
       
        for db_name in db_names:
            #get 2D correspondences
            
            points2D_q, points2D_db, _ = get_matches_loftr(matches_path, qname, db_name)
            points2D_q += 0.5  # COLMAP coordinates 
            # get 3D Points
            depth_name = db_name.split('/')[-1].split('.')[0] + '0000.exr' #!
            depth_exr = str(depth_path / depth_name)
            pose_c2w = torch.tensor(poses_db[db_name]).float()
            K_w2c = torch.tensor(K[db_name]).float()
            K_c2w = K_w2c.inverse()
            depth, valid = read_valid_depth(depth_exr, points2D_db)
            # 2D -> 3D
            
            points3D_db = Get_Points3D(depth,pose_c2w[:3, :3],pose_c2w[:3, 3],K_c2w, torch.tensor(points2D_db[valid]),)
            # import pdb;pdb.set_trace()
            if points3D_db != None:
                if num_loc == 0:
                # if temp_3D_points_db == None:
                    temp_3D_points_db = points3D_db
                    temp_2D_points_q = points2D_q
                else:
                    if points3D_db != None:
                        temp_3D_points_db = torch.vstack((temp_3D_points_db, points3D_db))
                        temp_2D_points_q = np.vstack((temp_2D_points_q, points2D_q))
                num_loc += 1
            else:
                num_loc += 1
                continue
        
        num_matches = temp_2D_points_q.shape[0]
        if len(temp_2D_points_q) >= 8:
            ret = localizer.localize(temp_3D_points_db, temp_2D_points_q, query_camera)
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
            continue
        else:
            print(qname,'len less 8')
            continue
            
            
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
