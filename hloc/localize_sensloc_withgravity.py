import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Union
from tqdm import tqdm
import pickle
import torch
import time
import pycolmap
import collections.abc as collections
import os
from utils.io import parse_sensor  #!path
from .utils.read_write_model import qvec2rotmat, rotmat2qvec
from . import logger
from .utils.io import get_keypoints, get_matches, get_Points3D
from .utils.parsers import parse_image_lists, parse_retrieval

from pixlib.geometry import Camera, Pose
from pixlib.models.metric_utils import ransac_PnP_with_gravity

def parse_names(prefix, names, names_all):
    if prefix is not None:
        if not isinstance(prefix, str):
            prefix = tuple(prefix)
        names = [n for n in names_all if n.startswith(prefix)]
    elif names is not None:
        if isinstance(names, (str, Path)):
            names = parse_image_lists(names)
        elif isinstance(names, collections.Iterable):
            names = list(names)
        else:
            raise ValueError(
                f'Unknown type of image list: {names}.'
                'Provide either a list or a path to a list file.'
            )
    else:
        names = names_all
    return names
def get_query_sensor_pose(names, path):
    poses = {}
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = data[0]
            q, t = np.split(np.array(data[1:], float), [4])

            # rotation
            R = qvec2rotmat(q)

            poses[name] = R, t
    Rs = []
    ts = []
    for name in names:
        R, t = poses[name]
        Rs.append(R)
        ts.append(t)
    Rs = np.stack(Rs, 0)
    ts = np.stack(ts, 0)

    # Invert the poses from world-to-camera to camera-to-world.
    Rs = Rs.transpose(0, 2, 1)
    ts = -(Rs @ ts[:, :, None])[:, :, 0]

    # Only use 2d position (x,y)
    ts_2d = ts[:, :-1]

    return ts_2d, Rs

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
    query_path: Path,
    retrieval: Path,
    features_db: Path,
    features_query: Path,
    sensor_path : Path,
    matches_path: Path,
    results: Path,
    ransac_thresh: int = 12,
    covisibility_clustering: bool = False,
    prepend_camera_name: bool = False,
    config: Dict = None,
):
    '''
    PnP solver
    INPUT
    query_path: query intrinsics
    retrieval: top 1 retrieval candidate
    features_db: local features of each image in database
    features_query: local features of each query
    sensor_path: gravity prior for PnP
    matches_path: correspondences between query and retrieval candidate
    
    OUTPUT
    results: save estimated pose
    '''
    assert retrieval.exists(), retrieval
    assert features_db.exists(), features_db
    assert features_query.exists(), features_query
    assert matches_path.exists(), matches_path

    cameras = parse_image_lists(query_path, with_colmap=True)  # use colmap;  set with_intrinsics = True if use pycolmap
    queries = {n: c for n, c in cameras}
    
    retrieval_dict = parse_retrieval(retrieval)
    # gps-priored 
    if sensor_path is not None:
        sensor_dict = parse_sensor(sensor_path)

    logger.info('Reading the 3D points...')

    config = {"estimation": {"ransac": {"max_error": ransac_thresh}}, **(config or {})}

    logs = {
        'features db': features_db,
        'features query': features_query,
        'matches': matches_path,
        'retrieval': retrieval,
        'loc': {},
    }
    logger.info('Starting localization...')
    t_start = time.time()
    output_poses_poselib = {}
    output_poses_gravity = {}
    for qname, query_camera in tqdm(queries.items()):
        
        if qname not in retrieval_dict:
            logger.warning(f'No images retrieved for query image {qname}. Skipping...')
            continue
        db_names = retrieval_dict[qname]
        qcamera = Camera.from_colmap(query_camera)
            #get sensor poses
        sensor_pose = sensor_dict[os.path.basename(qname)]
        T_sensor = Pose.from_Rt(*sensor_pose)
        T_init = T_sensor
        ret = {'T_init': T_init}

        num_loc = 0
        for db_name in db_names:
            
            points2D_q = get_keypoints(features_query, qname)
            points2D_q += 0.5  # COLMAP coordinates
            # import pdb;pdb.set_trace()
            points3D_db = get_Points3D(features_db, db_name)
            #get correspondences
            matches, _ = get_matches(matches_path, qname, db_name)  
            # valid = matches > -1
            points2D_q = points2D_q[matches[:, 0]]
            points3D_db = points3D_db[matches[:, 1]]  #get 3D points


            if num_loc == 0:
                temp_3D_points_db = points3D_db
                temp_2D_points_q = points2D_q
            else:
                if len(points3D_db) != 0:
                    temp_3D_points_db = np.vstack((temp_3D_points_db, points3D_db))
                    temp_2D_points_q = np.vstack((temp_2D_points_q, points2D_q))
            num_loc += 1

        #  pose estimation with ransac Poselib PnP
        try:
            
            query_pose_pred_, query_pose_pred_homo_, info_, state_ = ransac_PnP_with_gravity(
                qcamera,
                temp_2D_points_q,
                temp_3D_points_db,
                T_init,
                img_hw=(int(qcamera.size[1]), int(qcamera.size[0])),
                pnp_reprojection_error=12,
                using_gravity_check=False, # False
                gravity_threshold=2
            )
            query_pose_pred_grav, query_pose_pred_grav_homo, info_grav, state_grav = ransac_PnP_with_gravity(
                qcamera,
                temp_2D_points_q,
                temp_3D_points_db,
                T_init,
                img_hw=(int(qcamera.size[1]), int(qcamera.size[0])),
                pnp_reprojection_error=12,
                using_gravity_check=True, # True
                gravity_threshold=2 # 2
            )

   
            ####
            # poselib: with and without checking gravity direction
            if state_ == True:
                T_opt_ = Pose.from_4x4mat(query_pose_pred_homo_)
            else:
                logger.info(f"PnP with gravity failed for query {qname}")
                T_opt_ = T_init
            if state_grav == True:
                T_opt_grav = Pose.from_4x4mat(query_pose_pred_grav_homo)
            else:
                logger.info(f"PnP with gravity failed for query {qname}")
                T_opt_grav = T_init
            # Compute relative pose w.r.t. initilization
            # dR, dt = (T_init @ T_opt.inv()).magnitude()
            ret = {   
                **ret,
                'success': True,
                # 'T_refined': T_opt,
                'T_refined_poselib': T_opt_,
                'T_refined_gravity': T_opt_grav,       
                # 'diff_R': dR.item(),
                # 'diff_t': dt.item(),
            }           
    
                               
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                logger.info('Out of memory')
                torch.cuda.empty_cache()
                ret = {'success': False}
            else:
                raise
        # output_logs['localization'][name] = ret
        if ret['success']:
            # R, tvec = ret['T_refined'].numpy()
            R_poselib, tvec_poselib = ret['T_refined_poselib'].numpy()
            R_gravity, tvec_gravity = ret['T_refined_gravity'].numpy()
        elif 'T_init' in ret:
            # R, tvec = ret['T_init'].cpu().numpy()
            R_poselib, tvec_poselib = ret['T_init'].cpu().numpy()
            R_gravity, tvec_gravity = ret['T_init'].cpu().numpy()
        else:
            continue

        try:
            #  output_poses[name] = (rotmat2qvec(R), tvec)
             output_poses_poselib[qname] = (rotmat2qvec(R_poselib), tvec_poselib)
             output_poses_gravity[qname] = (rotmat2qvec(R_gravity), tvec_gravity)
        except :
            continue
           



    t_end = time.time()
    logger.info(f'Localize uses {t_end-t_start} seconds.')
    logger.info(f'Localized {len(output_poses_poselib)} / {len(queries)} images.')
    logger.info(f'Writing poses to {results}...')
    with open(results, 'w') as f:
        for q in output_poses_gravity:
            qvec, tvec = output_poses_gravity[q]
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
