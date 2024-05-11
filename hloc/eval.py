import numpy as np
import os
from pathlib import Path
from scipy.spatial.transform import Rotation
from collections import defaultdict
from . import  logger
import pyproj
import numpy as np
from geopy.distance import distance
import torch
from .transform import parse_intrinsic_list, parse_pose_list
from .utils.io import get_matches_loftr
from .utils.parsers import parse_image_lists, parse_retrieval
from .get_3dpoints import Get_Points3D, read_valid_depth
from tqdm import tqdm

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def warp_kpts(kpts, depth1, T_0to1, T0,T1,K0, K1):

    kpts_cam = K0.inverse() @ kpts.transpose(1,0)
    temp_kps = torch.linalg.inv(T0[:3,:3]) @ kpts_cam + T0[:3,[3]]
    import pdb;pdb.set_trace()
    kpts_cam_w = T1[:3,:3] @ temp_kps + T1[:3,[3]]
    # kpts_cam_w = T_0to1[:3,:3] @ kpts_cam + T_0to1[:3,[3]] #[3,L]
    depths_com = kpts_cam_w[2,:]


    w_kpts_h = (K0 @ kpts_cam_w).transpose(1,0)
    w_kpts0 = w_kpts_h[:,:2].abs() / (w_kpts_h[:,[2]] + 1e-4)

    h, w = K1[0,2]*2, K1[1,2]*2
    covisible_mask = (w_kpts0[:,0]>0) * (w_kpts0[:,0]<w-1) * \
                    (w_kpts0[:,1]>0) * (w_kpts0[:,1]<h-1)
    consistent_mask = ((depths_com - depth1) / depth1).abs()<0.2
    valid_mask = covisible_mask * consistent_mask
    percentage = torch.sum(valid_mask) / len(kpts) * 100

def evaluate_retrieval(queries,retrieval, gt, matches_path,depth_path,q_depth_path,ref_pose,ref_intrinsics,num_loc):

    queries = parse_image_lists(queries, with_intrinsics=True)
    retrieval_dict = parse_retrieval(retrieval)
    poses_db = parse_pose_list(ref_pose)
    poses_gtq = parse_pose_list(gt)
    K = parse_intrinsic_list(ref_intrinsics)
    
    for qname, query_camera in tqdm(queries):
        for index in range(num_loc):
            if qname not in retrieval_dict:
                logger.warning(f'No images retrieved for query image {qname}. Skipping...')
                continue
            db_names = retrieval_dict[qname]
            db_name = db_names[index]
            #get 2D correspondences
            points2D_q, points2D_db, _ = get_matches_loftr(matches_path, qname, db_name)
            points2D_q += 0.5  # COLMAP coordinates 
            # get 3D Points
            depth_name_db = db_name.split('/')[-1].split('.')[0] + '0000.exr' #!
            depth_exr_db = str(depth_path / depth_name_db)
            import pdb;pdb.set_trace()
            depth_name_qt = qname.split('/')[-1].split('.')[0] + '0000.exr' #!
            depth_exr_qt = str(q_depth_path / depth_name_qt)
            pose_c2w = torch.tensor(poses_db[db_name]).float()
            pose_c2w_gt = torch.tensor(poses_gtq[qname]).float()

            pose_r = torch.linalg.inv(pose_c2w) @ pose_c2w_gt

            K_w2c = torch.tensor(K[db_name]).float()

            depth_db, valid_db = read_valid_depth(depth_exr_db, points2D_db)
            depth_qt, valid_qt = read_valid_depth(depth_exr_qt, points2D_q)
            points2D_q_tensor = torch.tensor(points2D_q)
            
            kpts = torch.cat([points2D_q_tensor, torch.unsqueeze(depth_qt[valid_qt],1)], dim=-1)
            
            import pdb;pdb.set_trace()

            # kpts_cam_w = T_0to1[:3,:3] @ kpts_cam + T_0to1[:3,[3]] #[3,L]
            # depths_com = kpts_cam[2,:]
            warp_kpts(kpts, depth_db[valid_db], pose_r, pose_c2w, pose_c2w_gt, K_w2c, K_w2c)
            # warp_kpts(kpts, depth_db[valid_db], pose_c2w, pose_c2w_gt, K_w2c, K_w2c)
            



def evaluate(results, gt, result, only_localized=False):
    predictions = {}
    with open(results, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = data[0]
            q, t = np.split(np.array(data[1:], float), [4])

            predictions[os.path.splitext(os.path.basename(name))[0]] = (qvec2rotmat(q), t)

    
    test_names = []
    gts = {}
    with open(gt, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = data[0].split('/')[-1]
            q, t = np.split(np.array(data[1:], float), [4])

            gts[os.path.splitext(os.path.basename(name))[0]] = (qvec2rotmat(q), t)
            test_names.append(os.path.splitext(os.path.basename(name))[0])
    
    errors_t = []
    errors_R = []
    index = 0
    # import pdb;pdb.set_trace()
    for name in test_names:
        if name not in predictions:
            print("nm",name)
            if only_localized:
                continue
            e_t = np.inf
            e_R = 180.
        else:
            index+=1
            R_gt, t_gt = gts[name]
            R, t = predictions[name]
            
            e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
            # print(R.T @ t-R_gt.T @ t_gt)
            # print()
            # print('-')
            cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1., 1.)
            e_R = np.rad2deg(np.abs(np.arccos(cos)))
            

        errors_t.append(e_t)
        errors_R.append(e_R)
    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)
    med_t = np.median(errors_t)
    std_t = np.std(errors_t)
    med_R = np.median(errors_R)
    std_R = np.std(errors_R)

    


    # out = f'Results for file {results.name}:'
    out = f'\nMedian errors: {med_t:.3f}m, {med_R:.3f}deg'
    out += f'\nStd errors: {std_t:.3f}m, {std_R:.3f}deg'

    out += '\nPercentage of test images localized within:'
    threshs_t = [1, 3.0, 5.0]
    threshs_R = [1.0, 3.0, 5.0]
    for th_t, th_R in zip(threshs_t, threshs_R):
        ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
        out += f'\n\t{th_t:.2f}m, {th_R:.0f}deg : {ratio*100:.2f}%'
    logger.info(out)
    # print (out)

    with open(result,'w') as f:
        for i in range(len(test_names)):
            name = test_names[i]
            et = errors_t[i]
            eR = errors_R[i]
            info = str(name) + ' ' + str(et) + ' ' + str(eR) + '\n'
            f.write(info)
        f.writelines(out)



def load_gt(path, transformer):
    d = {}
    with open(path, 'r') as file:
        for line in file.read().rstrip().split('\n'):
            name = line.split(' ')[0]
            if name[-5] is 'Z':
                # key = line.split(' ')[0].split('_')[-2]
                value = list(map(float,line.rstrip().split(' ')[1:]))
                #x, y, z = transformer.transform(value[1], value[0], value[2])  # for RTK equipment
                d[name] = [value[0], value[1], value[2]]
    return d

def load_gt_w(path, transformer):
    d = {}
    with open(path, 'r') as file:
        for line in file.read().rstrip().split('\n'):
            name = line.split(' ')[0]
            if name[-5] is 'W':
                # key = line.split(' ')[0].split('_')[-2]
                value = list(map(float,line.rstrip().split(' ')[1:]))
                #x, y, z = transformer.transform(value[1], value[0], value[2])  # for RTK equipment
                d[name] = [value[0], value[1], value[2]]
    return d

def load_predict_old(path, object_name):
    d = {}
    with open(path, 'r') as file:
        for line in file.read().rstrip().split('\n'):
            key = line.split(' ')[0].split('_')[-2]
            name = line.split(' ')[1]
            if object_name in name:
                value = list(map(float,line.rstrip().split(' ')[2:]))
                d[key] = value
            else:
                continue
    return d

def load_predict(path):
    d = {}
    with open(path, 'r') as file:
        for line in file.read().rstrip().split('\n'):
            name = line.split(' ')[0]
            value = list(map(float,line.rstrip().split(' ')[2:]))
            d[name] = value
    return d

def eval_absolute_XYZ(gt_list, pred_list, result):
    
    errors_t = [] 
    image_num = len(pred_list)
   
    for name in pred_list.keys():
        if name not in gt_list.keys():
            e_t = np.inf
            # e_R = 180.
            continue
        else:
            gt = gt_list[name]
            t_gt = np.array(gt[:2])
            pred = pred_list[name]
            t = np.array(pred[:2])
            e_t = np.linalg.norm(t_gt - t, axis=0)
        errors_t.append(e_t)

    errors_t = np.array(errors_t)
    med_t = np.median(errors_t)
    max_t = np.max(errors_t)
    min_t = np.min(errors_t)
    # import ipdb; ipdb.set_trace();
    out = f'\nTarget Localization results'
    out = f'\nTest image nums: {image_num}'
    out += f'\nMedian errors: {med_t:.3f}m'
    out += f'\nMax errors: {max_t:.3f}m'
    out += f'\nMin errors: {min_t:.3f}m'
    # print(out)
    out += '\nPercentage of test images localized within:'
    threshs_t = [1, 3, 5, 10]
    for th_t in threshs_t:
        ratio = np.mean((errors_t < th_t))
        out += f'\n\t{th_t:.0f}m : {ratio*100:.2f}%'
    
    logger.info(out)
    with open(result,'w') as f:
            f.writelines(out)
    f.close()
    # logger.info(out)   
def main(estimated_pose: Path, 
         gt_pose: Path,
         gt_path: Path, 
         predict_path: Path,
         object_name: Path,
         result_save_path: Path):
    # 定义WGS84坐标系和CGCS_2000_3_Degree_GK_CM_90E投影坐标系
    wgs84 = pyproj.CRS('EPSG:4326')
    cgcs2000_114E = pyproj.CRS('EPSG:4547')
    # cgcs2000_90E = pyproj.CRS('EPSG:4539')

    # 定义转换函数
    transformer = pyproj.Transformer.from_crs(wgs84, cgcs2000_114E, always_xy=True)
    # import pdb;pdb.set_trace()
    # 变焦
    gt_dict = load_gt(gt_path, transformer)
    # 广角
    # gt_dict = load_gt_w(gt_path, transformer)
    predict_dict = load_predict(predict_path)    
    # import pdb;pdb.set_trace()
    print ('Evaluation localization')
    # evaluate(estimated_pose, gt_pose)
    
    eval_absolute_XYZ(gt_dict,predict_dict,result_save_path)
    


    


