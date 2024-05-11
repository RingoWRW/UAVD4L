import numpy as np
import os
from collections import defaultdict
from pathlib import Path
from .transform import parse_db_image_list, parse_pose_list
from typing import Dict, List, Union, Optional
from .utils.parsers import parse_image_lists, parse_retrieval
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

def read_instrincs(intrinsc_path):
    all_K = []
    # all_query_name =[]
    with open(intrinsc_path,'r') as file:
        for line in file:
            data_line=line.strip("\n").split(' ')
            
            img_name = os.path.basename(data_line[0])
            w,h,fx,fy,cx,cy = list(map(float,data_line[2:8]))[:]   #! :8
            focal_length = fx
            K_w2c = np.array([
            [fx,0.0,cx],
            [0.0,fy,cy],
            [0.0,0.0,1.0],
            ]) 

            all_K.append([data_line[0], img_name,K_w2c,focal_length, w, h])
    
    return all_K

def parse_pose_list(path):
    poses = {}
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = os.path.basename(data[0])
            q, t = np.split(np.array(data[1:], float), [4])
            
            R = np.asmatrix(qvec2rotmat(q)).transpose()  #c2w
            
            T = np.identity(4)
            T[0:3,0:3] = R
            T[0:3,3] = -R.dot(t)   #!  c2w
            
            poses[name] = T
            
    
    assert len(poses) > 0
    return poses

def get_pairs(all_render_name, all_query_name, render_path, query_path, iterative_num):   
    render_dir = {}
    for query_name in all_query_name:
        renders = []
        render_candidate = []
        render = []
        query = (query_name.split('/')[-1]).split('.')[0]
        imgq_pth = query_path + query_name
        for render_name in all_render_name:
            if query in render_name:
                render_candidate.append(render_name)
        for imgr_name in render_candidate: 
            imgr_pth = render_path + imgr_name
            imgr = imgr_name.split('/')
            if iterative_num == 0: 
                print(imgr_pth)
                exrr_pth = render_path + imgr[0] +'/' + imgr[1]+'/' + imgr[2] +'/'+ imgr[3] +'/' +'depth/'+query+'/'+ imgr[-1].split('.')[0]+'0001.exr'  #!
                
            else:
                exrr_pth = render_path + imgr_name.split('.')[0]+'0001.exr' 
            # exrr_pth = render_path + imgr.split('.')[0]+'0001.exr' #!
            render = [imgr_pth, exrr_pth]
            renders.append(render)
        render_dir[imgq_pth] = renders
           
    return render_dir  #return pairs dict
def parse_retrieval(path):
    retrieval = defaultdict(list)
    with open(path, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            if len(p) == 0:
                continue
            q, r = p.split()
            retrieval[q].append(r)
    return dict(retrieval)
def get_pairs_imagepath(pairs, render_path, image_path):
    
    render_dir = {}
    for query_name, imgr_name_list in pairs.items():
        renders = []
        imgq_pth = image_path / query_name
        for imgr_name in imgr_name_list:
            imgr = imgr_name.split('/')[-1] 
            imgr_pth = render_path / imgr
            
            exrr_pth = render_path / (imgr.split('.')[0] + '0005.exr') #!  
            render = [imgr_pth, exrr_pth] 
            renders.append(render)
        render_dir[imgq_pth] = renders
    return render_dir
def get_render_candidate(renders, queries):
    render_candidate = []
    pairs = {}
    for query_name in queries:
        query = (query_name.split('/')[-1]).split('.')[0]
        for render_name in renders:
            if query in render_name:
                render_candidate.append(render_name)
        pairs[query_name] = render_candidate
    return pairs
def main(
        image_dir: Path,
        render_dir: Path,
        query_camera: Path,
        render_camera: Path,
        render_extrinsics: Path, data, iter):




    
    assert render_camera.exists(), render_camera
    assert render_extrinsics.exists(), render_extrinsics
    assert query_camera.exists(), query_camera  
      
    # get query cmmera
    K_q = parse_image_lists(query_camera, with_intrinsics=True)
    query_name = [key for key, _ in K_q] 
    
    # get render poses & camera
    K_render = parse_db_image_list(render_camera)
    render_pose = parse_pose_list(render_extrinsics)
    render_name = [key for key, _ in render_pose.items()] 

    pairs = get_render_candidate(render_name, query_name)
    all_pairs_path = get_pairs_imagepath(pairs, render_dir, image_dir)

    
    data["quries"] = K_q
    data["query_name"] = query_name
    
    data["render_intrinsics"] = K_render
    data["render_pose"] = render_pose
    data["render_name"] = render_name
    
    data["pairs"] = all_pairs_path
    data["iter"] = iter  
    # import ipdb; ipdb.set_trace(); 
    

    return data
