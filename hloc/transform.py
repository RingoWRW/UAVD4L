import numpy as np
import os
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

def parse_intrinsic_list(path):
    images = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            
            data_line=line.split(' ')
            name = data_line[0].split('/')[-1]
            _,_,fx,fy,cx,cy = list(map(float,data_line[2:8]))[:]  
            
            K_w2c = np.array([ #!
            [fx,0.0,cx],
            [0.0,fy,cy],
            [0.0,0.0,1.0],
            ]) 
            images[name] = K_w2c
  
    return images

def parse_pose_list(path):
    poses = {}
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = data[0].split('/')[-1]
            q, t = np.split(np.array(data[1:], float), [4])
            
            R = np.asmatrix(qvec2rotmat(q)).transpose() 
            
            T = np.identity(4)
            T[0:3,0:3] = R
            T[0:3,3] = -R.dot(t)  
            poses[name] = T
            
    
    assert len(poses) > 0
    return poses