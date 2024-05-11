import cv2
import numpy as np
import os
from pathlib import Path
import glob
import shutil
def read_intrinsics(intrinsics_path: Path):
    images = {}
    with open(intrinsics_path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            
            data_line=line.split(' ')
            name = data_line[0].split('/')[-1]
            w,h,fx,fy,cx,cy = list(map(float,data_line[2:8]))[:]  
            
            K_w2c = np.array([ #!
            [fx,0.0,cx],
            [0.0,fy,cy],
            [0.0,0.0,1.0],
            ]) 
            break
            # images[name] = K_w2c
  
    return K_w2c, int(w), int(h)   
def write_intrinsics(intrinsics_path, NewCameraMatrix, w, h, name_list):
    fx, fy, cx, cy = NewCameraMatrix[0][0], NewCameraMatrix[1][1], NewCameraMatrix[0, 2], NewCameraMatrix[1,2]
    
    with open(intrinsics_path, 'w+') as f:
        for name in name_list:
            outline = name + ' ' + 'PINHOLE' +' '+ str(w) + ' ' + str(h) + ' ' + str(fx) + ' ' + str(fy) + ' ' + str(cx) + ' ' + str(cy) + '\n'
            f.write(outline)
    
def main(image_path: Path,
         w_image_path: Path,
         z_image_path: Path,
         intrinsics_path: Path,
         kp,
         ):
    kp = np.array(kp).astype(np.float32)
    CameraMatrix, w, h = read_intrinsics(intrinsics_path)
    NewCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(CameraMatrix, kp, (w, h), 1,  (w, h), 0)

    name_list_write = []
    if not os.path.exists(w_image_path):
        try:
            os.makedirs(w_image_path)
            os.makedirs(z_image_path)
        except Exception as e:
            print(e)
    img_list = glob.glob(str(image_path)+ "/*.JPG")
    img_list = np.sort(img_list)

    for filename in img_list:
        filename = filename.split('/')[-1]
        if filename.split('.')[0][-1] == 'W':
            raw_path = os.path.join(image_path, filename)
            img_raw = cv2.imread(raw_path)
            img_disort = cv2.undistort(img_raw, CameraMatrix, kp,None, NewCameraMatrix) # None, NewCameraMatrix
            save_path = str(w_image_path) + '/'+ filename
            name_list_write.append(filename)
            cv2.imwrite(save_path, img_disort)
            print(save_path)
    write_intrinsics(intrinsics_path, NewCameraMatrix, w, h, name_list_write)
 
    img_list = glob.glob(str(image_path)+ "/*.JPG")            
    for filename in img_list:
        filename = filename.split('/')[-1]
        if filename.split('.')[0][-1] == 'Z':
            shutil.copy(os.path.join(image_path,filename), z_image_path)     
        
        