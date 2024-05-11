import numpy as np
import os
import glob
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import xmltodict
from pathlib import Path
# from utils.transfrom import qvec2rotmat,rotmat2qvec, compute_pixel_focal
from .utils.transfrom import qvec2rotmat,rotmat2qvec, compute_pixel_focal
import pdb
import sys




def parse_image_list(path):
    images = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            name, *data = line.split()
            model, width, height, *params = data
            params = np.array(params, float)
            images[os.path.basename(name).split('.')[0]] = (model, int(width), int(height), params)
  
    assert len(images) > 0
    return images

def parse_pose_list(path, origin_coord):
    poses = {}
    X = []
    Y = []
    Z = [] # 存放x，y的位置
    q_list = []
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = (data[0].split('/')[-1]).split('.')[0]
            q, t = np.split(np.array(data[1:], float), [4])
            
            R = np.asmatrix(qvec2rotmat(q)).transpose()  #c2w
            
            T = np.identity(4)
            T[0:3,0:3] = R
            T[0:3,3] = -R.dot(t)   #!  c2w

            if origin_coord is not None:
                origin_coord = np.array(origin_coord)
                T[0:3,3] -= origin_coord
            transf = np.array([
                [1,0,0,0],
                [0,-1,0,0],
                [0,0,-1,0],
                [0,0,0,1.],
            ])
            T = T @ transf

            X.append(T[0,3])
            Y.append(T[1,3])
            Z.append(T[2,3])
            q_list.append([q])
            
            
            poses[name] = T
            
    
    assert len(poses) > 0
    return poses, X, Y, Z, q_list

def euler2quad(euler):
    """
    欧拉转四元数（xyzw）
    """
    ret = R.from_euler('xyz', euler,degrees=True)
    quad = ret.as_quat()

    return quad

def write_pose(db_pose_file, db_intrinsic_file, euler_path, write_path, intrinsic, origin_coord=[399961, 3138435, 0]):
    """
    db_pose_file: wumu_pose
    db_intrinsic_file: wumu_intrinsic
    euler_path: blender render angle (.txt)
    write_path: render path
    intrinsic: [w,h,fx,fy]
    origin_coord:
    """
    poses, X, Y, Z, _ = parse_pose_list(db_pose_file, origin_coord)
    wumu_intrinsic = parse_image_list(db_intrinsic_file)

    write_pose_path = write_path + '/' + 'pose.txt'
    write_intrinsic_path = write_path + '/' + 'intrinsic.txt'

    all_euler = []
    quad_name = ['z', 'y', 'x', 'q', 'h', 'zq', 'yq', 'zh', 'yh', 'xz', 'xzz', 'xzzz', 'xzzzz', 'xy', 'xyy', 'xyyy']
    all_quad = []

    names = list(wumu_intrinsic.keys())

    with open(euler_path, 'r') as f_r:
        for line in f_r:
            line = line.strip('\n')
            euler = line.split(' ')
            all_euler.append([-180+eval(euler[1]),eval(euler[2]), eval(euler[3])])
            temp_quad = euler2quad(all_euler[-1])
            ret = R.from_quat(temp_quad)
            matrix = ret.as_matrix()
            temp_quad = rotmat2qvec(matrix.T)
            all_quad.append(temp_quad)

    with open(write_pose_path, 'w') as file_pose:
        for name in names:
            for i in range(len(all_euler)):
                R1 = np.asmatrix(qvec2rotmat(all_quad[i]))
                T = np.identity(4)
                T[0:3, 0:3] = R1
                T[0:3, 3] = -R1.dot(np.array([X[i]+origin_coord[0], Y[i]+origin_coord[1], 200+origin_coord[2]]))
                out_line =  str(name) + quad_name[i] + '.JPG' + ' ' + str(all_quad[i][0]) + ' ' + str(all_quad[i][1]) + ' ' + str(all_quad[i][2]) + ' ' + str(all_quad[i][3]) + ' ' + str(T[0:3, 3][0]) + ' ' + str(T[0:3, 3][1]) + ' ' + str(T[0:3, 3][2]) + '\n'
                file_pose.write(out_line)

    w,h,fx,fy = intrinsic
    with open(write_intrinsic_path, 'w') as file_intri:
        for name in names:
            for i in range(len(all_euler)):
                out_line = 'db/' + str(name) + '.JPG' + ' ' + 'PINHOLE' + ' ' + str(w) + ' ' + str(h) + ' ' + str(fx) + ' ' + str(fy) + ' ' + str(w//2) + ' ' + str(h//2) + '\n'
                file_intri.write(out_line)

def compute_k_b(start_point, end_point):
    #
    k = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
    b = 0.5 * (start_point[1] - k * start_point[0] + end_point[1] - k * end_point[0])

    return k, b

def generate_end_point(k, b, point, number):
    # point:[min, max]
    new_x = np.linspace(point[0], point[1], number)
    new_y = []
    for i in range(len(new_x)):
        y = k * new_x[i] + b
        new_y.append(y)

    return new_x, new_y


def interpolate_before(xml_path, write_path, orientation_path, sequences, number, base_height, add_height, intrinsic):

    pose_file = write_path + '/' + 'db_pose.txt'
    intrinsic_file = write_path + '/' + 'db_intrinsics.txt'

    all_euler = []
    quad_name = ['z', 'y', 'x', 'q', 'h', 'zq', 'yq', 'zh', 'yh', 'xz', 'xzz', 'xzzz', 'xzzzz', 'xy', 'xyy', 'xyyy']
    all_quad = []

    w, h, sensorW, sensorH, focal_len = intrinsic
    fx, fy = compute_pixel_focal(sensorW, sensorH, focal_len, w, h)

    file_object = open(xml_path, encoding='utf-8')
    try:
        all_the_xmlStr = file_object.read()
    finally:
        file_object.close()
        # transfer to dict
    dictdata = dict(xmltodict.parse(all_the_xmlStr))

    position = []

    for sequence in sequences:
        tilponts = dictdata['BlocksExchange']['Block']['Photogroups']['Photogroup'][sequence]  

        img_list = tilponts['Photo']
  
        for i in range(len(img_list)): #len(img_list)
            x = eval(img_list[i]['Pose']['Center']['x'])
            y = eval(img_list[i]['Pose']['Center']['y'])
            position.append([x,y])

    position = np.array(position)

    # left_boundary
    x_min_index = np.argmin(position[:,0], axis=0)
    left_boundary = [401676, 3130905]#1[400736,3131076]#[400355, 3130984]#position[x_min_index]  [min_x, min_y]
    print(left_boundary)
    # right_boundary
    x_max_index = np.argmax(position[:,0], axis=0)
    right_boundary = [401944, 3131906]#[401539, 3131077]#[400749, 3130984]#position[x_max_index]  [max_x, min_y]
    # up_boundary
    y_max_index = np.argmax(position[:,1], axis=0)
    up_boundary = [401945, 3131426]#[401540, 3131565]#position[y_max_index]  [max_x, max_y]
    # down_boundary
    y_min_index = np.argmin(position[:,1], axis=0)
    down_boundary = [401677, 3131427]#[400737, 3131565]#position[y_min_index]  [min_x, max_y]
    
    k_left_down, b_left_down = compute_k_b(left_boundary, down_boundary)
    k_right_up, b_right_up = compute_k_b(right_boundary, up_boundary)

    x_left_down, y_left_down = generate_end_point(k_left_down, b_left_down, [left_boundary[0], up_boundary[0]], number)
    x_right_up, y_right_up = generate_end_point(k_right_up, b_right_up, [left_boundary[0], up_boundary[0]], number)

    with open(orientation_path, 'r') as f_r:
        for line in f_r:
            line = line.strip('\n')
            euler = line.split(' ')
            all_euler.append([-180+eval(euler[1]),eval(euler[2]), eval(euler[3])])
            temp_quad = euler2quad(all_euler[-1])
            ret = R.from_quat(temp_quad)
            matrix = ret.as_matrix()
            temp_quad = rotmat2qvec(matrix.T)
            all_quad.append(temp_quad)
    
    with open(pose_file, 'w') as file_pose:
        with open(intrinsic_file, 'w') as file_intri:
            for j in range(len(x_left_down)):
                k_temp, b_temp = compute_k_b([x_left_down[j],y_left_down[j]], [x_right_up[j], y_right_up[j]])
                x_temp_list, y_temp_list = generate_end_point(k_temp, b_temp, [x_left_down[j], x_right_up[j]], number)
                for i in range(len(all_euler)):
                    for k in range(len(x_temp_list)):
                        R1 = np.asmatrix(qvec2rotmat(all_quad[i]))
                        T = np.identity(4)
                        T[0:3, 0:3] = R1
                        T[0:3, 3] = -R1.dot(np.array([x_temp_list[k], y_temp_list[k], base_height + add_height]))

                        pitch, roll, yaw = int(all_euler[i][0]+180), int(all_euler[i][1]), int(all_euler[i][2])
                        word_x, word_y, word_z =  int(x_temp_list[k]), int(y_temp_list[k]), base_height+add_height
                        name = str(word_x) + '@' + str(word_y) + '@' + str(word_z) + '@' + str(pitch) + '@' + str(yaw) + '@' + str(roll)

                        out_line =  name + '.jpg' + ' ' + str(all_quad[i][0]) + ' ' + str(all_quad[i][1]) + ' ' + str(all_quad[i][2]) + ' ' + str(all_quad[i][3]) + ' ' + str(T[0:3, 3][0]) + ' ' + str(T[0:3, 3][1]) + ' ' + str(T[0:3, 3][2]) + '\n'
                        file_pose.write(out_line)
                        out_line_in = name + '.jpg' + ' ' + 'PINHOLE' + ' ' + str(w) + ' ' + str(h) + ' ' + str(fx) + ' ' + str(fy) + ' ' + str(w//2) + ' ' + str(h//2) + '\n'
                        file_intri.write(out_line_in)

    print('DB pose and intrinsic files has finished generated !!!  ')

def interpolate(xml_path, write_path, orientation_path, sequences, interval, base_height, add_height, intrinsic):

    pose_file = write_path + '/' + 'db_pose.txt'
    intrinsic_file = write_path + '/' + 'db_intrinsics.txt'

    all_euler = []
    quad_name = ['z', 'y', 'x', 'q', 'h', 'zq', 'yq', 'zh', 'yh', 'xz', 'xzz', 'xzzz', 'xzzzz', 'xy', 'xyy', 'xyyy']
    all_quad = []

    w, h, sensorW, sensorH, focal_len = intrinsic
    fx, fy = compute_pixel_focal(sensorW, sensorH, focal_len, w, h)

    file_object = open(xml_path, encoding='utf-8')
    try:
        all_the_xmlStr = file_object.read()
    finally:
        file_object.close()
        # transfer to dict
    dictdata = dict(xmltodict.parse(all_the_xmlStr))

    position = []

    for sequence in sequences:
        tilponts = dictdata['BlocksExchange']['Block']['Photogroups']['Photogroup'][sequence]  

        img_list = tilponts['Photo']
  
        for i in range(len(img_list)): #len(img_list)
            x = eval(img_list[i]['Pose']['Center']['x'])
            y = eval(img_list[i]['Pose']['Center']['y'])
            position.append([x,y])

    position = np.array(position)

    # left_boundary ##xml
    x_min_index = np.argmin(position[:,0], axis=0)
    left_boundary = position[x_min_index]  #[min_x, min_y]
    print(left_boundary)
    # right_boundary
    x_max_index = np.argmax(position[:,0], axis=0)
    right_boundary = position[x_max_index]  #[max_x, min_y]
    # up_boundary
    y_max_index = np.argmax(position[:,1], axis=0)
    up_boundary = position[y_max_index]  #[max_x, max_y]
    # down_boundary
    y_min_index = np.argmin(position[:,1], axis=0)
    down_boundary = position[y_min_index]  #[min_x, max_y]

    # # no xml
    min_x, min_y, max_x, max_y = 400683,3131100,401563,3131566
    left_up = [min_x, max_y-1] 
    left_donw = [min_x+1, min_y-1]
    right_up = [max_x, max_y]
    right_down = [max_x+1, min_y]
    
    k_down, b_down = compute_k_b(left_donw, right_down)
    k_up, b_up = compute_k_b(left_up, right_up)

    number_x = (max_x - min_x) // interval
    number_y = (max_y - min_y) // interval

    x_left_down, y_left_down = generate_end_point(k_down, b_down, [left_donw[0], right_down[0]], number_x)
    x_left_up, y_left_up = generate_end_point(k_up, b_up, [left_up[0], right_up[0]], number_x)

    with open(orientation_path, 'r') as f_r:
        for line in f_r:
            line = line.strip('\n')
            euler = line.split(' ')
            all_euler.append([-180+eval(euler[1]),eval(euler[2]), eval(euler[3])])
            temp_quad = euler2quad(all_euler[-1])
            ret = R.from_quat(temp_quad)
            matrix = ret.as_matrix()
            temp_quad = rotmat2qvec(matrix.T)
            all_quad.append(temp_quad)
    xx_list = []
    yy_list = []
    with open(pose_file, 'w') as file_pose:
        with open(intrinsic_file, 'w') as file_intri:
            for j in range(len(x_left_down)):
                k_temp, b_temp = compute_k_b([x_left_down[j],y_left_down[j]], [x_left_up[j], y_left_up[j]])
                x_temp_list, y_temp_list = generate_end_point(k_temp, b_temp, [x_left_down[j], x_left_up[j]], number_y)
                for i in range(len(all_euler)):
                    for k in range(len(x_temp_list)):
                        R1 = np.asmatrix(qvec2rotmat(all_quad[i]))
                        T = np.identity(4)
                        T[0:3, 0:3] = R1
                        T[0:3, 3] = -R1.dot(np.array([x_temp_list[k], y_temp_list[k], base_height + add_height]))
                        
                        xx_list.append(x_temp_list[k])
                        yy_list.append(y_temp_list[k])

                        pitch, roll, yaw = int(all_euler[i][0]+180), int(all_euler[i][1]), int(all_euler[i][2])
                        word_x, word_y, word_z =  int(x_temp_list[k]), int(y_temp_list[k]), base_height+add_height
                        name = str(word_x) + '@' + str(word_y) + '@' + str(word_z) + '@' + str(pitch) + '@' + str(yaw) + '@' + str(roll)

                        out_line =  name + '.jpg' + ' ' + str(all_quad[i][0]) + ' ' + str(all_quad[i][1]) + ' ' + str(all_quad[i][2]) + ' ' + str(all_quad[i][3]) + ' ' + str(T[0:3, 3][0]) + ' ' + str(T[0:3, 3][1]) + ' ' + str(T[0:3, 3][2]) + '\n'
                        file_pose.write(out_line)
                        out_line_in = name + '.jpg' + ' ' + 'PINHOLE' + ' ' + str(w) + ' ' + str(h) + ' ' + str(fx) + ' ' + str(fy) + ' ' + str(w//2) + ' ' + str(h//2) + '\n'
                        file_intri.write(out_line_in)

    plt.scatter(xx_list, yy_list)
    print(len(xx_list)//16)
    plt.show()
    print('DB pose and intrinsic files has finished generated !!!  ')

def main(config):

    input_xml_path = config["xml_path"] # xml file
    write_render_path = config["write_path"] # save path
    euler_txt = config["orientation_path"] # euler.txt
    sequence = config["sequences"] # 
    number = config["number"]
    add_height = config["add_height"]
    base_height = config["base_height"]
    intrinsic = config["intrinsic"]
    interpolate(input_xml_path, write_render_path, euler_txt, sequence, number, base_height, add_height, intrinsic)
    # interpolate(args.input_xml_path, args.write_render_path, args.euler_txt, args.sequence, 10, 100,[1920,1080,1400.3421449,1065.78947])
    # write_pose(args.db_wu_pose, args.db_wu_intrinsic, args.euler_txt, args.write_render_path, args.intrinsic)




if __name__ == "__main__":
    # config = {
    #     "generate_render_pose": {
    #     "image": {
    #         "enable": True,
    #         "xml_path": "/media/guan/data/CityofStars/3D-models/cc/reference.xml",
    #         "write_path": "/media/guan/data/CityofStars/Render_db_5",
    #         "orientation_path": "./datasets/CityofStars/Render_db_5/bamu_euler.txt",
    #         "sequences": [
    #             0,
    #             1,
    #             2,
    #             3,
    #             4
    #         ],
    #         "number": 10,
    #         "base_height": 0,
    #         "add_height": 150,
    #         "intrinsic": [
    #             4056,
    #             3040,
    #             6.29,
    #             4.71,
    #             4.52
    #         ]
    #     }
    # },
    # } 
    config = {
            "enable": True,
            "xml_path": "G:/CityofStars/3D-models/cc/reference.xml",
            "write_path": "G:/CityofStars/Render_75_150_1",
            "orientation_path": "G:/CityofStars/Render_50_100_1/bamu_euler.txt",
            "sequences": [
                0,
                1,
                2,
                3,
                4
            ],
            "number": 75,
            "base_height": 1600,
            "add_height": 150,
            "intrinsic": [
                4056,
                3040,
                6.29,
                4.71,
                4.52
            ]
        }
    #! read cc boundary
    main(config)
   

    





