import numpy as np
import os
import glob
import pandas as pd
import pyproj
from pyproj import Transformer
from pyproj import CRS
from scipy.spatial.transform import Rotation as R
import json
import argparse
from .utils.transfrom import qvec2rotmat, rotmat2qvec, compute_pixel_focal

def write_to_line(information, GROUPLENGTH, i, SRT_name, fps, sensorH, sensorW, focal_len, width, height, video_type):
    """
    information: SRT information
    GROUPLENGTH: group length
    """
    frameCount = eval(information[i * GROUPLENGTH])
    position = information[(frameCount-1)*GROUPLENGTH+5].split(':')
    lat, lon, alt = eval(position[1].split(']')[0]),eval(position[2].split(']')[0]), eval(position[3].split(' ')[1])
    oritation = information[(frameCount-1)*GROUPLENGTH+8].split(' ')
    yaw, pitch, roll = eval(oritation[1]),eval(oritation[3]),eval(oritation[5][:-1])
    
    if video_type == 'Z':
        focal_info = information[(frameCount-1)*GROUPLENGTH+4].split(' ')[9]
        focal_len_com = eval(focal_info[:-1])
        focal_len = 6.83 * focal_len_com / 31.7

    # GPS转世界坐标
    wgs84 = pyproj.CRS('EPSG:4326')
    cgcs2000 = pyproj.CRS('EPSG:4539')
    # from_crs = crs_WGS84
    # to_cgcs = crs_CGCS2000
    transformer = pyproj.Transformer.from_crs(wgs84, cgcs2000, always_xy=True)
    # transformer = Transformer.from_crs(from_crs, to_cgcs)
    new_x, new_y = transformer.transform(lon, lat)

    # crs_CGCS2000 = CRS.from_wkt('PROJCS["CGCS_2000_3_Degree_GK_CM_114E",GEOGCS["GCS_China_Geodetic_Coordinate_System_2000",DATUM["D_China_2000",SPHEROID["CGCS2000",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",114.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')  # degree
    # crs_WGS84 = CRS.from_epsg(4326)
    # from_crs = crs_WGS84
    # to_cgcs = crs_CGCS2000
    # transformer = Transformer.from_crs(from_crs, to_cgcs)
    # new_x, new_y = transformer.transform(lat, lon)

    # 欧拉角转四元数
    euler = [yaw,pitch,roll]
    ret = R.from_euler('zxy',[float(euler[0]), 90-float(euler[1]), float(euler[2])],degrees=True)
    R_matrix = ret.as_matrix()
    qw, qx, qy, qz = rotmat2qvec(R_matrix)

    # w2c
    q = [qw, qx, qy, qz]
    R1 = np.asmatrix(qvec2rotmat(q))
    T = np.identity(4)
    T[0:3, 0:3] = R1
    T[0:3, 3] = -R1.dot(np.array([new_x, new_y, alt]))

    out_line_pose_str =  SRT_name + str(int((i) // fps)) + '.jpg'  + ' ' + str(qw) + ' ' + str(qx) + ' ' + str(qy) + ' ' + str(qz) + ' ' + str(T[0:3, 3][0]) + ' ' + str(T[0:3, 3][1]) + ' ' + str(T[0:3, 3][2]) + '\n'
    
    fx, fy = compute_pixel_focal(sensorW, sensorH, focal_len, width, height)
    cx, cy = width // 2, height // 2
    out_line_intrinsic_str =  SRT_name + str(int((i+1) / fps)) + '.jpg' + ' ' + 'PINHOLE' + ' ' + str(width) + ' ' + str(height) + ' ' + str(fx) + ' ' + str(fy) + ' ' + str(cx) + ' ' + str(cy) + '\n'

    return out_line_pose_str, out_line_intrinsic_str

def write_time_to_line(information, GROUPLENGTH, i):
    
    frameCount = eval(information[i * GROUPLENGTH])
    time = information[(frameCount-1)*GROUPLENGTH+3]
    
    return time
    
def read_SRT_to_txt(input_SRT_file, write_path, fps, group_type):
    """
    input_SRT_file: SRT path file
    out_txt_file: intrinsic path file
    output_query_file: query path file
    intrinsic:[w, h, sensorW, sensorH, cx, cy, focal]
    group_type: video type (dict)
    """
    
    SRT_root_file = []
    for i in os.listdir(input_SRT_file):
        full_path = os.path.join(input_SRT_file, i)
        if full_path.endswith('.SRT'): 
            SRT_root_file.append(full_path)
    
    SRT_num = len(SRT_root_file)
    
    if SRT_num != 0:
        State = True
    else:
        print('The srt path is incorrect, or the number of file is 0.')

    output_intrinsic_file = write_path + '/' + 'intrinsics'
    output_query_file = write_path + '/' + 'poses'
    output_time_file = write_path + '/' + 'time'

    # video_type = input_SRT_file.split('/')[-2]
    # GROUPLENGTH = group_type[video_type]

    if os.path.isdir(output_intrinsic_file):
        pass
    else:
        os.makedirs(output_intrinsic_file)

    if os.path.isdir(output_query_file):
        pass
    else:
        os.makedirs(output_query_file)
        
    if os.path.isdir(output_time_file):
            pass
    else:
        os.makedirs(output_time_file)

    SRT_name_file_w = output_query_file + '/'  + 'w' + '_' + 'pose.txt'
    SRT_name_file_intrinsic_w = output_intrinsic_file + '/'  + 'w' + '_' + 'intrinsic.txt'
    SRT_name_file_z = output_query_file + '/'  + 'z' + '_' + 'pose.txt'
    SRT_name_file_intrinsic_z = output_intrinsic_file + '/'  + 'z' + '_' + 'intrinsic.txt'
    SRT_name_file_t = output_query_file + '/'  + 't' + '_' + 'pose.txt'
    SRT_name_file_intrinsic_t = output_intrinsic_file + '/'  + 't' + '_' + 'intrinsic.txt'
    t_name = output_time_file + '/' + 'time.txt'
    
    Time_FLAG = True
    
    for index in range(SRT_num):
        with open(SRT_root_file[index]) as f:
            information = f.readlines()
        
        SRT_name = SRT_root_file[index].split('/')[-1][:-4]
        video_type = SRT_root_file[index].split('/')[-1][-5]
        
        
        if video_type == 'W':
            GROUPLENGTH = group_type[video_type]
            sensorH, sensorW, focal_len, width, height = 4.71, 6.29, 4.5, 1920, 1080
            with open(SRT_name_file_w, 'w') as f_wp:
                with open(SRT_name_file_intrinsic_w, 'w') as f_wi:
                    for i in range(len(information)//GROUPLENGTH):
                        if i % fps == 0:
                            out_line_pose, out_line_intrinsic = write_to_line(information, GROUPLENGTH, i, SRT_name, fps, sensorH, sensorW, focal_len, width, height, video_type)
                            f_wp.write(out_line_pose)   
                            f_wi.write(out_line_intrinsic)
        elif video_type == 'T':
            GROUPLENGTH = group_type[video_type]
            sensorH, sensorW, focal_len, width, height = 6.144, 7.68, 13.5, 640, 512
            with open(SRT_name_file_t, 'w') as f_tp:
                with open(SRT_name_file_intrinsic_t, 'w') as f_ti:
                    for i in range(len(information)//GROUPLENGTH):
                        if i % fps == 0:
                            out_line_pose, out_line_intrinsic = write_to_line(information, GROUPLENGTH, i, SRT_name, fps, sensorH, sensorW, focal_len, width, height, video_type)
                            f_tp.write(out_line_pose)   
                            f_ti.write(out_line_intrinsic)
        elif video_type == 'Z':
            GROUPLENGTH = group_type[video_type]
            sensorH, sensorW, focal_len, width, height = 5.56, 7.41, 4.5, 1920, 1080
            with open(SRT_name_file_z, 'w') as f_zp:
                with open(SRT_name_file_intrinsic_z, 'w') as f_zi:
                    with open(t_name, 'w') as f_time:
                        for i in range(len(information)//GROUPLENGTH):
                            if i % fps == 0:
                                out_line_pose, out_line_intrinsic = write_to_line(information, GROUPLENGTH, i, SRT_name, fps, sensorH, sensorW, focal_len, width, height, video_type)
                                time = write_time_to_line(information, GROUPLENGTH, i)
                                outline = out_line_pose.split(' ')[0] + time
                                f_zp.write(out_line_pose)   
                                f_zi.write(out_line_intrinsic)
                                f_time.write(outline)
        

            

    print("------------intrinsic and pose has finished read.------------")

def load_json(json_path):
    # z_json_file
    json_list = []
    json_list = glob.glob(json_path + '/*.json')

    detection_dict = {}
    # {'img_name':[{'lable':[point_x, point_y]}], }
    for i in json_list:
        with open(i) as fp:
            one_json_info = json.load(fp)
            obj_pos_dict = {}
            label_dict = {}
            center_point_list = []
            img_name = one_json_info["imagePath"]
            label = one_json_info["shapes"][0]["label"]
            points = [0.5*(one_json_info["shapes"][0]["points"][0][0] + one_json_info["shapes"][0]["points"][1][0]),
                    0.5 * (one_json_info["shapes"][0]["points"][0][1] + one_json_info["shapes"][0]["points"][1][1])]
            label_dict[label] = points
            center_point_list.append(label_dict)
            detection_dict[img_name] = center_point_list
    
    return detection_dict

def main_bak():
    
    parser = argparse.ArgumentParser(description="write SRT information (name qw qx qy qz x y z) in txt file")
    parser.add_argument("--input_SRT_path", default="/home/ubuntu/Documents/code/SensLoc/datasets/Tibet/Queries/Raw/video/seq1")
    # parser.add_argument("--output_intrinsic_path", default="D:/SomeCodes/SensLoc/datasets/jinxia/query/txt/queries")
    parser.add_argument("--output_path", default="/home/ubuntu/Documents/code/SensLoc/datasets/Tibet/Queries/process/video/seq1")
    # parser.add_argument("--intrinsic", default=[1920, 1080, 6.16, 4.62,  4.5])
    parser.add_argument("--group_type", default={'S':15,'T':14,'W':14,"Z":15})
    parser.add_argument("--fps", default=30)
    args = parser.parse_args()
    # intrinsic = [args.imgWidth, args.imgHeight, args.sensorW, args.sensorH, args.imgWidth//2, args.imgHeight//2, args.focal_length]

    read_SRT_to_txt(args.input_SRT_path, args.output_path, args.fps, args.group_type)

def main(config):
    
    # intrinsic = [args.imgWidth, args.imgHeight, args.sensorW, args.sensorH, args.imgWidth//2, args.imgHeight//2, args.focal_length]
    input_SRT_path = config["input_SRT_data"]
    output_path = config["output_path"]
    fps = config["fps"]
    group_type = config["group_type"]

    read_SRT_to_txt(input_SRT_path, output_path, fps, group_type)
if __name__ == "__main__":

    main_bak()