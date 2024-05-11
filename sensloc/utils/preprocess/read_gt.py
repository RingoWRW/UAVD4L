import numpy as np
import os
import glob
import pandas as pd
from scipy.spatial.transform import Rotation as R
import xmltodict
# from utils.transfrom import rotmat2qvec, rotation_to_quat
from .utils.transfrom import rotmat2qvec, rotation_to_quat
import argparse

def write_ground_truth(write_path, input_path, sequence):
    
    file_object = open(input_path, encoding='utf-8')
    try:
        all_the_xmlStr = file_object.read()
    finally:
        file_object.close()
        # transfer to dict
    dictdata = dict(xmltodict.parse(all_the_xmlStr))

    #for sequence in sequences:
    tilponts = dictdata['BlocksExchange']['Block']['Photogroups']['Photogroup']

    img_list = tilponts['Photo']

    with open(write_path, 'w') as file_w:
        for i in range(len(img_list)): #len(img_list)
            img_path = img_list[i]['ImagePath'].split('/')[-1]  
            pose_M00 = img_list[i]['Pose']['Rotation']['M_00']
            pose_M01 = img_list[i]['Pose']['Rotation']['M_01']
            pose_M02 = img_list[i]['Pose']['Rotation']['M_02']
            pose_M10 = img_list[i]['Pose']['Rotation']['M_10']
            pose_M11 = img_list[i]['Pose']['Rotation']['M_11']
            pose_M12 = img_list[i]['Pose']['Rotation']['M_12']
            pose_M20 = img_list[i]['Pose']['Rotation']['M_20']
            pose_M21 = img_list[i]['Pose']['Rotation']['M_21']
            pose_M22 = img_list[i]['Pose']['Rotation']['M_22']
            RotateMatrix = [pose_M00, pose_M01, pose_M02,
                            pose_M10, pose_M11, pose_M12,
                            pose_M20, pose_M21, pose_M22]
            a_qvec, a_np = rotation_to_quat(RotateMatrix)
            qw, qx, qy, qz = a_qvec[0], a_qvec[1], a_qvec[2], a_qvec[3]

            # degug
            # R = qvec2rotmat([qw, qx, qy, qz])

            x = img_list[i]['Pose']['Center']['x']
            y = img_list[i]['Pose']['Center']['y']
            z = str(float(img_list[i]['Pose']['Center']['z']  )) 
            
            R = np.asmatrix(a_np)
            t = np.array([float(x), float(y), float(z)])
            t = t[:, np.newaxis]
            
            t_w2c = np.array(-R.dot(t))
            out_line = img_path + ' ' + str(qw) + ' ' + str(qx) + ' ' + str(qy) + ' ' + str(qz) + ' ' + str(t_w2c[0][0]) + ' ' + str(t_w2c[1][0]) + ' ' + str(t_w2c[2][0]) + '\n'

            file_w.write(out_line)

write_ground_truth("/media/guan/data/CityofStars/new_xml/s7.txt","/media/guan/data/CityofStars/new_xml/query7 - AT - AT - cali - AT - export.xml",None)

def read_RTK_info(path):
    # input path: RTK file path
    # output: information about object from RTK (lat,lon,H,x,y,h,time)
    RTK_list = []
    RTK_list += glob.glob(path+"*.csv")
    RTK_info = {}

    for i in range(len(RTK_list)):
        ground_truth_information = pd.read_csv(RTK_list[i], encoding = 'gb2312')
        ground_truth_information_time = ground_truth_information["time"]
        ground_lat = ground_truth_information['lat']
        ground_lon = ground_truth_information['lon']
        ground_H = ground_truth_information['H']
        ground_time = ground_truth_information['time']
        for k in range(len(ground_truth_information_time)):

            RTK_info[ground_time[k]] = [ground_lat[k]]
            RTK_info[ground_time[k]].append(ground_lon[k])
            RTK_info[ground_time[k]].append(ground_H[k])

    return RTK_info

def read_SRT(path, GROUPLENGTH):
    # input path : SRT file path
    # output path: time and name

    SRT_list = []
    SRT_list += glob.glob(path+"*.SRT")

    name_list = []
    time_list = []
    SRT_infor = {}
    
    for i in (SRT_list):
        with open(i) as f:
            information = f.readlines()
            k = 0
            for j in range(len(information)//GROUPLENGTH):
                if j % 30 == 0: # 30 is fps
                    frameCount = eval(information[j * GROUPLENGTH])
                    time = information[(frameCount-1)*GROUPLENGTH+3].split('.')[0]
                    name = i.split('\\')[1][:-4]
                    name_list.append(name+str(k))
                    time_list.append(time)
                    SRT_infor[name+str(k)] = time
                    k += 1
    
    return SRT_infor

def query_txt_gt(uav_path, RTK_info, SRT_info, txt_write_path, intrinsic):

    txt_list = []
    txt_list += glob.glob(uav_path+"*.txt")

    query_info = {}

    for i in txt_list:
        with open(i) as f:
            informations = f.readlines()
            for information in informations:
                name = information.split(' ')[0]
                qw = information.split(' ')[1]
                qx = information.split(' ')[2]
                qy = information.split(' ')[3]
                qz = information.split(' ')[4]
                x = information.split(' ')[5]
                y = information.split(' ')[6]
                h = information.split(' ')[7]
                query_info[name] = [qw,qx,qy,qz,x,y,h]
    
    all_name = list(query_info.keys())
    SRT_info_name = list(SRT_info.keys())

    name_info_dict_time = {} 
    name_info_dict = {}
    for name in all_name:
        if name[:-4] in SRT_info_name:
            name_info_dict_time[SRT_info[name[:-4]]] = query_info[name]
            name_info_dict[SRT_info[name[:-4]]] = name
    
    time_query = list(name_info_dict_time.keys())
    time_RTK = list(RTK_info.keys())

    all_info_dict = {}
    name_list = []
    
    txt_write_path_pose = txt_write_path + '/' + 'pose.txt'
    txt_write_path_intrinsic = txt_write_path + '/' + 'intrinsic.txt'

    with open(txt_write_path_pose, 'w') as wf:
        for time in time_query:
            if time in time_RTK:
                write_name= name_info_dict[time]
                qw, qx, qy, qz, x, y, z = name_info_dict_time[time]
                lat, lon, H = RTK_info[time]
                # outline = write_name + ' ' + str(qw) + ' ' + str(qx) + ' ' + str(qy) + ' ' + str(qz) + ' ' + str(x) + ' ' + str(y) + ' ' + str(z[:-1]) + ' ' + str(lat) + ' ' + str(lon) + ' ' + str(H)
                outline = write_name + ' ' + str(qw) + ' ' + str(qx) + ' ' + str(qy) + ' ' + str(qz) + ' ' + str(x) + ' ' + str(y) + ' ' + str(z[:-1])
                wf.write(outline+'\n')

    w,h,fx,fy = intrinsic

    with open(txt_write_path_intrinsic, 'w') as wf:
        for time in time_query:
            if time in time_RTK:
                write_name= name_info_dict[time]
                outline = write_name + ' ' + 'PINHOLE' + ' ' + str(w) + ' ' + str(h) + ' ' + str(fx) + ' ' + str(fy) + ' ' + str(w//2) + ' ' + str(h//2) + '\n'
                wf.write(outline)
    
    print("end!")


def main():

    parser = argparse.ArgumentParser(description="Write .xml and object information.")
    parser.add_argument("--input_xml_path", default="D:/SomeCodes/SensLoc/datasets/jinxia/db/contextcapture/JX_with_wide_gt.xml")
    parser.add_argument("--write_uav_path", default="D:/SomeCodes/SensLoc/datasets/jinxia/query/txt/uav/")
    parser.add_argument("--sequence", default=[6,7,8,9,10,11])
    parser.add_argument("--RTK_csv_path", default="D:/SomeCodes/SensLoc/datasets/jinxia/query/txt/RTK/")
    parser.add_argument("--SRT_path", default="D:/SomeCodes/SensLoc/datasets/jinxia/query/video/SRT/W/")
    parser.add_argument("--GROUPLENGTH", default=14)
    parser.add_argument("--txt_write_path", default="D:/SomeCodes/SensLoc/datasets/jinxia/query/txt/object/")
    parser.add_argument('--intrinsic', default=[1920,1080,1400.3421449,1065.78947])
    args = parser.parse_args()

    
    write_ground_truth(args.write_uav_path, args.input_xml_path, args.sequence)
    
    RTK_info = read_RTK_info(args.RTK_csv_path)
    SRT_info = read_SRT(args.SRT_path, args.GROUPLENGTH)

    query_txt_gt(args.write_uav_path, RTK_info, SRT_info, args.txt_write_path, args.intrinsic)

if __name__ == "__main__":

    main()
