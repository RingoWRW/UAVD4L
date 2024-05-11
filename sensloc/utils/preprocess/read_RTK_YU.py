import numpy as np
import os
import glob
import pandas as pd
from pyproj import Transformer
import pyproj
from pyproj import CRS
from scipy.spatial.transform import Rotation as R
from datetime import datetime
from PIL import Image
import argparse
import exifread
from pyexiv2 import Image
from pathlib import Path
def get_dji_exif(exif_file):
    
    with open(exif_file, "rb") as f:
        # 读取exif信息
        exif_data = exifread.process_file(f)
        # 获取GPS对应的标签
        gps_latitude_tag = "GPS GPSLatitude"
        gps_latitude_ref_tag = "GPS GPSLatitudeRef"
        gps_longitude_tag = "GPS GPSLongitude"
        gps_longitude_ref_tag = "GPS GPSLongitudeRef"
        gps_altitude_tag = "GPS GPSAltitude"
        time = "Image DateTime"
        if gps_latitude_tag in exif_data and gps_latitude_ref_tag in exif_data and gps_longitude_tag in exif_data and gps_longitude_ref_tag in exif_data:
        
            # 获取GPS纬度和经度的分数值和方向值
            gps_latitude_value = exif_data[gps_latitude_tag].values
            gps_latitude_ref_value = exif_data[gps_latitude_ref_tag].values
            gps_longitude_value = exif_data[gps_longitude_tag].values
            gps_longitude_ref_value = exif_data[gps_longitude_ref_tag].values
            gps_altitude_value = exif_data[gps_altitude_tag].values
            # 将GPS纬度和经度的分数值转换为浮点数值
            gps_latitude = (float(gps_latitude_value[0].num) / float(gps_latitude_value[0].den) +
                            (float(gps_latitude_value[1].num) / float(gps_latitude_value[1].den)) / 60.0 +
                            (float(gps_latitude_value[2].num) / float(gps_latitude_value[2].den)) / 3600.0)
            gps_longitude = (float(gps_longitude_value[0].num) / float(gps_longitude_value[0].den) +
                             (float(gps_longitude_value[1].num) / float(gps_longitude_value[1].den)) / 60.0 +
                             (float(gps_longitude_value[2].num) / float(gps_longitude_value[2].den)) / 3600.0)
            gps_altitude = eval(str(gps_altitude_value[0]).split('/')[0]) / eval(str(gps_altitude_value[0]).split('/')[1])
            really_time = exif_data[time].values
        
            # 根据GPS纬度和经度的方向值，判断正负号
            if gps_latitude_ref_value != "N":
                gps_latitude = -gps_latitude
            if gps_longitude_ref_value != "E":
                gps_longitude = -gps_longitude
            # 返回这些值
            return gps_latitude, gps_longitude, gps_altitude, really_time
        else:
            # 如果不存在这些标签，返回None
            return None   
        
def read_SRT_to_txt(input_SRT_file, output_file, time_dict):
    """
    input_SRT_file: SRT path file
    out_txt_file: intrinsic path file
    output_query_file: query path file
    intrinsic:[w, h, sensorW, sensorH, cx, cy, focal]
    group_type: video type (dict)
    """

    GROUPLENGTH = 6

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

    
    with open(output_file,'w') as file_w:
        for j in range(SRT_num):
            with open(SRT_root_file[j]) as f:
                information = f.readlines()
            
            for i in range(len(information)//GROUPLENGTH):
                if i % 3 == 0:
                    try:
                        time = information[i * GROUPLENGTH +3 ].split('\n')[0][:-4]
                        time = time.replace('-', ':')
                        all_infor = information[i * GROUPLENGTH +4].split(' ')
                        lat = eval(all_infor[18][:-1])
                        lon = eval(all_infor[20][:-1])
                        alt = eval(all_infor[24][:-1])
                        picture_name = time_dict[time]
                        crs_CGCS2000 = CRS.from_wkt('PROJCS["CGCS_2000_3_Degree_GK_CM_114E",GEOGCS["GCS_China_Geodetic_Coordinate_System_2000",DATUM["D_China_2000",SPHEROID["CGCS2000",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",114.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')  # degree
                        crs_WGS84 = CRS.from_epsg(4326)
                        from_crs = crs_WGS84
                        to_cgcs = crs_CGCS2000
                        transformer = Transformer.from_crs(from_crs, to_cgcs)
                        new_x, new_y = transformer.transform(lat, lon)

                        out_line_str =  picture_name[0] + ' ' + str(new_x) + ' ' + str(new_y) + ' ' + str(alt) +  '\n'
                        file_w.write(out_line_str)   
                    except:
                        continue
            print(SRT_root_file[j])
            f.close()
                        
    print("-----------RTK has finished read.------------")

def read_exif_data(folder_path):
    """
    读取指定文件夹中每个图像文件的EXIF信息, 并返回一个字典, 其中包含每个图像的EXIF数据
    """
    
    time_dict = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".JPG"):

            # 打开图像文件并获取exif信息
            image_path = os.path.join(folder_path, filename)
            gps_latitude, gps_longitude, gps_altitude, really_time = get_dji_exif(image_path)     

            # 保存exif信息到字典中
            # if result:
            #     exif_dict[filename] = result
            if really_time:
                time_dict[really_time] = [filename]

            # exif_file.append(filename)

    # return exif_dict, exif_file
    return time_dict

def main(input_EXIF_path: Path,
         input_RTK_path: Path,
         output_path: Path):

    time_dict = read_exif_data(input_EXIF_path)
    read_SRT_to_txt(input_RTK_path, output_path, time_dict)

if __name__ == "__main__":

    main()