import cv2
import numpy as np
from osgeo import gdal

from scipy.ndimage import map_coordinates
import time
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import os


def DSM2npy (config):
    DSM_path = config["DSM_path"]
    save_path = config["DSM_npy_path"]
    # Transfer .tif to .npy and extract minZ
    # DSM_path = "/home/victor/disk1/Disk_E/Meshloc_phone/data_phone/DSM1/DSM1_DSM_merge.tif"
    dataset = gdal.Open(DSM_path)
    band = dataset.GetRasterBand(1)             #读取波段1

    area = band.ReadAsArray()
    mask = np.ma.masked_values(area, -9999)
    area_minZ = mask.min()

    # 将数组存储为txt文件
    if not os.path.exists(save_path):
        np.save(save_path, area)
    return area_minZ


def sample_points_on_line(line_equation, num_sample, x_minmax):
    # 根据斜截式计算 y 范围
    # y_min, y_max = (-A/B)*x_min - (C/B), (-A/B)*x_max - (C/B)
    A, B, C = line_equation[0], line_equation[1], line_equation[2]
    x_min, x_max = x_minmax[0], x_minmax[1]

    # 在 y 范围内均匀采样 num_points 个点
    x = np.linspace(x_min, x_max, num_sample)
    y = (-A/B)*x - (C/B)
    # y = np.linspace(y_min, y_max, num_points)

    return x, y

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

def interpolate_along_line(area, x, y, num_points):
    
    # 构造一个网格坐标系
    # xx, yy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    
    # 根据采样点的坐标，从图像中取得对应的像素值
    sample_values = map_coordinates(area, [y, x], order=1)
    # sample_values = np.max(sample_values, axis=1).reshape(-1, 1)

    # 将采样点对应的像素值重新排列成num_samples长度的数组
    sample_array = sample_values.reshape((num_points,))

    return sample_array


def pixel_to_world_coordinate(K,R,t,u,v):
    # 相机内参矩阵
    # K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # # 相机外参矩阵
    # R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
    # t = np.array([tx, ty, tz])

    # 将2D像素坐标转换为相机坐标系下的坐标
    p_camera = np.array([[u], [v], [1]])
    p_camera = np.linalg.inv(K).dot(p_camera)

    # 将相机坐标系下的坐标转换为世界坐标系下的坐标
    p_world = R.dot(p_camera) + t

    return p_world


def get_index_array(dataset):
    # 读取tif图像
    ds = dataset

    # 获取地理信息
    geotransform = ds.GetGeoTransform()
    x_origin, x_pixel_size, _, y_origin, _, y_pixel_size = geotransform

    # 获取图像大小
    rows, cols = ds.RasterYSize, ds.RasterXSize

    # 生成x坐标数组和y坐标数组
    x = np.arange(cols) * x_pixel_size + x_origin
    y = np.arange(rows) * y_pixel_size + y_origin

    # index_array = [x, y]
    # 生成索引数组
    # xx, yy = np.meshgrid(x, y)
    # index_array = np.stack([yy.ravel(), xx.ravel()], axis=1)
    return x, y


def line_equation_3d(point1, point2):
    """
    求两点所在直线的方程

    :param point1: 第一个点的坐标，形如 [x1, y1, z1]
    :param point2: 第二个点的坐标，形如 [x2, y2, z2]
    :return: 直线方程的系数，形如 [a, b, c, d]，表示 ax + by + cz + d = 0
    """
    # 将点转换成向量形式
    p1 = np.array(point1)
    p2 = np.array(point2)

    # 求解直线方向向量
    direction = p2 - p1

    # 求解直线方程系数
    a, b, c = direction
    d = -(a * p1[0] + b * p1[1] + c * p1[2])

    return [a, b, c, d]


def line_equation_2d(x1, y1, x2, y2):
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return [A, B, C]

def line_equation(A, B, Z):
    """
    计算射线和投影直线的方程
    :param A: A点坐标 (x,y,z)
    :param B: B点坐标 (x1,y1,z1)
    :param Z: 平面Z的值
    :return: 射线方程和投影直线方程
    """
    # 计算射线方程
    x, y, z = A
    x1, y1, z1 = B
    t = np.array([x1 - x, y1 - y, z1 - z])
    ray = lambda k: np.array([x, y, z]) + k * t

    # 计算投影直线方程
    k = (Z - z) / t[2]
    projection = ray(k)[:2]

    return ray, projection

def intersection(ray_eqn, Z):
    """
    计算射线与平面Z的交点
    :param ray_eqn: 射线方程
    :param Z: 平面Z的值
    :return: 交点坐标
    """
    # 计算k值
    k = (Z - ray_eqn(0)[2]) / (ray_eqn(1)[2] - ray_eqn(0)[2])
    
    # 计算交点坐标
    intersection_point = ray_eqn(k)
    
    return intersection_point


def geo_coords_to_array_index(x, y, geotransform):
    x_origin, x_pixel_size, _, y_origin, _, y_pixel_size = geotransform
    # print(x_origin, x_pixel_size, y_origin, y_pixel_size)
    col = ((x - x_origin) / x_pixel_size).astype(int)
    row = ((y - y_origin) / y_pixel_size).astype(int)
    return row, col


def find_z(ray_eqn, points):
    """
    计算投影直线上的点在射线方程中对应的Z值
    :param ray_eqn: 射线方程
    :param points: 投影直线上的点的平面坐标 (x,y)
    :return: Z值列表
    """
    # 计算k值
    k = (points[0] - ray_eqn(0)[0]) / (ray_eqn(1)[0] - ray_eqn(0)[0])
    # z_values = (-d-a*x-b*y)/c
    
    # 计算Z值
    z_values = [ray_eqn(k_i)[2] for k_i in k]
    
    return z_values

def read_DSM_config(config, area_minZ):
    DSM_path = config["DSM_path"]
    DSM_npy_path = config["DSM_npy_path"]
    num_sample = config["num_sample"]
    # area_minZ = config["area_minZ"]

    # 读取DSM
    start = time.time() 
    dataset = gdal.Open(DSM_path)
    band = dataset.GetRasterBand(1)             #读取波段1
    bandXSize = band.XSize
    bandYSize = band.YSize
    geotransform = dataset.GetGeoTransform()
    area = np.load(DSM_npy_path)

    print ("data size:",bandXSize,"x",bandYSize)
    print("area_minZ: ", area_minZ)
    print ("Loading time cost: ",time.time()-start,"s")
    dataset = None
    return geotransform, area, num_sample, area_minZ

def caculate_predictXYZ(K, pose, objPixelCoords, geotransform, area, num_sample, area_minZ):

    start_time_2 = time.time()

    # W2C->C2W
    # R  = qvec2rotmat(pose[:4]).T
    # t = np.array(pose[4:])
    # t= (-R.dot(t)).reshape([3,1])
    # query_t = t
    R = pose[:3, :3]
    t = pose[:3, 3].reshape([3,1])

    target = pixel_to_world_coordinate(K,R,t,objPixelCoords[0],objPixelCoords[1])

    ray_eqn, projection_eqn = line_equation(t, target, area_minZ)
    line2D_abcd = line_equation_2d(t[0], t[1], target[0], target[1])
    # print(projection_eqn, line2D_abcd)
    

    intersection_point = intersection(ray_eqn, area_minZ)
    x_minmax = [t[0],intersection_point[0]]

    # 直线采样
    x, y = sample_points_on_line(line2D_abcd, num_sample, x_minmax)

    # DSM采样,先将xy对应的地理信息坐标索引求到
    row, col = geo_coords_to_array_index(x, y, geotransform)
    sampleHeight = interpolate_along_line(area, col ,row, num_sample)

    #得到三维直线上的z值
    z_values = find_z(ray_eqn,[x, y])
    z_values = torch.tensor(z_values)
    z_values = z_values.squeeze()

    #寻找最近点
    sampleHeight = torch.tensor(sampleHeight)
    abs_x = torch.abs(z_values - sampleHeight)
    min_val, min_idx = torch.min(abs_x, dim=0)
    # print(min_val)

    # print ("Resulting time cost: ",time.time()-start_time_2,"s") 
    
    return [x[min_idx], y[min_idx], z_values[min_idx]]


def parse_pose_list(pose_list):
    poses = {}
    for i in range(len(pose_list)):
        
        q = pose_list[i][0],pose_list[i][1],pose_list[i][2],pose_list[i][3]
        t = pose_list[i][4],pose_list[i][5],pose_list[i][6]
        R = np.asmatrix(qvec2rotmat(q)).transpose() 
        
        T = np.identity(4)
        T[0:3,0:3] = R
        T[0:3,3] = -R.dot(t)  
        poses[i] = T
    return poses

def compute_3D_point(K_list, pose_list, config):


    points = []
    area_minZ = DSM2npy(config)
    geotransform, area, num_sample, area_minZ = read_DSM_config(config, area_minZ)
    cx,cy,fx,fy = K_list[:]
    K_w2c = np.array([
        [fx,0.0,cx],
        [0.0,fy,cy],
        [0.0,0.0,1.0],
    ])
    poses = parse_pose_list(pose_list)

    center_array = np.array([[0,0], [cx*2,0], [0, cy*2], [cx*2, cy*2]])

    for i in range(len(pose_list)):
        pose = poses[i]
        for k in range(len(center_array)):

            predict_XYZ = caculate_predictXYZ(K_w2c, pose, center_array[k], geotransform, area, num_sample, area_minZ)
            points.append(predict_XYZ)
    
    return points







if __name__=="__main__":
   

    num_sample=100
    DSM_path = "/media/guan/新加卷/north_dataset/Production_1 (2)_DSM_merge.tif"
    DSM_npy_path = "/media/guan/新加卷/north_dataset/my_array.npy"


    config = {
        "DSM_path":DSM_path,
        "DSM_npy_path":DSM_npy_path
    }
    DSM2npy(config)
