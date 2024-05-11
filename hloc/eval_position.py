import pyproj
import numpy as np
from geopy.distance import distance


def load_gt(path, transformer):
    d = {}
    with open(path, 'r') as file:
        for line in file.read().rstrip().split('\n'):
            key = line.split(' ')[0].split('_')[-2]
            value = list(map(float,line.rstrip().split(' ')[1:]))
            x, y, z = transformer.transform(value[1], value[0], value[2])
            d[key] = [x, y, z]
    return d

def load_predict(path, object_name):
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

def eval_absolute_poes(gt_list, pred_list, result):
    errors_t = []
    # errors_R = []
    # image_num = len(test_list)
    image_num = len(pred_list)
    for name in pred_list.keys():
        if name not in gt_list.keys():
            e_t = np.inf
            # e_R = 180.
            continue
        else:
            gt = gt_list[name]
            t_gt = np.array(gt[:])
            pred = pred_list[name]
            t = np.array(pred[:])

            # e_t = distance(t_gt, t).meters * 100.0
            e_t = np.linalg.norm(t_gt - t, axis=0)
            # e_t = np.linalg.norm(t_gt - t, axis=0)
            # cos = np.clip((np.trace(np.dot(R_gt, R.T)) - 1) / 2, -1., 1.)
            # e_R = np.rad2deg(np.abs(np.arccos(cos)))
        # print(name,e_t,e_R)
        errors_t.append(e_t)
        # errors_R.append(e_R)

    errors_t = np.array(errors_t)
    med_t = np.median(errors_t)
    max_t = np.max(errors_t)
    min_t = np.min(errors_t)

    out = f'\nTest image nums: {image_num}'
    out += f'\nMedian errors: {med_t*100:.3f}cm'
    out += f'\nMax errors: {max_t*100:.3f}cm'
    out += f'\nMin errors: {min_t*100:.3f}cm'
    # print(out)
    out += '\nPercentage of test images localized within:'
    threshs_t = [0.01, 0.02, 0.03, 0.05, 0.25, 0.5, 1.0]
    # threshs_R = [1.0, 2.0, 3.0, 5.0, 2.0, 5.0, 10.0]
    for th_t in threshs_t:
        ratio = np.mean((errors_t < th_t))
        out += f'\n\t{th_t*100:.0f}cm : {ratio*100:.2f}%'
    print(out)
    with open(result,'w') as f:
            f.writelines(out)
    f.close()
    # logger.info(out)

def main(gt_path, predict_path, object_name, result_save_path):

    # 定义WGS84坐标系和CGCS_2000_3_Degree_GK_CM_90E投影坐标系
    wgs84 = pyproj.CRS('EPSG:4326')
    cgcs2000_114E = pyproj.CRS('EPSG:4547')
    # cgcs2000_90E = pyproj.CRS('EPSG:4539')

    # 定义转换函数
    transformer = pyproj.Transformer.from_crs(wgs84, cgcs2000_114E, always_xy=True)

    gt_dict = load_gt(gt_path, transformer)
    predict_dict = load_predict(predict_path, object_name)

    eval_absolute_poes(gt_dict,predict_dict,result_save_path)

    # # 定义经纬度和高程
    # lat = 29.56434
    # lon = 91.055593
    # height = 3640

    # # 转换为CGCS_2000_3_Degree_GK_CM_90E坐标系下的投影坐标
    # x, y, z = transformer.transform(lon, lat, height)

    # # 输出结果
    # print("CGCS_2000_3_Degree_GK_CM_90E坐标系下的投影坐标为: ", x, y, z)

    return 0

if __name__=="__main__":
    # 单图的测试名字为:DJI_20230426170641_0001_Z.JPG
    
    # gt.txt格式：
    # image_name lat lon height
    # DJI_20230426170641_0001_Z.JPG 29.56583513 91.0573632 3639.8699

    # predict.txt格式：
    # image_name object_name x y z
    # DJI_20230426170641_0001_Z.JPG pedestrian1 602438.8615997782 3272491.7163627455 3622.789491440625

    # object_name 测试的物体名字
    # result_save_path 保存路迳

    gt_path = "/home/victor/disk1/Disk_E/SensLoc_pipeline/datasets/rtk/output.txt"
    predict_path = "/home/victor/disk1/Disk_E/SensLoc/predictXYZ.txt"
    object_name = "car"
    result_save_path = "./predict_test.txt"

    main(gt_path, predict_path, object_name, result_save_path)

    # 定义WGS84坐标系和CGCS_2000_3_Degree_GK_CM_90E投影坐标系
    # wgs84 = pyproj.CRS('EPSG:4326')
    # cgcs2000_90E = pyproj.CRS('EPSG:4539')

    # # 定义转换函数
    # transformer = pyproj.Transformer.from_crs(wgs84, cgcs2000_90E, always_xy=True)

    #  # 定义经纬度和高程
    # # lat = 29.56484099
    # # lon = 91.05524225
    # # height = 3636.7131

    # lat = 29.564821
    # lon = 91.055260
    # height = 3636.7131

    # # 转换为CGCS_2000_3_Degree_GK_CM_90E坐标系下的投影坐标
    # x, y, z = transformer.transform(lon, lat, height)

    # # 输出结果
    # print("CGCS_2000_3_Degree_GK_CM_90E坐标系下的投影坐标为: ", x, y, z)
