import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from sensloc.utils.preprocess.utils.ray_casting import compute_3D_point
from collections import defaultdict
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



def qvec2rotmat(qvec):  #!wxyz
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

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def compute_pixel_focal(sensorWidth, sensorHeight, focallength, imgWidth, imgHeight):

    pixelSizeW = sensorWidth/imgWidth
    pixelSizeH = sensorHeight/imgHeight
    fx = focallength / pixelSizeW
    fy = focallength / pixelSizeH
    
    return fx, fy

def euler2quad(euler):
    """
    欧拉转四元数（xyzw）
    """
    ret = R.from_euler('xyz', euler,degrees=True)
    quad = ret.as_quat()

    return quad

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


def interpolate(orientation_path, base_height, intrinsic, min_position, max_position):


    all_euler = []
    all_quad = []

    w, h, sensorW, sensorH, focal_len = intrinsic
    fx, fy = compute_pixel_focal(sensorW, sensorH, focal_len, w, h)
    interval = base_height // 2

    # no xml
    min_x, min_y, max_x, max_y = min_position[0], min_position[1], max_position[0], max_position[1]     #400683,3131100,401563,3131566
    left_up = [min_x, max_y-1] 
    left_donw = [min_x+1, min_y-1]
    right_up = [max_x, max_y]
    right_down = [max_x+1, min_y]
    four_corner = [left_up, left_donw, right_up, right_down]
    
    k_down, b_down = compute_k_b(left_donw, right_down)
    k_up, b_up = compute_k_b(left_up, right_up)

    number_x = round((max_x - min_x) / interval)
    number_y = round((max_y - min_y) / interval)

    x_left_down, y_left_down = generate_end_point(k_down, b_down, [left_donw[0], right_down[0]], number_x)
    x_left_up, y_left_up = generate_end_point(k_up, b_up, [left_up[0], right_up[0]], number_x)

    intrinsic_info = []
    position_info = []
    decision_info = []
    position_list = []
    name_list = []

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
    
    f_r.close()
            # 判断点的生成
    for point_num in range(len(four_corner)):
        for i in range(len(all_euler)):
            R1 = np.asmatrix(qvec2rotmat(all_quad[i]))
            T = np.identity(4)
            T[0:3, 0:3] = R1
            T[0:3, 3] = -R1.dot(np.array([four_corner[point_num][0], four_corner[point_num][1], base_height]))
            decision_info.append([all_quad[i][0], all_quad[i][1], all_quad[i][2], all_quad[i][3], T[0:3, 3][0], T[0:3, 3][1], T[0:3, 3][2]])
    # DB pose generate
    for j in range(len(x_left_down)):
        k_temp, b_temp = compute_k_b([x_left_down[j],y_left_down[j]], [x_left_up[j], y_left_up[j]])
        x_temp_list, y_temp_list = generate_end_point(k_temp, b_temp, [x_left_down[j], x_left_up[j]], number_y)
        for i in range(len(all_euler)):
            for k in range(len(x_temp_list)):
                R1 = np.asmatrix(qvec2rotmat(all_quad[i]))
                T = np.identity(4)
                T[0:3, 0:3] = R1
                T[0:3, 3] = -R1.dot(np.array([x_temp_list[k], y_temp_list[k], base_height]))
                
                pitch, roll, yaw = int(all_euler[i][0]+180), int(all_euler[i][1]), int(all_euler[i][2])
                word_x, word_y, word_z =  int(x_temp_list[k]), int(y_temp_list[k]), base_height
                name = str(word_x) + '@' + str(word_y) + '@' + str(word_z) + '@' + str(pitch) + '@' + str(yaw) + '@' + str(roll)
                name_list.append(name)

                position_list.append([all_quad[i][0], all_quad[i][1], all_quad[i][2], all_quad[i][3], T[0:3, 3][0], T[0:3, 3][1], T[0:3, 3][2]])

                out_line =  name + '.jpg' + ' ' + str(all_quad[i][0]) + ' ' + str(all_quad[i][1]) + ' ' + str(all_quad[i][2]) + ' ' + str(all_quad[i][3]) + ' ' + str(T[0:3, 3][0]) + ' ' + str(T[0:3, 3][1]) + ' ' + str(T[0:3, 3][2]) + '\n'
                
                out_line_in = name + '.jpg' + ' ' + 'PINHOLE' + ' ' + str(w) + ' ' + str(h) + ' ' + str(fx) + ' ' + str(fy) + ' ' + str(w//2) + ' ' + str(h//2) + '\n'
                
                position_info.append(out_line)
                intrinsic_info.append(out_line_in)
    
    return position_info, intrinsic_info, decision_info, position_list, name_list, fx, fy


def generate_obj_dict_and_save(obj_path, obj_dict_path):

    obj_all_tile = sorted(os.listdir(obj_path))
    obj_dict = {}
    save_path = obj_dict_path + '/' + 'obj_dict.npy'

    if os.path.exists(save_path):
        # 如果存在obj_dict则不执行
        print("Skip, obj_dict already exists!!")
        return save_path
    
    for i in range(len(obj_all_tile)):
        one_obj_path = obj_path + obj_all_tile[i] + '/' + obj_all_tile[i] + '.obj'
        # 加载obj
        with open(one_obj_path) as file:
            x_points = []
            y_points = []
            while 1:
                line = file.readline()
                if not line:
                    break
                strs = line.split(" ")
                if strs[0] == "v":
                    x_points.append(float(strs[1]))
                    y_points.append(float(strs[2]))
                if strs[0] == "vt":
                    break

            if len(x_points) != 0 and len(y_points) != 0:
                point_min_x = min(x_points)
                point_min_y = min(y_points)
                point_max_x = max(x_points)
                point_max_y = max(y_points)
        
        obj_dict[obj_all_tile[i]] = [point_min_x, point_min_y, point_max_x, point_max_y]

    
    np.save(save_path, obj_dict)
    print('Generate obj dict success!!!')
    return save_path

def decision_2D(obj_dict, point):

    for i in obj_dict:
        if (obj_dict[i][0] == 'inf') or (obj_dict[i][1] == 'inf') or (obj_dict[i][2] == 'inf') or (obj_dict[i][3] == 'inf'):
            continue
        obj_min_x = obj_dict[i][0] + orgin_coord[0]
        obj_min_y = obj_dict[i][1] + orgin_coord[1]
        obj_max_x = obj_dict[i][2] + orgin_coord[0]
        obj_max_y = obj_dict[i][3] + orgin_coord[1]
        # if (point >= obj_min_x) and (point <= obj_max_x) and (point >= obj_min_y) and (points_3D <= obj_max_y):
        #     obj_exist_list.append(i)
    ##
    pass
def decison_obj(obj_dict, orgin_coord, points_3D, name_list):
    # 判断3D点在哪个obj里面

    obj_exist = {}
    
    for j in range(len(name_list)):
        obj_exist_list = []
        for k in range(4):
            points_3D_x = points_3D[j*4+k][0]
            points_3D_y = points_3D[j*4+k][1]
            for i in obj_dict:
                if (obj_dict[i][0] == 'inf') or (obj_dict[i][1] == 'inf') or (obj_dict[i][2] == 'inf') or (obj_dict[i][3] == 'inf'):
                    continue
                obj_min_x = obj_dict[i][0] + orgin_coord[0]
                obj_min_y = obj_dict[i][1] + orgin_coord[1]
                obj_max_x = obj_dict[i][2] + orgin_coord[0]
                obj_max_y = obj_dict[i][3] + orgin_coord[1]
                if (points_3D_x >= obj_min_x) and (points_3D_x <= obj_max_x) and (points_3D_y >= obj_min_y) and (points_3D_y <= obj_max_y):
                    obj_exist_list.append(i)
        obj_exist[name_list[j]] = list(set(obj_exist_list))

    return obj_exist

# 每个块单独生成 db_pose
# 判断可视域需要加载的块
# 块小于30，继续

def decision_obj(obj_path, obj_dict_path, orgin_coord, oritation_path, intrincis, base_height, block_num, config):
    
    save_dict_path = generate_obj_dict_and_save(obj_path, obj_dict_path)
    obj_dict = np.load(save_dict_path, allow_pickle=True).item()


    all_position_info = []
    all_intrincis_info = []
    obj_info = {}

    point_x_list = []
    point_y_list = []

    for i in obj_dict.keys():
        point_min_x = obj_dict[i][0]
        point_min_y = obj_dict[i][1]
        point_max_x = obj_dict[i][2]
        point_max_y = obj_dict[i][3]
        if (point_min_x == 'inf') or (point_min_y == 'inf') or (point_max_x == 'inf') or (point_max_y == 'inf'):
            continue
        min_point_x, min_point_y = point_min_x+orgin_coord[0], point_min_y+orgin_coord[1]
        max_point_x, max_point_y= point_max_x+orgin_coord[0], point_max_y+orgin_coord[1]
        point_x_list.append(min_point_x)
        point_x_list.append(max_point_x)
        point_y_list.append(min_point_y)
        point_y_list.append(max_point_y)
    
    min_point = [min(point_x_list), min(point_y_list)]
    max_point = [max(point_x_list), max(point_y_list)]

    position_info, intrincis_info, decesion_info, position_list, name_list, fx, fy = interpolate(oritation_path, base_height, intrincis, min_point, max_point)
    K_list = [int(intrincis[0]//2), int(intrincis[1]//2), fx, fy]

    points_3D = compute_3D_point(K_list, position_list, config)
    
    obj_exist = decison_obj(obj_dict, orgin_coord, points_3D, name_list)
    all_position_info += position_info
    all_intrincis_info += intrincis_info
    obj_info = obj_exist
     
    print("Desicion complete!!!")
    return all_position_info, all_intrincis_info, obj_info, obj_dict

def generate_all(min_col, max_col, min_row, max_row, obj_dict):
    # 创建一个空的列表
    all_lst = []
    # 遍历数字的范围
    for i in range(min_col, max_col):
        for j in range(min_row, max_row):
            # 格式化数字为字符串
            i_str = f"{i:03d}"
            j_str = f"{j:03d}"
            # 拼接字符串
            s = f"Tile_+{i_str}_+{j_str}"
            # 将字符串添加到列表中
            if s in obj_dict.keys():
                all_lst.append(s)
    # 返回列表
    return sorted(all_lst)


def complement(obj_info, obj_dict):
    
    info_dict = {}
    for i in obj_info.items():
        j=i[1]
        if len(j) != 0:
            lst = j
        # 获取tile的最小列值'Tile_+000_+003'
            row_list = []
            col_list = []
            for k in range(len(j)):
                row_list.append(int(j[k].split('+')[2]))
                col_list.append(int(j[k].split('+')[1][:-1]))
            min_col = min(col_list)
            max_col = max(col_list)+1
            
            min_row = min(row_list)
            max_row = max(row_list)+1
            all_lst = generate_all(min_col, max_col, min_row, max_row, obj_dict)
            # 将原始列表和生成的列表转换成集合
            if all_lst == []:
                continue
            lst_set = set(lst)
            all_set = set(all_lst)

        # 求出两个集合的差集，得到缺失的元素
            missing_set = all_set - lst_set

        # 将缺失的元素添加到原始列表中
            lst.extend(missing_set)
            #
            info_dict[i[0]] = lst
    return info_dict

def compare_add(dict1, dict2, num_blocks):
    
    d = {}
    # 遍历第一个字典中的键值对
    for k1, v1 in dict1.items():
        # 将值转换为集合
        s1 = set(v1)
        # 遍历第二个字典中的键值对
        for k2, v2 in dict2.items():
            # 将值转换为集合
            s2 = set(v2)
            # 求两个集合的并集
            s = s1 | s2
            if len(s) > num_blocks:
                return False, dict1
            else:
            # 将集合转换为列表，并更新字典
                d.update({k1: list(s), k2: list(s)})
    # print(len(s))
    return True, d

def one_compare(info_dict):

    merged = defaultdict(list)

    for i in info_dict.items():
        merged[tuple(i[1])].append(i[0])
        
    # 创建一个空的列表
    clusters = []

    # 遍历合并后的大字典 #相同的tile会在一起
    for v, ks in merged.items():
        # 创建一个子字典，包含所有对应的键值对
        sub_dict = dict((k, sorted(list(v))) for k in ks)
        # 将子字典添加到列表中
        clusters.append(sub_dict)
    
    return clusters

def main(obj_path, obj_dict_path, orgin_coord, oritation_path, intrincis, base_height, write_path, block_num, config):

    all_point = 0
    obj_name_file = write_path + '/' + 'point_tile' +  '.npy'
    obj_info_file = write_path + '/' + 'obj_info_file' +  '.npy'
    k = 0

    print("Start generate pose!!!!")
    if not (os.path.exists(obj_name_file) and os.path.exists(obj_info_file)):
    
        all_position_info, all_intrincis_info, obj_info, obj_dict = decision_obj(obj_path, obj_dict_path, orgin_coord, oritation_path, intrincis, base_height,  block_num, config)
        pose_file = write_path + '/' + 'pose' + str(k) + '.txt'
        intrin_file = write_path + '/' + 'intrinsic' + str(k) + '.txt'

        with open(pose_file, 'w') as fp:
            with open(intrin_file, 'w') as fi:
                for i in range(len(all_position_info)):
                    name = all_position_info[i].split(' ')[0]
                    qw, qx, qy, qz = all_position_info[i].split(' ')[1],all_position_info[i].split(' ')[2],all_position_info[i].split(' ')[3],all_position_info[i].split(' ')[4]
                    x,y,z = all_position_info[i].split(' ')[5],all_position_info[i].split(' ')[6],all_position_info[i].split(' ')[7][:-1]
                    info = name + ' ' + qw + ' ' + qx + ' ' + qy + ' ' + qz + ' ' + x + ' ' + y + ' ' + z + '\n'
                    w,h = all_intrincis_info[i].split(' ')[2], all_intrincis_info[i].split(' ')[3]
                    fx, fy = all_intrincis_info[i].split(' ')[4], all_intrincis_info[i].split(' ')[5]
                    cx, cy = all_intrincis_info[i].split(' ')[6], all_intrincis_info[i].split(' ')[7][:-1]
                    info_in = name + ' ' + 'PINHOLE' + ' ' + w + ' ' + h + ' ' + fx + ' ' + fy + ' ' + cx + ' ' + cy + '\n'
                    fp.write(info)
                    fi.write(info_in)
        fp.close()
        fi.close()

        info_dict = complement(obj_info, obj_dict)
        np.save(obj_info_file, obj_info)
        np.save(obj_name_file, info_dict) #每个点对应的区块
        print("Finish generate pose!!!")
    else:
        print("Please modify the path, --- generate_render_obj_compare in (403)!!!!")
        obj_dict = np.load('/media/guan/3CD61590D6154C10/SomeCodes/3DV_2024/output/offline/DB_100/obj_dict.npy', allow_pickle=True).item()
        obj_info = np.load(obj_info_file, allow_pickle=True).item()
        info_dict = complement(obj_info, obj_dict)  
        # info_dict = np.load(obj_name_file, allow_pickle=True).item()
       
        # 第一次相同的存在一起
    clusters = one_compare(info_dict)
    # 根据block_num保存
    Flag = True
    d = {}

    i = 0
    Flag, d = compare_add(clusters[i],clusters[i+1], block_num)
    i += 1

    k = 0
    # write_tile_path = 'tile' + str(k) + '.txt'
    # write_name_path = 'point' + str(k) + '.txt'

    write_path = write_path + '/test/'
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    while i < len(clusters):
        # with open('test.txt', 'w') as ft:
        write_name_path = write_path +'point' + str(k) + '.txt'
        while Flag and (i < len(clusters)):
            Flag,d = compare_add(d,clusters[i],block_num)
            # for j in d.keys():
            #     infor = str(sorted(d[j]))
            #     break
                # ft.write(infor)
                # ft.write('\n')
            i += 1
        k1 = list(d.keys())
        v1 = sorted(list(d.values())[0])
        with open(write_name_path, 'w') as fw:
            for tile in v1:
                fw.write(tile+' ')
            fw.write(tile + ' ')
            fw.write('\n')
            for name in k1:
                fw.write(name+'\n')
                all_point += 1
        fw.close()
        Flag = True
        d = clusters[i-1]
        k += 1
    print('all point are', all_point)






if __name__ == "__main__":

    """根据电脑属性自动生成块和DB pose
     obj_path: obj的路径
    obj_dict_path: 想要保存obj_dict的路径
    orgin_coord: 3D 模型的原点坐标
    oritation_path: DB 角度文件
    base_height: 生成DB的绝对高度
    write_path: pose,intrinsic,name的写入路径
    block_num: 电脑最多可加载几个块
    """
    obj_path = "/media/guan/新加卷/north_dataset/Production_1 (2)/Data/"
    obj_dict_path = "/media/guan/新加卷/north_dataset/"
    orgin_coord = np.array([615527,4579640,0])#np.array([401448,3131258,0])
    oritation_path = "/media/guan/data/CityofStars/Render_new_1/bamu_euler.txt"
    intrincis = [4056, 3040,6.29,4.71,4.52] # W
    base_height = 1850
    write_path = "/media/guan/新加卷/north_dataset/" 
    block_num = 62
    config = {
    "DSM_path" : "/media/guan/新加卷/north_dataset/Production_1 (2)_DSM_merge.tif",
    "DSM_npy_path" : "/media/guan/新加卷/north_dataset/my_array.npy",
    "num_sample":100
}   
    
    main(obj_path, obj_dict_path, orgin_coord, oritation_path, intrincis, base_height, write_path, block_num, config)
