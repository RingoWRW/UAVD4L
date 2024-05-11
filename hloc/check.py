import glob

def main(image_path, read_path, write_path):

    pose_path = read_path + 'db_pose.txt'
    intrinsic_path = read_path + 'db_intrinsics.txt'
    img_list = sorted(glob.glob(image_path + '/*.jpg'))
    pose_dict = {}
    intrinsic_dict = {}
    with open(pose_path, 'r') as f_p:
        for line in f_p.read().rstrip().split('\n'):
            name = line.split(' ')[0].split('/')[-1]
            temp_list = line.split(' ')[1:8]
            pose_dict[name] = temp_list
    
    with open(intrinsic_path, 'r') as f_i:
        for line in f_i.read().rstrip().split('\n'):
            name = line.split(' ')[0].split('/')[-1]
            temp_list = line.split(' ')[1:8]
            intrinsic_dict[name] = temp_list
    
    f_p.close()
    f_i.close()

    w_pose_path = write_path + 'new_db_pose.txt'
    w_intrinsic_path = write_path + 'new_db_intrinsic.txt'

    with open(w_pose_path, 'w') as f_pn:
        with open(w_intrinsic_path, 'w') as f_in:
            for i in img_list:
                name = i.split('/')[-1]
                if name in pose_dict.keys():
                    pose_temp = name + ' ' + pose_dict[name][0] + ' ' + pose_dict[name][1] + ' ' +  pose_dict[name][2] + ' ' + pose_dict[name][3] + ' ' + pose_dict[name][4] + ' ' \
                                + pose_dict[name][5] + ' ' + pose_dict[name][6] + '\n'
                    intrinsic_temp = name + ' ' + intrinsic_dict[name][0] + ' ' + intrinsic_dict[name][1] + ' ' +  intrinsic_dict[name][2] + ' ' + intrinsic_dict[name][3] + ' ' + intrinsic_dict[name][4] + ' ' \
                                + intrinsic_dict[name][5] + ' ' + intrinsic_dict[name][6] + '\n'
                    f_pn.write(pose_temp)
                    f_in.write(intrinsic_temp)
    
    f_pn.close()
    f_in.close()

    print("Check image and txt file has finished!!!")

if __name__ == "__main__":

    main('/media/guan/data/CityofStars/Render_all_images/images', '/media/guan/data/CityofStars/Render_all_images/', '/media/guan/data/CityofStars/Render_all_images/')