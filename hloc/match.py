import os
import glob

def main(image_path, write_path):
    # match w and z
    # return match.txt
    img_list = sorted(glob.glob(image_path + '/*.JPG'))
    img_W_list = []
    img_Z_list = []
    match_path = write_path + '/' + 'match.txt'
    for i in img_list:
        if i.split('/')[-1][-5] == "W":
            img_W_list.append(i)
        elif i.split('/')[-1][-5] == "Z":
            img_Z_list.append(i)
    j = 0
    k = 0
    with open(match_path, 'w') as fm:
        
        while k <= len(img_W_list)-1:
            num_z = img_Z_list[j].split('/')[-1][-10:-6]
            num_w = img_W_list[k].split('/')[-1][-10:-6]
            if num_z == num_w:
                info = img_W_list[k].split('/')[-1] + ' ' + img_Z_list[j].split('/')[-1] + '\n'
                fm.write(info)
                j += 1
                k += 1
            else:
                j += 1
    print("write match info finished!!")


main("/media/guan/data/20230512feicuiwan/H20t/seq_all_new", "/media/guan/data/20230512feicuiwan/H20t")


