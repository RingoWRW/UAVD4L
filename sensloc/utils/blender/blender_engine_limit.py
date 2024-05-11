import os
from pathlib import Path
import json
def blender_engine(
    blender_path,
    project_path,
    script_path,
    project_test,
    sensor_height,
    sensor_width,
    f_mm,
    intrinscs_path,
    extrinsics_path,
    image_save_path,
    input_Objs,
    depth_save_path
):
    '''
    blender_path: .exe path, start up blender
    project_path: .blend path,
    script_path: .py path, batch rendering script
    intrinscs_path: colmap format
    extrinsics_path: colmap format
    image_save_path: rendering image save path
    '''
    cmd = '{} -b {} -P {} -- {} {} {} {} {} {} {} {} {}'.format(
        blender_path,
        project_path,
        script_path,
        depth_save_path, 
        input_Objs ,
        project_test,
        sensor_height,
        sensor_width,
        f_mm,
        intrinscs_path,
        extrinsics_path,
        image_save_path, 
        
    )
    os.system(cmd)
def import_Objs(
    blender_path,
    project_path,
    script_path,
    input_Objs,
    origin,

):
    '''
    blender_path: .exe path, start up blender
    project_path: .blend path,
    script_path: .py path, batch rendering script
    intrinscs_path: colmap format
    extrinsics_path: colmap format
    image_save_path: rendering image save path
    '''
    cmd = '{} -b {} -P {} -- {} {}'.format(
        blender_path,
        project_path,
        script_path,
        input_Objs,
        origin
    )
    os.system(cmd)

def main(config, 
         ):
    # python_rgb_path = config["python_rgb_path"]
    # python_obj_path = config["importObjs_path"]
    project_path = config["project_path"]
    python_path = config["python_path"]
    blender_path = config["blender_path"]
    f_mm = config["f_mm"]
    sensor_width = config["sensor_width"]
    sensor_height = config["sensor_height"]
    # project_test = config["project_test"]
    input_Objs = config["input_Objs"]
    intrinscs_path = config["intrinscs_path"]
    extinsics_path = config["extinsics_path"]
    img_save_path = config["img_save_path"]
    depth_save_path = config["depth_images"]   

    print("render....")

    # render rgb and depth images.
    project_test_list = os.listdir(config["project_test"])
    for j in sorted(project_test_list):
        project_test = config["project_test"] + '/' + j
        print(project_test)
        blender_engine(
            blender_path,
            project_path,
            python_path,
            project_test,
            sensor_height,
            sensor_width,
            f_mm,
            str(intrinscs_path),
            str(extinsics_path),
            str(img_save_path),
            input_Objs,
            depth_save_path
        )

if __name__ == "__main__":

    config = {
    "blender_path" : "/home/guan/下载/blender-3.0.0-linux-x64/blender",
    "sensor_height" : 4.71,
    "sensor_width" : 6.29,
    "f_mm" : 4.5,
    "project_path":"/media/guan/data/CityofStars/DB_150/rgb.blend",
    "project_test" : "/media/guan/data/CityofStars/shcnew/query_gt3/test1", #test 工程路径
    "input_Objs":"/media/Shen/Data/RingoData/Production_1_obj/Data",
    "python_path":"/media/guan/3CD61590D6154C10/SomeCodes/3DV_2024/sensloc/utils/blender/blender.py",
    "depth_images":"/media/guan/data/CityofStars/shcnew/query_gt3/depths",
    "img_save_path":"/media/guan/data/CityofStars/shcnew/query_gt3/image_prior",
    "intrinscs_path":"/media/guan/data/CityofStars/shcnew/query_gt3/intrinsic0.txt",
    "extinsics_path":"/media/guan/data/CityofStars/shcnew/query_gt3/pose3.txt",
    }
    
    main(config)
    