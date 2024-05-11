import os
from pathlib import Path

def blender_engine(
    blender_path,
    project_path,
    script_path,
    origin,
    sensor_height,
    sensor_width,
    f_mm,
    intrinscs_path,
    extrinsics_path,
    image_save_path,
):
    '''
    blender_path: .exe path, start up blender
    project_path: .blend path,
    script_path: .py path, batch rendering script
    intrinscs_path: colmap format
    extrinsics_path: colmap format
    image_save_path: rendering image save path
    '''
    # cmd = '{} -b {} -P {} -- {} {} {}'.format(
    #     blender_path,
    #     project_path,
    #     script_path,
    #     intrinscs_path,
    #     extrinsics_path,
    #     image_save_path,  
    # )
    cmd = '{} -b {} -P {} -- {} {} {} {} {} {} {}'.format(
        blender_path,
        project_path,
        script_path,
        origin,
        sensor_height,
        sensor_width,
        f_mm,
        intrinscs_path,
        extrinsics_path,
        image_save_path,  
    )
    os.system(cmd)


def main(config, 
         intrinscs_path: Path,
         extinsics_path: Path,
         img_save_path: Path
         ):
    python_rgb_path = config["python_rgb_path"]
    python_depth_path = config["python_depth_path"]
    rgb_path = config["rgb_path"]
    depth_path = config["depth_path"]
    blender_path = config["blender_path"]
    f_mm = config["f_mm"]
    sensor_width = config["sensor_width"]
    sensor_height = config["sensor_height"]
    origin = config["origin"]
    
    depth_save_path = config["depth_images"]

    print("render....")

    # render rgb and depth images.
    blender_engine(
        blender_path,
        rgb_path,
        python_rgb_path,
        origin,
        sensor_height,
        sensor_width,
        f_mm,
        str(intrinscs_path),
        str(extinsics_path),
        str(img_save_path),

    )
    blender_engine(
        blender_path,
        depth_path,
        python_depth_path,
        origin,
        sensor_height,
        sensor_width,
        f_mm,
        str(intrinscs_path),
        str(extinsics_path),
        str(depth_save_path),
    )

if __name__ == "__main__":
    blender_path = ""
    sensor_height = 4.71
    sensor_width = 6.29
    f_mm = 4.5
    origin = ""