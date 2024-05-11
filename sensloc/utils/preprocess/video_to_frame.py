import cv2
import numpy as np
import os
import numpy as np
import math
import argparse

def video2frame(input_video_path, ouput_image_path):
    """
    video convert to image
    video type is .MP4
    """

    video_root_file = []

    for i in os.listdir(input_video_path):
        full_path = os.path.join(input_video_path, i)
        if full_path.endswith('.MP4'):  # Modifiable suffix
            video_root_file.append(full_path)
    
    if os.path.isdir(ouput_image_path):
        print('output path already exists!')
    else:
        os.makedirs(ouput_image_path) 
    
    video_num = len(video_root_file)
    
    if video_num != 0:
        State = True
    else:
        print('The video path is incorrect, or the number of videos is 0.')
    
    for i in range(video_num):
        # Each video is traversed for frame extraction and saving
        cap = cv2.VideoCapture(video_root_file[i])
        fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
        print("fps:",fps)
        print("h,w:",height, width)
        cap.set(cv2.CAP_PROP_FPS, 0)  

        video_name = video_root_file[i].split('/')[-1][:-4]
        video_name_file = os.path.join(ouput_image_path, video_name)
        video_file_name = os.path.join(ouput_image_path, video_name.split('_')[-1])

        if os.path.isdir(video_file_name):
            pass
        else:
            os.makedirs(video_file_name)
        
        imageNum = 0
        while State:
            capState, frame = cap.read()
            if capState == True and (imageNum % fps) == 0:
                # save image
                cv2.imwrite(video_file_name + "/" + video_name + str(int(imageNum // fps)) + '.jpg', frame)
            if capState == False:
                cap.release()
                break
            imageNum += 1
            
        print('-----------',video_name,'-----------')
    
    print("----------------video to frame end---------------------------")

def main_bak():
    parser = argparse.ArgumentParser(description="Convert video to frame, 1 frame in 1 second.")
    parser.add_argument("--input_video_path", default="/home/ubuntu/Documents/code/SensLoc/datasets/jinxia/Queries/Raw/video/seq1")
    parser.add_argument("--output_image_path", default="/home/ubuntu/Documents/code/SensLoc/datasets/jinxia/Queries/process/video/seq1/images")
    args = parser.parse_args()

    video2frame(input_video_path=args.input_video_path, ouput_image_path=args.output_image_path)
def main(config):
    # parser = argparse.ArgumentParser(description="Convert video to frame, 1 frame in 1 second.")
    # parser.add_argument("--input_video_path", default="/home/ubuntu/Documents/code/SensLoc/datasets/jinxia/Queries/Raw/video/seq1")
    # parser.add_argument("--output_image_path", default="/home/ubuntu/Documents/code/SensLoc/datasets/jinxia/Queries/process/video/seq1/images")
    # args = parser.parse_args()
    input_video_path = config["input_video_path"]
    output_image_path = config["output_image_path"]

    video2frame(input_video_path=input_video_path, ouput_image_path=output_image_path)

if __name__ == "__main__":
    main_bak()
