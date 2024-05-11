# Ultralytics YOLO 🚀, AGPL-3.0 license
from pathlib import Path
import torch
import sys
sys.path.append("/media/guan/3CD61590D6154C10/SomeCodes/sensloc_new")
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
import argparse
from typing import Dict

class DetectionPredictor(BasePredictor):

    def preprocess(self, img):
        """Convert an image to PyTorch tensor and normalize pixel values."""
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results


def predict_position(source_path, cfg=DEFAULT_CFG,use_python=True):
    """Runs YOLO model inference on input image(s)."""
    # model = cfg.model or 'yolov8n.pt'
    model = "/media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/third_party/yolov8/best.pt"
    # source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
    #     else 'https://ultralytics.com/images/bus.jpg'
    source = source_path
    # "/media/guan/3CD61590D6154C10/downloads/ultralytics-main/ultralytics-main/imgs"

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        results = YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
    
    return results
# def afterprocess(detection_dict: Dict):
#     for k, v_list in detection_dict.items():
        
def main(source_path):
    
    """
    center_point_dict:{{img_name:[{'car1':[center_point_x]},{}....]}}
    """

    results = predict_position(source_path)
    center_point_dict = {}
    detection_dict = {}
    for result in results:
        center_point_list = []
        all_label = result.names

        boxes = result.boxes.data
        boxes = boxes.to('cpu')
        boxes = boxes.numpy() 

        label = result.boxes.cls
        label = label.to('cpu')
        label = label.numpy() 
        name = result.path.split('/')[-1]

        write_label = {0.0:0, 1.0:0, 2.0:0, 3.0:0, 4.0:0, 5.0:0, 6.0:0, 7.0:0, 8.0:0, 9.0:0}
        for index in range(len(boxes)):
            object_pos_dict = {}
            center_point = [0.5*(boxes[index][0]+boxes[index][2]), 0.5*(boxes[index][1]+boxes[index][3])]
            temp_label = label[index]
            if temp_label in write_label.keys():
                write_label[temp_label] += 1
            object_name = all_label[temp_label] + str(write_label[temp_label])
            object_pos_dict[object_name] = center_point
            center_point_list.append(object_pos_dict)
        detection_dict[name.split('_Z')[0]] = center_point_list
    
    # center_point_dict = afterprocess(detection_dict)
        
    
    return detection_dict


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default='/home/ubuntu/Documents/code/SensLoc/datasets/wild/images/object_img/W/DJI_20230419175715_0001_W',#!
                    help='Path to the dataset, default: %(default)s')
    args = parser.parse_args()
    
    main(args.dataset)
    

   

    
    
