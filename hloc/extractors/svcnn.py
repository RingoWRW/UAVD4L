import sys
from ..utils.base_model import BaseModel
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '../../third_party'))
from svcnn.svcnn_lite import MobileNetV2_V1

class SVCNN(BaseModel):
    default_conf = {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }
    required_inputs = ['image']
    detection_noise = 2.0

    def _init(self, conf):
        self.net = MobileNetV2_V1().cuda()
    def _forward(self, data):
        img = data['image']
        return self.net(img)        
