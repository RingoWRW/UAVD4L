import torch
import torchvision.transforms as tvf
import sys
from ..utils.base_model import BaseModel
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '../../third_party'))
from openibl.load_model import vgg16_netvlad
class OpenIBL(BaseModel):
    default_conf = {
        'model_name': 'vgg16_netvlad',
    }
    required_inputs = ['image']
    
    def _init(self, conf):
        self.net = vgg16_netvlad().eval()
        mean = [0.48501960784313836, 0.4579568627450961, 0.4076039215686255]
        std = [0.00392156862745098, 0.00392156862745098, 0.00392156862745098]
        self.norm_rgb = tvf.Normalize(mean=mean, std=std)

    def _forward(self, data):
        image = self.norm_rgb(data['image'])
        desc = self.net(image)
        return {
            'global_descriptor': desc,
        }
        
        