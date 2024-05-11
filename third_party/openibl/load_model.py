import sys
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from ibl import models
dependencies = ['torch']


def vgg16_netvlad(pretrained=True):
    base_model = models.create('vgg16', pretrained=False)
    pool_layer = models.create('netvlad', dim=base_model.feature_dim)
    model = models.create('embednetpca', base_model, pool_layer)
    model_path = Path(__file__).parent / 'vgg16_netvlad.pth'
    print(model_path)
    if pretrained:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    return model
