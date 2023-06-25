
# from src.multihead import MultiHeadAttention
# from src.vonly import PreNet
# from src.transform import TransformerEncoder
from src.models import MULTModel
import yaml
import torch
import torch.nn as nn

# multihead with vision only
# class CreateModel(nn.Module):
#     def __init__(self, yaml_dir):
#         super(CreateModel, self).__init__()
#         self.yaml_dir = yaml_dir
#         self.yaml = yaml.load(open(self.yaml_dir), Loader=yaml.FullLoader)
#         self.keyword = self.yaml['keyword']
#         self.config = self.yaml['config']
#         self.model = eval(self.keyword)(self.config)
#
#     def forward(self, vis, aud, tract):
#         out = self.model(vis, vis, vis)
#         return out


# transformer with vision only
# class CreateModel(nn.Module):
#     def __init__(self, yaml_dir):
#         super(CreateModel, self).__init__()
#         self.yaml_dir = yaml_dir
#         self.yaml = yaml.load(open(self.yaml_dir), Loader=yaml.FullLoader)
#         self.keyword = self.yaml['keyword']
#         self.config = self.yaml['config']
#         self.model = eval(self.keyword)(self.config)
#
#     def forward(self, vis, aud, tract):
#         out = self.model(vis, vis, vis)
#         return out

class CreateModel(nn.Module):
    def __init__(self, yaml_dir):
        super(CreateModel, self).__init__()
        self.yaml_dir = yaml_dir
        self.yaml = yaml.load(open(self.yaml_dir), Loader=yaml.FullLoader)
        self.keyword = self.yaml['keyword']
        self.config = self.yaml['config']
        self.model = eval(self.keyword)(self.config)

    def forward(self, vis, aud, tract):
        out = self.model(vis, aud, tract)
        return out




