from torch import nn
import torch
import numpy as np
import os 
from .model_nmr import SqueezeNet
from .gcn_model import GCN
from .gin_model import GINet

from .gnn_model import GNN_graphpred


import torch.nn.functional as F

class HSCG(nn.Module):
    def __init__(self,
                 config_json,
                 **kwargs
                 ):
        super(HSCG, self).__init__(**kwargs)
        self.config = config_json
        self.visual = SqueezeNet(version=config_json["model_config"]['squeezenet_version'], num_classes=config_json["model_config"]['output_channels'])
        
        self.GNN = GNN_graphpred(
                                num_layer=config_json["GCN_model"]['num_layer'],
                                emb_dim=config_json["GCN_model"]['emb_dim'],
                                JK=config_json["GCN_model"]['JK'],
                                drop_ratio=config_json["GCN_model"]['drop_ratio'],
                                graph_pooling=config_json["GCN_model"]['pool'],
                                gnn_type=config_json["GCN_model"]['gnn_type'],
                                feat_chang_dim=config_json["GCN_model"]['feat_chang_dim'],
                                feat_dim=config_json["GCN_model"]['feat_dim'],
                                smiles_version=config_json["GCN_model"]['version'])
        # 根据需求来选择是否网络自己学习
        #_scale = config_json["trainer_config"]['logits_scale']
        #self.scale = nn.Parameter(torch.ones([])* np.log(1/0.07))# * _scale)
        #self.scale = nn.Parameter(torch.ones([]) * _scale)

    def encode_HSQC(self, HSQC_Spectrum):
        out = self.visual(HSQC_Spectrum)
        HSQC_features = self.norm_feature(out)
        return HSQC_features

    def encode_structure(self, structure_map):
        out = self.GNN(structure_map)
        structure_features = self.norm_feature(out)
        return structure_features
    
    def norm_feature(self, feature):
        return feature / feature.norm(dim=-1, keepdim=True)

    def forward(self, image, data):
        HSQC_features = self.encode_HSQC(image)
        structure_features = self.encode_structure(data)
        # print('HSQC shape:', image_features.shape)
        # print('structure shape:：',  text_features.shape)
        return HSQC_features, structure_features  # , self.scale
