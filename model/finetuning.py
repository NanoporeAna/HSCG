# -*- coding:UTF-8 -*-
# author:Lucifer_Chen
# contact: zhangchen@17888808985.com
# datetime:2023/12/4 15:41

"""
文件说明：  迁移学习,在原先基础上添加一个mlp线性层
"""
from torch import nn

class FineTuner(nn.Module):
    def __init__(self, hscg_model):
        super(FineTuner, self).__init__()
        self.hsqc_encode_ = hscg_model.encode_image
        self.smiles_encode_ = hscg_model.encode_text
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(256, 128)

    def hsqc_encode(self, hsqc_features):
        out = self.hsqc_encode_(hsqc_features)
        image_features = self.linear1(out)
        return image_features

    def smiles_encode(self, smiles_features):
        out = self.smiles_encode_(smiles_features)
        smiles_features = self.linear1(out)
        return smiles_features

    def forward(self, img, smiles):
        hsqc_features = self.hsqc_encode(img)
        smiles_features = self.smiles_encode(smiles)
        return hsqc_features, smiles_features