import torch
import os
import numpy as np
from torchvision import transforms
import pandas as pd
import datetime
import logging
import sys
from transformers import RobertaTokenizer
import yaml
from torch.utils.data import Dataset
import json
from collections import Counter 
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from torch_geometric.data import Data#, Dataset, DataLoader
#device = 'cuda:7' if torch.cuda.is_available() else
ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC,
    BT.UNSPECIFIED # 临时添加对这个键的处理，放在最后一列，避免影响其他位置
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


# config_path = "./Lucifer-nick/coconut_smiles/config.yaml"

# with open(os.path.join(config_path), 'r') as file:
#     # 版本问题需要将Loader=yaml.CLoader加上
#     config_yaml = yaml.load(file.read(), Loader=yaml.CLoader)
# smiles_encode = RobertaTokenizer.from_pretrained(config_yaml["trainer_config"]['roberta_tokenizer_path'],
#                                                      max_len=config_yaml["dataset_config"]['smiles_maxlen'])

class smiles_cnmr_dataset(Dataset):
    def __init__(self, df, have_tags):
        self.smiles = df["smiles"]
        self.hsqcNMR = df["hsqcNMR"]
        if have_tags:
            self.tags = df["tags"]
        else:
            self.tags = []
        self.have_tags = have_tags

    def __len__(self):
        return len(self.smiles)
    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        cnmr = self.hsqcNMR[idx]
        if self.have_tags:
            tags = self.tags[idx]
        else:
            tags = None 
        smiles = dis_smiles_to_G(smiles)
        cnmr = get_hsqc_tensor(cnmr, tags, self.have_tags)

        return smiles, cnmr

class eval_smiles_cnmr_dataset(Dataset):
    ### 6月7日添加smiles检索
    def __init__(self, df, have_tags):
        self.id = df["id"]
        self.smiles = df["smiles"]
        self.hsqcNMR = df["hsqcNMR"]
        if have_tags:
            self.tags = df["tags"]
        else:
            self.tags = []
        self.have_tags = have_tags

    def __len__(self):
        return len(self.hsqcNMR)
    def __getitem__(self, idx):
        ids = self.id[idx]
        smile = self.smiles[idx]
        cnmr = self.hsqcNMR[idx]
        if self.have_tags:
            tags = self.tags[idx]
        else:
            tags = None
        cnmr = get_hsqc_tensor(cnmr, tags=tags, have_tags=self.have_tags)
        return  cnmr, ids, smile

class zero_smiles_G(Dataset):
    def __init__(self, smiles):
        self.smiles = smiles
    def __len__(self):
        return len(self.smiles)
    def __getitem__(self, index):
        smile = self.smiles[index]
        G_smiles = dis_smiles_to_G(smile)
        return G_smiles


class smiles_cnmr_dataset_new(Dataset):
    def __init__(self, hsqcs, smiless, tags, have_tags):
        self.hsqcs = hsqcs
        self.smiless = smiless
        self.tags = tags
        self.have_tags = have_tags

    def __len__(self):
        return len(self.smiless)
    def __getitem__(self, idx):
        smiles = self.smiless[idx]
        cnmr = self.hsqcs[idx]
        if self.have_tags:
            tagss = self.tags[idx]
        else:
            tagss = None
        smiles = dis_smiles_to_G(smiles)
        cnmr = get_hsqc_tensor(cnmr, tags=tagss, have_tags=self.have_tags)
        return smiles, cnmr

class evel_smiles_cnmr_dataset_new(Dataset):
    def __init__(self, val_hsqc, val_data, val_tags, have_tags, val_wts):
        self.hsqcs = val_hsqc
        self.val_data = val_data
        self.tags = val_tags
        self.have_tags =have_tags
        self.val_wts = val_wts
    def __len__(self):
        return len(self.hsqcs)
    def __getitem__(self, idx):
        cnmr = self.hsqcs[idx]
        val_data = self.val_data[idx]
        wt = self.val_wts[idx]
        tagss = self.tags[idx]
        t = Counter(tagss)
        mm = [t[1], t[2], t[3]]
        if self.have_tags:
            tagss_ = tagss
        else:
            tagss_ = None
        cnmr = get_hsqc_tensor(cnmr, tags=tagss_, have_tags=self.have_tags)
        return  cnmr, val_data, wt, mm

class Load_Data:
    def __init__(self, config):
        # print(config)
        # print(config["dataset_config"])
        self.smiles_type = config["dataset_config"]['smiles_type']
        self.hsqc_type = config["dataset_config"]['hsqc_type']
        self.tags_type = config["dataset_config"]['tags_type']
        self.ID_type = config["dataset_config"]['ID_type']
        self.JeolID_type = config["dataset_config"]['JeolID_type']
        self.wt_type = config["dataset_config"]['wt_type']
        self.have_tags = config["dataset_config"]['have_tags']
        self.data_enhancement = config["dataset_config"]['data_enhancement']
        self.data_enhancement_type = config["dataset_config"]['data_enhancement_type']
        self.data_reat = config["dataset_config"]['data_reat']
 
    #旧数据集加载
    def load_data_old(self, files, data_type):
        '''
        input：
            files: 需要加载的文件列表，files可以是多个文件组合，其中每个文件必须包含的字段为smiles数据、hsqc数据、ID值、tags值
            data_type：需要返回的是训练数据还是测试数据
        output：
            当data_type为train时，只返回一个包含smiles、hsqc或还有tags的字典用于训练
            当data_type为eval时，返回字典的同时，返回全部的smiles集合，ID所有集合，wts所有集合
        '''
        flag = 1
        all_smiles = []
        all_nmr = []
        all_tags = []
        data = None
        all_ids = []
        all_wts = []
        if isinstance(files, list):
            for file in files:
                print(file)
                if flag == 1:
                    data = pd.read_json(file)
                    smiles = data[self.smiles_type].values
                    cnmr = data[self.hsqc_type].values
                    tag = data[self.tags_type].values
                    if 'jeol' in data_type:
                        id = data[self.JeolID_type].astype(str).values
                    else:
                        id = data[self.ID_type].astype(str).values
                    wt = data[self.wt_type].values
                    all_smiles = smiles
                    all_nmr = cnmr
                    all_tags = tag
                    all_ids = id
                    all_wts = wt
                    flag += 1
                else:
                    data = pd.read_json(file)
                    smiles = data[self.smiles_type].values
                    cnmr = data[self.hsqc_type].values
                    tag = data[self.tags_type].values
                    if 'jeol' in data_type:
                        id = data[self.JeolID_type].astype(str).values
                    else:
                        id = data[self.ID_type].astype(str).values
               
                    wt = data[self.wt_type].values
                    all_smiles = np.concatenate((all_smiles, smiles), axis=0)
                    all_nmr = np.concatenate((all_nmr, cnmr), axis=0)
                    all_tags = np.concatenate((all_tags, tag), axis=0)
                    all_ids = np.concatenate((all_ids, id), axis=0)
                    all_wts = np.concatenate((all_wts, wt), axis=0)
        else:
            data = pd.read_json(files)
            all_smiles = data[self.smiles_type].values
            all_nmr = data[self.hsqc_type].values
            all_tags = data[self.tags_type].values
            if 'jeol' in data_type:
                all_ids = data[self.JeolID_type].astype(str).values
            else:
                all_ids = data[self.ID_type].astype(str).values
            
            all_wts = data[self.wt_type].values
        df = {'smiles': [], 'hsqcNMR': [], 'tags': []}
        index = int(len(all_smiles) * self.data_reat)
        if data_type == 'train':
            df['smiles'] = all_smiles[:index]
            df['hsqcNMR'] = all_nmr[:index]
            if self.have_tags:
                df['tags'] = all_tags[:index]
            return df
        if data_type == 'eval':
            df['smiles'] = all_smiles[index::]
            df['hsqcNMR'] = all_nmr[index::]
            df['id'] = all_ids[index::]
            if self.have_tags:
                df['tags'] = all_tags[index::]
            return df, all_smiles, all_ids, all_wts
        if data_type == 'jeol':
            df['smiles'] = all_smiles
            df['hsqcNMR'] = all_nmr
            df['id'] = all_ids
            if self.have_tags:
                df['tags'] = all_tags
            return df, all_smiles, all_ids, all_wts
        if data_type == 'jeol_test':
            # 这个类型为我后期测试Jeol数据添加
            test ={}
            df['smiles'] = list(all_smiles[0:index])
            test['smiles'] = list(all_smiles[index::])
            df['hsqcNMR'] = list(all_nmr[0:index])
            test['hsqcNMR'] = list(all_nmr[index::])
            df['id'] = list(all_ids[0:index])
            test['id'] = list(all_ids[index::])
            if self.have_tags:
                df['tags'] = all_tags[0:index].values
                test['tags'] = all_tags[index::].values
            ttt = data[data['split_flag']=='ODD']
            all_smiles = np.concatenate((all_smiles, ttt['smiles_2D']), axis=0)         
            all_ids = np.concatenate((all_ids, ttt['np_ID']), axis=0)
            all_wts = np.concatenate((all_wts, ttt['weight']), axis=0)
            return df, test, all_ids, all_smiles, all_wts

    def load_data_new(self, files, data_type):
        '''
        input：
            files: 需要加载的文件列表，files可以是多个文件组合，其中每个文件必须包含的字段为smiles数据、hsqc数据、ID值、tags值
            data_type：需要返回的是训练数据还是测试数据
        output：
            当data_type为train时，只返回训练所需的smiles、hsqc或还有tags的单个集合
            当data_type为eval时，datas：所有数据, all_ids：所有ID集合, all_wts所有质量约束集合, val_hsqc：训练hsqc数据, val_id：训练ID数据, val_tags：训练多重性数据
        '''
        if isinstance(files, list):
            datas = []
            for file in files:
                f = open(file, 'r', encoding='utf-8')
                data = json.load(f)
                datas += data
        else:
            f = open(files, 'r', encoding='utf-8')
            datas = json.load(f)
        all_hsqcs = []
        all_smiless = []
        all_tags = []
        all_ids = []
        all_wts = []
        for data in datas:
            hsqc = data[self.hsqc_type]
            smiles = data[self.smiles_type]
            tag = data[self.tags_type]
            id = data[self.ID_type]
            wt = data[self.wt_type]
            if self.data_enhancement:
                smiles_list = data[self.data_enhancement_type]
                for smile in smiles_list:
                    all_hsqcs.append(hsqc)
                    all_smiless.append(smile)
                    all_tags.append(tag)
                    all_ids.append(id)
                    all_wts.append(wt)
            else:
                all_hsqcs.append(hsqc)
                all_smiless.append(smiles)
                all_tags.append(tag)
                all_ids.append(id)
                all_wts.append(wt)
        if data_type == 'train':
            if len(all_hsqcs) != len(all_smiless): 
                print("The dataset have error !")
                sys.exit(1)
            else:
                train_hsqc = all_hsqcs[:int(len(all_hsqcs)*self.data_reat)]
                train_smiles = all_smiless[:int(len(all_hsqcs)*self.data_reat)]
                if self.have_tags:
                    train_tags = all_tags[:int(len(all_hsqcs)*self.data_reat)]
                else:
                    train_tags = []
            return train_hsqc, train_smiles, train_tags
        if data_type == 'eval':
            if len(all_hsqcs) != len(all_smiless): 
                print("The dataset have error !")
                sys.exit(1)
            else:
                val_hsqc = all_hsqcs[int(len(all_hsqcs)*self.data_reat):]
                val_id = all_ids[int(len(all_hsqcs)*self.data_reat):]
                if self.have_tags:
                    val_tags = all_tags[int(len(all_hsqcs)*self.data_reat):]
                else:
                    val_tags = []
                if len(val_hsqc) != len(val_id):
                    print("data have error !")
                    sys.exit(1)
            return datas,all_smiless, all_ids, all_wts, val_hsqc, val_id, val_tags



def init_logger(filename, logger_name):
    # get current timestamp
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(name)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(filename=filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Test
    logger = logging.getLogger(logger_name)
    logger.info('### Init. Logger {} ###'.format(logger_name))
    return logger


# def smiles_str_encode(smiles_str):#, smiles_encode, config_json):
#     if isinstance(smiles_str, str):
#         encode_dict = smiles_encode(text=smiles_str, max_length=config_yaml["dataset_config"]["smiles_maxlen"], padding='max_length', truncation=True)
#         smiles_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).view(1, -1)
#         smiles_mask = torch.from_numpy(np.array(encode_dict['attention_mask'])).view(1, -1)

#         return smiles_ids, smiles_mask


def get_hsqc_tensor(hsqc_nmr, tags, have_tags, c_scale=2, h_scale=20, c_min_value=-50, c_max_value=350, h_min_value=-4,
                    h_max_value=16):
    cunits = int((c_max_value - c_min_value) * c_scale)
    hunits = int((h_max_value - h_min_value) * h_scale)
    # 将原始(H,C)经过缩放、转化为C,H 顺序
    # for val in hsqc_nmr:
    #     t = 1
#     print(hsqc_nmr)
    
    data = [(round((value[1] - c_min_value) * c_scale), round((value[0] - h_min_value) * h_scale)) for value in hsqc_nmr]
        
    # 初始化单张图片的格式[800,400]shape的0矩阵
    hsqc_item = np.zeros((cunits, hunits))
    if have_tags:
        for i, ch in enumerate(data):
            hsqc_item[ch[0]][hunits - ch[1]] = tags[i]
    else:
        for ch in data:
            hsqc_item[ch[0]][hunits - ch[1]] = 1
    hsqc_item = hsqc_item.reshape((cunits, hunits, 1))
    # ToTensor()将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor
    hsqc = transforms.ToTensor()(hsqc_item)
    # 其中mean和std分别通过(0.5,0.5,0.5)和(0.5,0.5,0.5)进行指定。原来的0-1最小值0则变成(0-0.5)/0.5=-1，而最大值1则变成(1-0.5)/0.5=1.
    # 前面的（0.5，0.5，0.5） 是 R G B 三个通道上的均值， 后面(0.5, 0.5, 0.5)是三个通道的标准差，由于我们只有一个通道的数据，故只送一个值
    hsqc = transforms.Normalize(mean=0.5, std=0.5)(hsqc)
    # 类型与网络一致，转化为torch.FloatTensor
    hsqc = hsqc.type(torch.FloatTensor)
    return hsqc


def nmr2tensor(nmr, scale=10, min_value=-50, max_value=350):
    units = (max_value - min_value) * scale
    item = np.zeros(units)
    nmr = [round((value - min_value) * scale) for value in nmr]
    for index in nmr:
        if index < 0:
            item[0] = 1
        elif index >= units:
            item[-1] = 1
        else:
            item[index] = 1
    item = torch.from_numpy(item).to(torch.float32)
    return item


def dis_smiles_to_G(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # 添加H
    mol = Chem.AddHs(mol)
    type_idx = []
    chirality_idx = []
    atomic_number = []
    # aromatic = []
    # sp, sp2, sp3, sp3d = [], [], [], []
    # num_hs = []
    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        atomic_number.append(atom.GetAtomicNum())
        # aromatic.append(1 if atom.GetIsAromatic() else 0)
        # hybridization = atom.GetHybridization()
        # sp.append(1 if hybridization == HybridizationType.SP else 0)
        # sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        # sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
        # sp3d.append(1 if hybridization == HybridizationType.SP3D else 0)

    # z = torch.tensor(atomic_number, dtype=torch.long)
    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
    x = torch.cat([x1, x2], dim=-1)
    # x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, sp3d, num_hs],
    #                     dtype=torch.float).t().contiguous()
    # x = torch.cat([x1.to(torch.float), x2], dim=-1)

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        # edge_type += 2 * [MOL_BONDS[bond.GetBondType()]]
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
    datas = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return datas