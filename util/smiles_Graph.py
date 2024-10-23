# -*- coding:UTF-8 -*-
# author:Lucifer_Chen
# contact: zhangchen@17888808985.com
# datetime:2023/12/8 15:42

"""
文件说明：  重新修改smiles到图转化过程
"""
import torch
import numpy as np
from rdkit.Chem import GetAdjacencyMatrix
from torchvision import transforms
import datetime
import logging
import sys
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from torch_geometric.data import Data  # , Dataset, DataLoader


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
        smiles = smiles_to_Graph(smiles)
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
        return cnmr, ids, smile


class train_smiles_cnmr_dataset_new(Dataset):
    def __init__(self, val_data, have_tags):
        self.hsqcs = val_data['ExactHSQC'].values
        self.smiles = val_data['smiles'].values
        self.tags = val_data['tags'].values
        self.have_tags = have_tags
        # self.val_wts = val_data['weight'].values

    def __len__(self):
        return len(self.hsqcs)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        cnmr = self.hsqcs[idx]
        if self.have_tags:
            tagss = self.tags[idx]
        else:
            tagss = None
        smiles = smiles_to_Graph(smiles)
        cnmr = get_hsqc_tensor(cnmr, tags=tagss, have_tags=self.have_tags)
        return smiles, cnmr


class zero_smiles_G(Dataset):
    def __init__(self, smiles):
        self.smiles = smiles

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        smile = self.smiles[index]
        G_smiles = smiles_to_Graph(smile)
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
        smiles = smiles_to_Graph(smiles)
        cnmr = get_hsqc_tensor(cnmr, tags=tagss, have_tags=self.have_tags)
        return smiles, cnmr


class evel_smiles_cnmr_dataset_new(Dataset):
    def __init__(self, val_data, have_tags):
        self.hsqcs = val_data['ExactHSQC'].values
        self.compoundID = val_data['ID'].values
        self.tags = val_data['tags_triple'].values
        self.have_tags = have_tags
        self.val_wts = val_data['molecular_weight'].values

    def __len__(self):
        return len(self.hsqcs)

    def __getitem__(self, idx):
        current_nmr_ = self.hsqcs[idx]
        id = self.compoundID[idx]
        current_wt = self.val_wts[idx]
        current_tags = self.tags[idx]
        if self.have_tags:
            tagss_ = current_tags
        else:
            tagss_ = None
        current_nmr = get_hsqc_tensor(current_nmr_, tags=tagss_, have_tags=self.have_tags)
        return current_nmr, id, current_wt, current_tags


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


def get_hsqc_tensor(hsqc_nmr, tags, have_tags, c_scale=2, h_scale=20, c_min_value=-50, c_max_value=350, h_min_value=-4,
                    h_max_value=16):
    cunits = int((c_max_value - c_min_value) * c_scale)
    hunits = int((h_max_value - h_min_value) * h_scale)
    # 将原始(H,C)经过缩放、转化为C,H 顺序
    # for val in hsqc_nmr:
    #     t = 1
    #     print(hsqc_nmr)

    data = [(round((value[1] - c_min_value) * c_scale), round((value[0] - h_min_value) * h_scale)) for value in
            hsqc_nmr]

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

def one_hot_encoding(x, permitted_list):
    """
    将x映射到permitted_list里，不在就默认最后一位.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding


def get_atom_features(atom, use_chirality=False, hydrogens_implicit=False):
    """
    输入为RDKit 原子对象，输出是一维numpy类型的原子特征组合
    """

    # define list of permitted atoms
    permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I',
                               'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge',
                               'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']

    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms

    # compute atom features
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)  # 原子类型
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])  # 度
    # formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])  # 带电荷

    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])  # 杂化
    is_in_a_ring_enc = [int(atom.IsInRing())]  # 环
    is_aromatic_enc = [int(atom.GetIsAromatic())]  # 芳香环

    # atomic_mass_scaled = [float((atom.GetMass() - 10.812) / 116.092)]  # 质量缩放
    # vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)]  # 半径
    # covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)]

    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + hybridisation_type_enc + \
                          is_in_a_ring_enc + is_aromatic_enc

    # 立体信息
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()),
                                              ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW",
                                               "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    # 如果隐式H，则计算连接H个数
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc
    return np.array(atom_feature_vector)


def get_bond_features(bond,
                      use_stereochemistry=False):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """

    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC,
                                    Chem.rdchem.BondType.UNSPECIFIED
                                    ]

    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)


def smiles_to_Graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # 添加H
    mol = Chem.AddHs(mol)

    # get feature dimensions
    n_nodes = mol.GetNumAtoms()
    n_edges = 2 * mol.GetNumBonds()
    # 无关smiles只为求取原子特征和边特征长度，固定这个大小
    unrelated_smiles = "O=O"
    unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
    n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
    n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))

    # construct node feature matrix X of shape (n_nodes, n_node_features)
    X = np.zeros((n_nodes, n_node_features))
    for atom in mol.GetAtoms():
        X[atom.GetIdx(), :] = get_atom_features(atom)
    X = torch.tensor(X, dtype=torch.float32)

    # construct edge index array E of shape (2, n_edges)
    (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
    torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
    torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
    E = torch.stack([torch_rows, torch_cols], dim=0)

    # construct edge feature array edge_attr of shape (n_edges, n_edge_features)
    edge_attr = np.zeros((n_edges, n_edge_features))
    for (k, (i, j)) in enumerate(zip(rows, cols)):
        edge_attr[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    # construct Pytorch Geometric data object
    datas = Data(x=X, edge_index=E, edge_attr=edge_attr)
    return datas