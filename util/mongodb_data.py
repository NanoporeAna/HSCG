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
from rdkit.Chem.rdchem import BondType as BT
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
        return cnmr, ids, smile
    
class train_smiles_cnmr_dataset_new(Dataset):
    def __init__(self, val_data, have_tags, hsqc_mode='mode1', smiles_version='version1'):
        self.hsqcs = val_data['ExactHSQC'].values
        self.smiles = val_data['smiles'].values
        self.tags = val_data['tags'].values
        self.have_tags =have_tags
        self.hsqc_mode = hsqc_mode
        self.smiles_version = smiles_version
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
        if self.smiles_version == 'version1':
            smiles = dis_smiles_to_G(smiles)
        elif self.smiles_version == 'version2':
            smiles = smiles_to_Graph(smiles)
        if self.hsqc_mode == 'mode1':
            cnmr = get_hsqc_tensor(cnmr, tags=tagss, have_tags=self.have_tags, mode='mode1')
        elif self.hsqc_mode == 'mode2':
            cnmr = get_hsqc_tensor(cnmr, tags=tagss, have_tags=self.have_tags, mode='mode2')
        return smiles, cnmr


class zero_smiles_G(Dataset):
    def __init__(self, smiles, smiles_version='version1'):
        self.smiles = smiles
        self.smiles_version = smiles_version

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        smile = self.smiles[index]
        if self.smiles_version == 'version1':
            G_smiles = dis_smiles_to_G(smile)
        elif self.smiles_version == 'version2':
            G_smiles = smiles_to_Graph(smile)
        else:
            print('The mode of smiles encode  is Unknown!')
        return G_smiles

class zero_HSQC(Dataset):
    def __init__(self, HSQC, HSQC_version='mode1'):
        self.HSQC = HSQC
        self.hsqc_mode = HSQC_version

    def __len__(self):
        return len(self.HSQC)

    def __getitem__(self, index):
        current_nmr_ = self.HSQC[index]
        current_nmr = get_hsqc_tensor(current_nmr_, tags=False, have_tags=False, mode=self.hsqc_mode)
        return current_nmr


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
    def __init__(self, val_data, have_tags, hsqc_mode='mode1'):
        self.hsqcs = val_data['ExactHSQC'].values
        self.compoundID = val_data['ID'].values
        self.tags = val_data['tags_triple'].values
        self.have_tags = have_tags
        self.val_wts = val_data['molecular_weight'].values
        self.nonActivateH = val_data['non_active_h_count'].values
        self.hsqc_mode = hsqc_mode

    def __len__(self):
        return len(self.hsqcs)

    def __getitem__(self, idx):
        current_nmr_ = self.hsqcs[idx]
        id = self.compoundID[idx]
        current_wt = self.val_wts[idx]
        current_tags = self.tags[idx]
        non_active_h = self.nonActivateH[idx]
        # if self.have_tags:
        #     tagss_ = current_tags
        # else:
        #     tagss_ = None
        current_nmr = get_hsqc_tensor(current_nmr_, c_scale=2, h_scale=20, tags=current_tags, have_tags=self.have_tags, mode=self.hsqc_mode)
        return current_nmr, id, current_wt, current_tags, int(non_active_h)


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
                    h_max_value=16, mode='mode1'):
    cunits = int((c_max_value - c_min_value) * c_scale)
    hunits = int((h_max_value - h_min_value) * h_scale)
    # 初始化单张图片的格式[800,400]shape的0矩阵
    hsqc_item = np.zeros((cunits, hunits))
    if mode == 'mode1':
        data = [(round((value[1] - c_min_value) * c_scale), round((value[0] - h_min_value) * h_scale)) for value in
                hsqc_nmr]
        if have_tags:
            for i, ch in enumerate(data):
                hsqc_item[ch[0]][hunits - ch[1]] = tags[i]
        else:
            for ch in data:
                hsqc_item[ch[0]][hunits - ch[1]] = 1
    elif mode == 'mode2':
        unique_data = np.unique(hsqc_nmr, axis=0)
        data = [(round((value[1] - c_min_value) * c_scale), round((value[0] - h_min_value) * h_scale)) for value in
                unique_data]
        for ch in data:
            hsqc_item[ch[0]][hunits - ch[1]] += 1

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
    # 对于数据量的统计
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
    X = torch.tensor(X, dtype=torch.long)

    # construct edge index array E of shape (2, n_edges)
    (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
    torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
    torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
    E = torch.stack([torch_rows, torch_cols], dim=0)

    # construct edge feature array edge_attr of shape (n_edges, n_edge_features)
    edge_attr = np.zeros((n_edges, n_edge_features))
    for (k, (i, j)) in enumerate(zip(rows, cols)):
        edge_attr[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    # construct Pytorch Geometric data object
    datas = Data(x=X, edge_index=E, edge_attr=edge_attr)
    return datas