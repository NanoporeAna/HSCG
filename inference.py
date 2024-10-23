from datetime import datetime
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch.utils.tensorboard import SummaryWriter
#from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from model.model_build import HSCG
#from util.data import smiles_str_encode, load_data, smiles_cnmr_dataset, init_logger, eval_smiles_cnmr_dataset
import yaml, os, sys
from util.mongodb_data import init_logger, evel_smiles_cnmr_dataset_new, zero_smiles_G
from torch_geometric.data import DataLoader
from collections import Counter
import pymongo

def get_weights_from_smiles(smiles):
    res = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        mw = Descriptors.MolWt(mol)
        res.append(mw)
    return res

def get_tags_from_smiles(smiles):
    # print(smiles)
    res = []
    for ss in tqdm(smiles):
        try:
            mol = Chem.MolFromSmiles(ss)
            mol = Chem.AddHs(mol)
            tags = []
            for i, atom in enumerate(mol.GetAtoms()):
                if atom.GetSymbol() == 'C':
                    # print("have C")
                    neighbors = atom.GetNeighbors()
                    have_h = 0
                    for neighbor in neighbors:
                        if neighbor.GetAtomicNum() == 1:  # 氢的原子序数为1
                            # print("有氢")
                            have_h += 1
                    if have_h == 1:
                        tags.append(1)
                    elif have_h == 2:
                        tags.append(2)
                    elif have_h == 3:
                        tags.append(3)
            t = Counter(tags)
            mm = [t[1], t[2], t[3]]
            res.append(mm)
        except:
            res.append([0, 0, 0])

    return res

def zero_shot_classifier(model, allsmiles, device):
    """
    将smiles转化为 feature向量
    :param model:主函数加载的模型
    :param allsmiles: smiles表达式
    :param device: GPU or CPU
    :return: 参考库 allsmiles对应的特征向量库
    """
    with torch.no_grad():
        zeroshot_weights = []
        for texts in tqdm(allsmiles):
            G_smiles = texts.to(device)

            class_embeddings = model.encode_structure(G_smiles)

            zeroshot_weights.append(class_embeddings)
        zeroshot_weights = torch.cat(zeroshot_weights, dim=0).to(device)
        #print(zeroshot_weights.shape)
    return zeroshot_weights

def get_topK_result(nmr_feature, ID, smiles_feature, topK, device):
    """

    @param nmr_features: 输入的谱图特征（batchsize, out_dim）
    @param smiles_features: 参考库或选定数据库经过smiles编码器获得的特征 (n, out_dim)
    @param is_wt: 是否添加质量约束
    @param topK: 需要取前K个结果
    @param device: 指定GPU or CPU
    @return: 输入谱图推理出来的得分，正确下标
    """
    
    with torch.no_grad():
        nmr_smiles_distances_tmp = (
                            nmr_feature.unsqueeze(0) @ smiles_feature.t()).to(device)
                    # print(nmr_smiles_distances_tmp.shape)
        score, indice = nmr_smiles_distances_tmp.topk(topK, dim=1, largest=True, sorted=True)
        #根据返回的下标，找到其对应的smile数据表中的ID
        res_ID = []
        for i in range(0,topK):
            res_ID.append(ID[indice[0][i]])
    return res_ID, score[0]


def get_topK_result_new(nmr_feature, ID, smiles_feature, topK, device):
    """
    @param nmr_features: 输入的谱图特征（out_dim）
    ID：输入的smiles_feature对应的ID
    @param smiles_features: 参考库或选定数据库经过smiles编码器获得的特征 (n, out_dim)
    @param is_wt: 是否添加质量约束
    @param topK: 需要取前K个结果
    @param device: 指定GPU or CPU
    @return:
    ID：列表，记录符合输入多重性数据的ID项数据
    score：输入谱图推理出来的得分，按顺序对应每个ID
    """
    #判断ID和smiles_feature的长度是否一致
    if len(ID) != len(smiles_feature):
        print("The dataset have error !")
        sys.exit(1)
    with torch.no_grad():
        smiles_feature = torch.tensor(smiles_feature).to(device)

        # if topK<=smiles_feature.shape[0]:
        #     topK = smiles_feature.shape[0] -1
        try:
            #求出输入频谱特征编码与各个smile特征编码的相似性
            nmr_smiles_distances_tmp = (
                    nmr_feature.unsqueeze(0) @ smiles_feature.t()).to(device)
            # print(nmr_smiles_distances_tmp.shape)
            #找出前topk个，将前下表与相似得分记录，并排序
            scores, indice = nmr_smiles_distances_tmp.topk(topK, dim=1, largest=True, sorted=True)
            #根据返回的下标，找到其对应的smile数据表中的ID
            res_ID = []
            scores = scores.detach().cpu()
            scores = scores.tolist()
            # indice = indice.detach().cpu()
            # indice = indice.tolist()
            for i in range(0,topK):
                res_ID.append(ID[indice[0][i]])
        except:
            res_ID = ID[:10]
            scores = [[None]]
            print('error')
    #返回结果
    return res_ID, scores[0]

def change_data(datas, data_enhancement):
    """

    @param datas:
    @param data_enhancement:
    @return:
    """
    data_smiles = {} #key:smiles, value:id
    id_smiles_lists = {} # key:id, value:all smiles lists
    for data in datas:
        id = data["ID"]
        data_smiles[data["smiles"]] = id
        if data_enhancement:
            all_smiles_list = []
            all_smiles_list.append(data["smiles"])
            for smiles in data["smiles_list"]:
                all_smiles_list.append(smiles)
                data_smiles[smiles] = id
            id_smiles_lists[id] = all_smiles_list
        else:
            id_smiles_lists[id] = data["smiles"]
    return data_smiles, id_smiles_lists


def find_fit_wt(f_ms, temp_cur, da):
    '''
    切换MongoDB临时参考库查找质量约束的数据  # 临时参考库质量排序，然后
    输入：
    f_ms：需要查询的质量约束组（多组）
    temp_cur: MongoDB临时参考库
    da：质量范围控制系数
    输出：
    IDs：列表，记录符合输入多重性数据的ID项数据（多组）
    smiles_features：列表，记录符合输入质量约束数据的smiles特征编码（多组）
    '''
    IDs, smiles_features = [], []

    for f_m in f_ms:
        new_smiles_feature = []
        new_ID = []
        min_m = f_m - (f_m * da * 1e-5 + 1)
        max_m = f_m + (f_m * da * 1e-5 + 1)
        # 执行查询操作（需要核对数据库存储对象）
        rets = temp_cur.find({'molecular_weight': {'$gt': float(min_m), '$lt': float(max_m)}})

        # 记录查询结果
        for index, ret in enumerate(rets):
            # features_index = int(data[7])
            new_ID.append((ret['ID']))
            new_smiles_feature.append(ret['smiles_feature'])

        IDs.append(new_ID)
        smiles_features.append(new_smiles_feature)

    return IDs, smiles_features


def find_fit_mm(mms, temp_cur):
    '''
    切换MongoDB临时参考库查找多重性约束的数据，这里tags直接是数组（或元组）比较
    输入：
    mms：需要查询的多重性约束规则，其中源代码中筛选的是前三个质量约束匹配的数据组（多组）
    temp_cur: MongoDB临时参考库
    输出：
    IDs：列表，记录符合输入多重性数据的ID项数据（多组）
    smiles_features：列表，记录符合输入多重性数据的smiles特征编码（多组）
    '''
    IDs, smiles_features = [], []
    for i in range(len(mms[0])):
        # tt = [row[i].item() for row in mms]
        tt = [mms[0][i].item(), mms[1][i].item(), mms[2][i].item()]
        new_ID = []

        new_smiles_feature = []
        # 执行查询操作
        # rets = temp_cur.find({'tags.0': float(tt[0]), 'tags.1': float(tt[1]), 'tags.2': float(tt[2])})
        rets = temp_cur.find({'tags_triple': tt})
        # 记录查询结果
        for ret in rets:
            new_ID.append((ret['ID']))
            new_smiles_feature.append(ret['smiles_feature'])
        IDs.append(new_ID)
        smiles_features.append(new_smiles_feature)
    return IDs, smiles_features

def find_fit_wt_mm(f_ms, da, mms, temp_cur):
    IDs, smiles_features = [], []
    for i in range(len(mms[0])):
        # tt = [row[i].item() for row in mms]
        tt = [mms[0][i].item(), mms[1][i].item(), mms[2][i].item()]
        min_m = f_ms[i] - (f_ms[i] * da * 1e-5 + 1)
        max_m = f_ms[i] + (f_ms[i] * da * 1e-5 + 1)
        new_ID = []
        # 执行聚合查询
        pipeline = [
            {
                "$match": {
                    "$and": [
                        {"molecular_weight": {'$gt': float(min_m), '$lt': float(max_m)}},
                        {"tags_triple": {"$eq": tt}}
                    ]
                }
            }
        ]
        new_smiles_feature = []
        # 执行查询操作
        rets = temp_cur.aggregate(pipeline)
        # 记录查询结果
        for ret in rets:
            new_ID.append((ret['ID']))
            new_smiles_feature.append(ret['smiles_feature'])
        IDs.append(new_ID)
        smiles_features.append(new_smiles_feature)
    return IDs, smiles_features


def find_fit_non_activeH(non_actives, temp_cur):
    '''
        切换MongoDB临时参考库查找多重性约束的数据，这里tags直接是数组（或元组）比较
        输入：
        non_active：需要查询的非活泼氢约束规则，
        temp_cur: MongoDB临时参考库
        输出：
        IDs：列表，记录符合输入多重性数据的ID项数据（多组）
        smiles_features：列表，记录符合输入多重性数据的smiles特征编码（多组）
        '''
    IDs, smiles_features = [], []
    for non_activateh in non_actives:
        new_smiles_feature = []
        new_ID = []
        non_activateh = non_activateh.item()
        rets = temp_cur.find({'non_active_h_count': non_activateh})
        # 记录查询结果
        for index, ret in enumerate(rets):
            new_ID.append((ret['ID']))
            new_smiles_feature.append(ret['smiles_feature'])
        IDs.append(new_ID)
        smiles_features.append(new_smiles_feature)
    return IDs, smiles_features

def detection(val_dataloader, model, is_wt, is_m, is_no_activateH,  temp_cur, result_cur, da, device, all_smiles_features_tensor, IDs, dataset_ID, split_flag):
    """_summary_

    Args:
        val_dataloader (_type_): 验证数据集data迭代器
        model (nn): 编码器
        is_wt (bool): 质量约束
        is_m (bool): 多重性约束开关
        temp_cur (mongoDB): 临时表连接
        result_cur (mongoDB): 结果表连接
        da (float): 质量约束范围
        device (str): gpu设备
        all_smiles_features_tensor (Tensor): 参考库所有smiles特征
        IDs (list): 参考库所有ID
        source (str): 数据来源：jeol、COCONUT
        date_type (str): 数据类型：训练集、测试集、未知

    Returns:
        _type_: top1, top5, top10
    """
    top1_num = 0
    top5_num = 0
    top10_num = 0
    results = []
    results_new = []
    num = 0
    len_val_data = 0
    for data in tqdm(val_dataloader):
        #取数据
        images, ref_id, wt, mm, non_active_h_count = data
        # non_active_h_count = non_active_h_count.item()
        with torch.no_grad():
            images = images.to(device)
            nmr_feature = model.encode_HSQC(images)
        
        #进行质量约束
        # is_m = True
        # is_wt = True
        # 只添加非活泼H约束
        if is_no_activateH:
            IDs, new_smiles_features = find_fit_non_activeH(non_actives=non_active_h_count, temp_cur=temp_cur)
        # 只添加质量约束都不加
        if is_wt and (not is_m) and (not is_no_activateH):
            # m_screen_datas, new_smiles_features = find_fit_wt(wt, cur, da, all_smiles_features)
            IDs, new_smiles_features = find_fit_wt(f_ms=wt, temp_cur=temp_cur, da=da)
        # 只添加多重性约束
        elif (not is_wt) and is_m and (not is_no_activateH):
            # m_screen_datas, new_smiles_features = find_fit_mm(mm, cur, all_smiles_features)
            IDs, new_smiles_features = find_fit_mm(mms=mm, temp_cur=temp_cur)
        # 都加
        elif is_wt and is_m and (not is_no_activateH):
            IDs, new_smiles_features = find_fit_wt_mm(f_ms=wt, da=da, mms=mm, temp_cur=temp_cur)


            # IDs_wt, smiles_features_wt = find_fit_wt(f_ms=wt, temp_cur=temp_cur, da=da)
            # IDs_mm, smiles_features_mm = find_fit_mm(mms=mm, temp_cur=temp_cur)
            # #取交集
            # # IDs = [ID for ID in IDs_wt if ID in IDs_mm]
            # # smiles_features = [smile for smile in smiles_features_wt if smile in smiles_features_mm]
            # IDs = []
            # new_smiles_features = []
            # for i in range(len(IDs_wt)):
            #     for j in range(len(IDs_mm)):
            #         if IDs_wt[i] == IDs_mm[j]:
            #             IDs.append(IDs_wt[i])
            #             new_smiles_features.append(smiles_features_wt[i])
            #             break
            # if new_smiles_features == []:
            #     # 这里假设两个交集没有（理论上有），实验存在误差，质量测量查过误差范围
            #     IDs = IDs_wt
            #     new_smiles_features = smiles_features_wt

        #进行对比和统计
        for i in range(len(wt)):
            if (not is_wt) and (not is_m) and (not is_no_activateH):
                smiles_feature = all_smiles_features_tensor.clone().detach()
                ID_ = IDs
            else:
                smiles_feature = new_smiles_features[i]
                ID_ = IDs[i]
            # if smiles_feature == [[]]:
            #     print('没有匹配的项', data)
                # continue
            if len(smiles_feature) == 0:
                print('没有匹配的项', data)
                # continue
            result = {}
            num1 = 0
            num5 = 0
            num10 = 0
            result["ID"] = None
            result["true_smiles"] = None
            result["wt"] = None
            result["mm"] = None
            result["index"] = None
            result["forecast_data"] = None
            now_id = ref_id[i]
            result["ID"] = now_id
            result["wt"] = float(wt[i])
            result["mm"] = [float(row[i]) for row in mm]

            #返回前k个与之匹配的数据    
            if len(smiles_feature) >= 10:
                res_ID, scores = get_topK_result_new(nmr_feature=nmr_feature[i], ID=ID_, smiles_feature=smiles_feature, topK=500, device=device)
            else:
                res_ID, scores = get_topK_result_new(nmr_feature=nmr_feature[i], ID=ID_, smiles_feature=smiles_feature, topK=len(smiles_feature), device=device)
            if scores==[None]:
                print(ref_id[i])
            forecast_datas = []
            result_topK = []
            positive_position = 666
            for j, pre_mol_id in enumerate(res_ID):              
                find_smile_by_ID_rets = list(temp_cur.find({'ID': pre_mol_id}))
                if len(find_smile_by_ID_rets) == 0:
                    print(pre_mol_id)
                    pre_mol_smiles = 'fake'
                else:
                    pre_mol_smiles = find_smile_by_ID_rets[0]['smiles']
                
                if now_id == pre_mol_id:
                    positive_position = j+1
                if j < 1 and now_id == pre_mol_id:
                    num1 += 1
                if j < 5 and now_id == pre_mol_id:
                    num5 += 1
                if j < 10 and now_id == pre_mol_id:
                    num10 += 1
                
                # 保存预测的数据
                forecast_data = {}
                forecast_data["id"] = pre_mol_id
                forecast_data["smiles"] = pre_mol_smiles
                result_topK.append(pre_mol_smiles)
                forecast_data["score"] = float(scores[j])
                forecast_datas.append(forecast_data)
                
            result["forecast_data"] = forecast_datas
            if result["index"] == None:
                result["index"] = 11
            if num1 > 0:
                top1_num += 1
            if num5 > 0:
                top5_num += 1
            if num10 > 0:
                top10_num += 1
            results.append(result)

            # 结果算好了，进行一个写入数据库操作
            write_in_model_result_db_data = {
                'dataset_ID': dataset_ID,
                'splitFlag': split_flag,
                'algorithm_version': 'v2.0.0',
                "compound_ID": now_id,
                'rank': positive_position,
                'scores': scores,
                'wt': is_wt,
                'multiple': is_m,
                'non_activateH': is_no_activateH,
                'predictIDs': res_ID,
                "createBy": "张晨",
                "createAt": "2024-04-19",
                "updateAt": "2024-04-19",
                "updateBy": "张晨",
                "similarity": [1]
            }
            # result_cur.insert_one(write_in_model_result_db_data)
            len_val_data += 1

    top1 = top1_num / len_val_data
    top5 = top5_num / len_val_data
    top10 = top10_num / len_val_data
    return top1, top5, top10


def get_reference_data(coco_data, jeol_data, basic_compound_data, chemicalClass, select='Nature'):
    """_summary_

    Args:
        coco_data (_type_): _description_
        jeol_data (_type_): _description_
        all_smiles_data (_type_): _description_
        chemicalClass (_type_): _description_

    Returns:
        _type_: _description_
    """
    ref = None
    if select == 'Chinese Traditional Medicine':
        jeol_data = jeol_data[jeol_data['split_flag'] == 'C']
        coco_ref = coco_data[coco_data['chemicalClass'].isin(chemicalClass)]
        # jeol_ref = jeol_data[jeol_data['chemicalClass'].isin(chemicalClass)]
        # jeol_ref = jeol_ref.loc[:, ['smiles', 'tags_triple', 'weight', 'Ref_ID', 'non_active_h_count']]
        # jeol_ref.columns = ['smiles', 'tags_triple', 'molecular_weight', 'Ref_ID', 'non_active_h_count']
        ref = coco_ref.loc[:, ['smiles', 'tags_triple', 'molecular_weight', 'Ref_ID', 'non_active_h_count']]
        # ref = pd.concat([coco_ref.loc[:, ['smiles', 'tags_triple', 'molecular_weight', 'Ref_ID', 'non_active_h_count']], jeol_ref])
        ref.columns = ['smiles', 'tags_triple', 'molecular_weight', 'ID', 'non_active_h_count']
    elif select == 'Nature':
        ref = basic_compound_data.loc[:, ['Smiles', 'tags_triple', 'Molecular Weight', 'non_active_h_count', 'ID']]
        ref.columns = ['smiles', 'tags_triple', 'molecular_weight', 'non_active_h_count', 'ID']
    else:
        print('Unknown Select')
    return ref

def get_test_data(ref_tabel, select='JeolA', chemicalClass=''):
    """_summary_

    Args:
        coco_data (_type_): _description_
        jeol_data (_type_): _description_
        chemicalClass (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = None
    if 'Jeol' in select:
        objs = ref_tabel.find({'split_flag': select.strip('Jeol')})
        df = pd.DataFrame(list(objs))
        # tags 这里不是三元组，暂时占位
        df = df.loc[:, ['smiles', 'tags_triple', 'ExactMass', 'Ref_ID', 'ExactHSQC', 'np_ID', 'non_active_h_count']]
    elif 'Coconut' in select:
        # print(select.strip('Coconut_'))
        objs = ref_tabel.find({'split_flag': select.split('_')[-1]})
        df = pd.DataFrame(list(objs))
        df = df.loc[:, ['smiles', 'tags_triple', 'molecular_weight', 'Ref_ID', 'ExactHSQC', 'coconut_id', 'non_active_h_count']]
    elif select == 'Chinese Traditional Medicine':
        ref = pd.DataFrame(ref_tabel.find({"split_flag": {"$in": ["A", "B"]}}))
        chinese_test = ref[ref['chemicalClass'].isin(chemicalClass)]
        df = chinese_test.loc[:, ['smiles', 'tags_triple', 'ExactMass', 'Ref_ID', 'ExactHSQC', 'np_ID', 'non_active_h_count']]
    df.columns = ['smiles', 'tags_triple', 'molecular_weight', 'ID', 'ExactHSQC', 'source_ID', 'non_active_h_count']
    # chinese_test = jeol_data[jeol_data['chemicalClass'].isin(subclass)]
    # coco_data = coco_data[coco_data['split_flag']=='val']
    # jeol_data = jeol_data.loc[:, ['smiles_2D', 'tags_triple', 'molecular_weight','Ref_ID', 'ExactHSQC', 'np_ID']]
    # coconut_data = coco_data.loc[:, ['smiles', 'tags_triple', 'molecular_weight','Ref_ID', 'ExactHSQC', 'coconut_id']]
    # jeol_data.columns = ['smiles', 'tags_triple', 'molecular_weight','Ref_ID', 'ExactHSQC', 'source_ID']
    # coconut_data.columns = ['smiles', 'tags_triple', 'molecular_weight','Ref_ID', 'ExactHSQC', 'source_ID']
    # nature_test = {'coconut': coconut_data, 'jeol': jeol_data}
    # return chinese_test, nature_test
    return df

if __name__ == "__main__":
    import time
    start_time = time.time()
    #加载配置文件
    config_path = "./configs/config.yaml"
    with open(os.path.join(config_path), 'r') as file:
        config_yaml = yaml.load(file.read(), Loader=yaml.CLoader)
    device = config_yaml["val_config"]['device'] if torch.cuda.is_available() else 'cpu'
    batch_size = config_yaml["val_config"]['batch_size']
    filename = config_yaml["val_config"]['pre_model']
    have_tags = config_yaml["dataset_config"]["have_tags"]
    data_enhancement = config_yaml["dataset_config"]["data_enhancement"]
    load_data_type = config_yaml["dataset_config"]["load_data_type"]
    result_save_name = config_yaml["val_config"]["result_save_name"]
    is_wt = config_yaml["val_config"]["is_wt"]
    is_m = config_yaml["val_config"]["is_m"]
    epochs = config_yaml["trainer_config"]['epoches']
    learning_rate = config_yaml["trainer_config"]['lr']
    _scale = config_yaml["trainer_config"]['logits_scale']
    output_channels = config_yaml['model_config']['output_channels']
    data_reat = config_yaml["val_config"]['data_reat']
    save_log_name = config_yaml["val_config"]['save_logs_name']
    da = config_yaml["val_config"]["da"]

    writer = SummaryWriter('./' + os.path.join(save_log_name))

    # 加载临时表数据
    mongoclient = pymongo.MongoClient('mongodb://172.10.10.8:27017', username='root', password='123456')
    db = mongoclient['HSQC_Clean']
    basic_compound = db.Base_Compound_Clean
    coco_table = db.Coconut_Compound_Clean
    jeol_table = db.Jeol_Compound_Clean_Split
    chem_table = db.ChemOnt
    temp_cur = db.TempDB_Clean
    result_cur = db.pre_Result_with_Condition
    coco_data = pd.DataFrame(list(coco_table.find({})))
    jeol_data = pd.DataFrame(list(jeol_table.find({})))
    # 下面是选择参考数据库
    # basic_compound_data = pd.DataFrame(list(basic_compound.find({})))
    basic_compound_data = pd.DataFrame(list(basic_compound.find({'Source': {'$in': ["coconut"]}})))
    chemicalClass = pd.DataFrame(list(chem_table.find({})))['name_subclass'].values.tolist()

    # 根据条件获取参考库数据
    ref = get_reference_data(coco_data=coco_data, jeol_data=jeol_data, basic_compound_data=basic_compound_data,
                             chemicalClass=chemicalClass, select='Nature')
    JeolA = get_test_data(ref_tabel=jeol_table, chemicalClass=chemicalClass, select='JeolA')
    # chinese_med = get_test_data(ref_tabel=jeol_table, chemicalClass=chemicalClass, select='Chinese Traditional Medicine')
    JeolB = get_test_data(ref_tabel=jeol_table, chemicalClass=chemicalClass, select='JeolB')
    # JeolC = get_test_data(ref_tabel=jeol_table, select='JeolC')
    # CoconutB = get_test_data(ref_tabel=coco_table, select='Coconut_val')
    # dataset_chinese_med = evel_smiles_cnmr_dataset_new(chinese_med, have_tags, hsqc_mode='mode1')
    # chinese_dataloader = DataLoader(dataset_chinese_med, batch_size=batch_size, num_workers=2)
    datasetA = evel_smiles_cnmr_dataset_new(JeolA, have_tags, hsqc_mode='mode1')
    JeolA_dataloader = DataLoader(datasetA, batch_size=batch_size, num_workers=2)
    datasetB = evel_smiles_cnmr_dataset_new(JeolB, have_tags, hsqc_mode='mode1')
    JeolB_dataloader = DataLoader(datasetB, batch_size=batch_size, num_workers=2)
    # datasetC = evel_smiles_cnmr_dataset_new(JeolC, have_tags, hsqc_mode='mode1')
    # JeolC_dataloader = DataLoader(datasetC, batch_size=batch_size, num_workers=2)
    # datasetD = evel_smiles_cnmr_dataset_new(CoconutB, have_tags, hsqc_mode='mode1')
    # CoconutB_dataloader = DataLoader(datasetD, batch_size=batch_size, num_workers=2)
    # print("加载测试数据Coconut B 长度：", CoconutB.shape)
    # print("加载JEOL数据中测中药相关数据 大小：", chinese_med.shape)
    print("加载测试数据JeolA 长度：", JeolA.shape)
    print("加载测试数据JeolB 长度：", JeolB.shape)
    # print("加载测试数据JeolC 长度：", JeolC.shape)

    # 构建参考库数据
    data_pd = ref.copy()
    all_ID = ref['ID'].values
    data_pd['model_ID'] = [4 for _ in range(len(data_pd))]
    data_pd['create_time'] = [str(datetime.now()) for _ in range(len(data_pd))]
    print("加载数据完成,数据库长度：", len(data_pd))
    load_data_time = time.time()
    # 加载分子库
    print("开始加载smile分子库")
    ss = data_pd['smiles'].values
    smiles_dataset = zero_smiles_G(ss, smiles_version='version1')
    smiles_dataloader = DataLoader(dataset=smiles_dataset, batch_size=batch_size, num_workers=2)

    for epoch in range(600, epochs + 1, 10):
        torch.cuda.empty_cache()
        # 加载模型
        model = HSCG(config_yaml)
        model_path = config_yaml["val_config"]["pre_model"]
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model'])
        model.eval()
        model.to(device)

        # 遍历打印模型参数详细信息
        # for name, param in model.named_parameters():
        #     print(f"Parameter Name: {name}")
        #     print(f"Shape: {param.shape}")
        #     print(f"Data Type: {param.dtype}")
        #     # print(f"Values (first 5 elements): {param.data}")  # 可选，打印前5个元素作为示例
        #     print("---------")
        load_model_time = time.time()

        print("加载模型使用的时间：", load_model_time-start_time)
        all_smiles_features = zero_shot_classifier(model=model, allsmiles=smiles_dataloader, device=device)
        print("The smiles library shape: ", all_smiles_features.shape)
        load_smiles_feater_time = time.time()
        print("编译smile分子库使用的时间：", load_smiles_feater_time - load_data_time)
        all_smiles_features_ = all_smiles_features.detach().cpu().numpy()
        all_smiles_features_ = all_smiles_features_.tolist()

        print('开始写入temp数据库...')
        data_pd['smiles_feature'] = all_smiles_features_
        data_pd['HSQC_feature'] = [None for _ in range(len(data_pd))]
        temp_cur.delete_many({})
        # result_cur.delete_many({})
        temp_cur.insert_many(data_pd.to_dict(orient='records'))
    
        # 创建索引
        temp_cur.create_index('ID')
        temp_cur.create_index('molecular_weight')
        temp_cur.create_index('tags_triple')
        temp_cur.create_index('non_active_h_count')
        # temp_cur.ensure_index([('molecular_weight', 1), ('tags_triple', 1)])
        store_data_time = time.time()
        print("创建数据库，存入数据使用的时间：", store_data_time - load_smiles_feater_time)
        print("开始检测")
        # init_logger("result_with_mm_wt.log", f'特别说明：修改质量约束，这次采用rdkit计算的精准质量作为约束条件，而不是上面实际实验测量的质量')
        # —————————————————————————————————
        # 处理Jeol C数据
        # JeolCtop1, JeolCtop5, JeolCtop10 = detection(JeolC_dataloader, model, False, False, False, temp_cur, result_cur,
        #                                              da, device, all_smiles_features, all_ID, 'xx05', 'JeolC')
        # print("未加约束 JeolC:", JeolCtop1, JeolCtop5, JeolCtop10)

        # init_logger("result_with_mm_wt.log", f'未加约束 JeolB:{JeolBtop1}, {JeolBtop5}, {JeolBtop10}')
        # 处理JeolB数据
        JeolBtop1, JeolBtop5, JeolBtop10 = detection(JeolB_dataloader, model, False, False, False, temp_cur, result_cur,
                                                     da, device, all_smiles_features, all_ID, 'xx04', 'JeolB')
        print("未加约束 JeolB:", JeolBtop1, JeolBtop5, JeolBtop10)
        # init_logger("result_with_mm_wt.log", f'未加约束 JeolB:{JeolBtop1}, {JeolBtop5}, {JeolBtop10}')
        # JeolBtop1, JeolBtop5, JeolBtop10 = detection(JeolB_dataloader, model, is_wt, is_m, True, temp_cur, result_cur,
        #                                              da, device, all_smiles_features, all_ID, 'xx04', 'JeolB')
        # print("非活泼H JeolB:", JeolBtop1, JeolBtop5, JeolBtop10)
        # init_logger("result_with_mm_wt.log", f'非活泼H JeolB:{JeolBtop1}, {JeolBtop5}, {JeolBtop10}')
        #
        # JeolBtop1, JeolBtop5, JeolBtop10 = detection(JeolB_dataloader, model, True, is_m, False, temp_cur, result_cur,
        #                                              da, device, all_smiles_features, all_ID, 'xx04', 'JeolB')
        # print("质量约束 JeolB:", JeolBtop1, JeolBtop5, JeolBtop10)
        # init_logger("result_with_mm_wt.log", f'质量约束 JeolB:{JeolBtop1}, {JeolBtop5}, {JeolBtop10}')
        #
        # JeolBtop1, JeolBtop5, JeolBtop10 = detection(JeolB_dataloader, model, False, True, False, temp_cur, result_cur,
        #                                              da, device, all_smiles_features, all_ID, 'xx04', 'JeolB')
        # print("多重性约束 JeolB:", JeolBtop1, JeolBtop5, JeolBtop10)
        # init_logger("result_with_mm_wt.log", f'多重性约束 JeolB:{JeolBtop1}, {JeolBtop5}, {JeolBtop10}')
        # JeolBtop1, JeolBtop5, JeolBtop10 = detection(JeolB_dataloader, model, True, True, False, temp_cur, result_cur,
        #                                              da, device, all_smiles_features, all_ID, 'xx04', 'JeolB')
        # print("质量+多重性约束 JeolB:", JeolBtop1, JeolBtop5, JeolBtop10)
        # init_logger("result_with_mm_wt.log", f'质量+多重性约束 JeolB:{JeolBtop1}, {JeolBtop5}, {JeolBtop10}')
        # —————————————————————————————————
        # JeolAtop1, JeolAtop5, JeolAtop10 = detection(chinese_dataloader, model, False, False, False, temp_cur, result_cur,
        #                                              da, device, all_smiles_features, all_ID, 'xx03', 'Jeol')
        # print("未加约束 中药测试数据结果:", JeolAtop1, JeolAtop5, JeolAtop10)

        JeolAtop1, JeolAtop5, JeolAtop10 = detection(JeolA_dataloader, model, False, False, False, temp_cur, result_cur,
                                                     da, device, all_smiles_features, all_ID, 'xx03', 'JeolA')
        print("未加约束 JeolA:", JeolAtop1, JeolAtop5, JeolAtop10)
        # init_logger("result_with_mm_wt.log", f'未加约束 JeolA:{JeolAtop1}, {JeolAtop5}, {JeolAtop10}')
        # JeolAtop1, JeolAtop5, JeolAtop10 = detection(JeolA_dataloader, model, is_wt, is_m, True, temp_cur, result_cur,
        #                                              da, device, all_smiles_features, all_ID, 'xx03', 'JeolA')
        # print("非活泼H JeolA:", JeolAtop1, JeolAtop5, JeolAtop10)
        # init_logger("result_with_mm_wt.log", f'非活泼H JeolA:{JeolAtop1}, {JeolAtop5}, {JeolAtop10}')
        #
        # JeolAtop1, JeolAtop5, JeolAtop10 = detection(JeolA_dataloader, model, True, is_m, False, temp_cur, result_cur,
        #                                              da, device, all_smiles_features, all_ID, 'xx03', 'JeolA')
        # print("质量约束 JeolA:", JeolAtop1, JeolAtop5, JeolAtop10)
        # init_logger("result_with_mm_wt.log", f'质量约束 JeolA:{JeolAtop1}, {JeolAtop5}, {JeolAtop10}')
        #
        # JeolAtop1, JeolAtop5, JeolAtop10 = detection(JeolA_dataloader, model, False, True, False, temp_cur, result_cur,
        #                                              da, device, all_smiles_features, all_ID, 'xx03', 'JeolA')
        # print("多重性约束 JeolA:", JeolAtop1, JeolAtop5, JeolAtop10)
        # init_logger("result_with_mm_wt.log", f'多重性约束 JeolA:{JeolAtop1}, {JeolAtop5}, {JeolAtop10}')
        # JeolAtop1, JeolAtop5, JeolAtop10 = detection(JeolA_dataloader, model, True, True, False, temp_cur, result_cur,
        #                                              da, device, all_smiles_features, all_ID, 'xx03', 'JeolA')
        # print("质量+多重性约束 JeolA:", JeolAtop1, JeolAtop5, JeolAtop10)
        # init_logger("result_with_mm_wt.log", f'质量+多重性约束 JeolA:{JeolAtop1}, {JeolAtop5}, {JeolAtop10}')
        # —————————————————————————————————
        # 处理COCONUT验证集
        # CoconutBtop1, CoconutBtop5, CoconutBtop10 = detection(CoconutB_dataloader, model, is_wt, is_m, True, temp_cur, result_cur,
        #                                              da, device, all_smiles_features, all_ID, 'xx02', 'CoconutB')
        # print("非活泼H CoconutB:", CoconutBtop1, CoconutBtop5, CoconutBtop10)
        # init_logger("result_with_mm_wt.log", f'非活泼H CoconutB:{CoconutBtop1}, {CoconutBtop5}, {CoconutBtop10}')
        # CoconutBtop1, CoconutBtop5, CoconutBtop10 = detection(CoconutB_dataloader, model, False, is_m, False, temp_cur,
        #                                                       result_cur,
        #                                                       da, device, all_smiles_features, all_ID, 'xx02',
        #                                                       'CoconutB')
        # print("未加约束 CoconutB:", CoconutBtop1, CoconutBtop5, CoconutBtop10)
        # CoconutBtop1, CoconutBtop5, CoconutBtop10 = detection(CoconutB_dataloader, model, True, is_m, False, temp_cur, result_cur,
        #                                              da, device, all_smiles_features, all_ID, 'xx02', 'CoconutB')
        # print("质量约束 CoconutB:", CoconutBtop1, CoconutBtop5, CoconutBtop10)
        # init_logger("result_with_mm_wt.log", f'质量约束 CoconutB:{CoconutBtop1}, {CoconutBtop5}, {CoconutBtop10}')
        #
        # CoconutBtop1, CoconutBtop5, CoconutBtop10 = detection(CoconutB_dataloader, model, False, True, False, temp_cur, result_cur,
        #                                              da, device, all_smiles_features, all_ID, 'xx02', 'CoconutB')
        # print("多重性约束 CoconutB:", CoconutBtop1, CoconutBtop5, CoconutBtop10)
        # init_logger("result_with_mm_wt.log", f'多重性约束 CoconutB:{CoconutBtop1}, {CoconutBtop5}, {CoconutBtop10}')
        # CoconutBtop1, CoconutBtop5, CoconutBtop10 = detection(CoconutB_dataloader, model, True, True, False, temp_cur, result_cur,
        #                                              da, device, all_smiles_features, all_ID, 'xx02', 'CoconutB')
        # print("质量+多重性约束 CoconutB:", CoconutBtop1, CoconutBtop5, CoconutBtop10)
        # init_logger("result_with_mm_wt.log", f'质量+多重性约束 CoconutB:{CoconutBtop1}, {CoconutBtop5}, {CoconutBtop10}')

        detection_time = time.time()
        print("开始推理使用的时间：", detection_time-store_data_time)
        # local_epoch = checkpoint['epoch']
        # writer.add_scalars('Jeol/Top1', {'JeolA': JeolAtop1, 'JeolB': JeolBtop1, 'JeolC': JeolCtop1}, epoch)
        # writer.add_scalars('Jeol/Top5', {'JeolA': JeolAtop5, 'JeolB': JeolBtop5, 'JeolC': JeolCtop5}, epoch)
        # writer.add_scalars('Jeol/Top10', {'JeolA': JeolAtop10, 'JeolB': JeolBtop10, 'JeolC': JeolCtop10}, epoch)

        # CoconutBtop1, CoconutBtop5, CoconutBtop10 = detection(CoconutB_dataloader, model, is_wt, is_m, temp_cur,
        #                                                       result_cur, da,
        #                                                       device,
        #                                                       all_smiles_features, all_ID, 'xx06', 'CoconutA')
        # writer.add_scalars('CoconutB', {'top1': CoconutBtop1, 'top5': CoconutBtop5, 'top10': CoconutBtop10}, epoch)
        # print(CoconutBtop1, CoconutBtop5, CoconutBtop10)
        end_time = time.time()
        print("使用的总时间：", end_time-start_time)
    mongoclient.close()
    

    
