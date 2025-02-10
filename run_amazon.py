import sys
sys.path.append(".")
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from SDSP.basic.features import DenseFeature, SparseFeature
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from SDSP.trainers.ctr_trainer import CTRTrainer
from SDSP.basic.data import DataGenerator, reduce_mem_usage
from SDSP.model.MMOE_SDSP import MMOE_SDSP
from SDSP.model.PLE_SDSP import PLE_SDSP
import time


def adjust_batch_num(domain_num, batch_size):
    total_count = sum(domain_num)
    proportions = [count / total_count for count in domain_num]
    
    scaled_batch = [round(p * batch_size) for p in proportions]
    
    total_scaled = sum(scaled_batch)
    while total_scaled != batch_size:
        if total_scaled > batch_size:
            max_idx = np.argmax(scaled_batch)
            scaled_batch[max_idx] -= 1
        elif total_scaled < batch_size:
            min_idx = np.argmin(scaled_batch)
            scaled_batch[min_idx] += 1
        total_scaled = sum(scaled_batch)

    return scaled_batch

def get_amazon_data_dict(data_path='./data/amazon_5_core'):
    data = pd.read_csv(data_path + '/amazon.csv')
    domain_num = 3
    col_names = data.columns.values.tolist()
    dense_cols = []
    sparse_cols = [col for col in col_names if col not in dense_cols and col not in ['label', 'domain_indicator']]

    for feature in dense_cols:
        data[feature] = data[feature].apply(lambda x: convert_numeric(x))
    if dense_cols:
        sca = MinMaxScaler()  # scaler dense feature
        data[dense_cols] = sca.fit_transform(data[dense_cols])

    for feat in tqdm(sparse_cols):  # encode sparse feature
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    print("sparse cols:%d dense cols:%d" % (len(sparse_cols), len(dense_cols)))

    dense_feas = [DenseFeature(col) for col in dense_cols]
    sparse_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in sparse_cols]

    y = data["label"]
    del data["label"]
    x = data
    return dense_feas, sparse_feas, x, y, domain_num


def get_amazon_data_dict_adasparse(data_path='./data/amazon_5_core'):
    data = pd.read_csv(data_path + '/amazon.csv')
    domain_num = 3
    scenario_fea_num = 1

    col_names = data.columns.values.tolist()
    dense_cols = []
    scenario_cols = ['domain_indicator']
    sparse_cols = [col for col in col_names if col not in dense_cols and col not in ['label', 'domain_indicator']]

    for feature in dense_cols:
        data[feature] = data[feature].apply(lambda x: convert_numeric(x))
    if dense_cols:
        sca = MinMaxScaler()  # scaler dense feature
        data[dense_cols] = sca.fit_transform(data[dense_cols])

    for feat in tqdm(sparse_cols):  # encode sparse feature
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    print("scenario_cols:%d sparse cols:%d dense cols:%d" % (len(scenario_cols), len(sparse_cols), len(dense_cols)))

    dense_feas = [DenseFeature(col) for col in dense_cols]
    sparse_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in sparse_cols]
    scenario_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in scenario_cols]

    y = data["label"]
    del data["label"]
    x = data

    return (dense_feas, sparse_feas, scenario_feas, scenario_fea_num,
            x, y, domain_num)

def get_amazon_data_dict_ppnet(data_path='./data/amazon_5_core'):
    data = pd.read_csv(data_path + '/amazon.csv')
    domain_num = 3
    scenario_fea_num = 1
    col_names = data.columns.values.tolist()
    dense_cols = []
    id_cols = ['user', 'item']
    scenario_cols = ['domain_indicator']
    sparse_cols = [col for col in col_names if col not in dense_cols and col not in id_cols and col not in ['label', 'domain_indicator']]

    for feature in dense_cols:
        data[feature] = data[feature].apply(lambda x: convert_numeric(x))
    if dense_cols:
        sca = MinMaxScaler()  # scaler dense feature
        data[dense_cols] = sca.fit_transform(data[dense_cols])

    for feat in tqdm(sparse_cols):  # encode sparse feature
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    for feat in tqdm(id_cols):  # encode id feature
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    print("scenario_cols:%d sparse cols:%d dense cols:%d id cols:%d" % (len(scenario_cols), len(sparse_cols), len(dense_cols), len(id_cols)))

    dense_feas = [DenseFeature(col) for col in dense_cols]
    sparse_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in sparse_cols]
    scenario_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in scenario_cols]
    id_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in id_cols]

    y = data["label"]
    del data["label"]
    x = data
    return (dense_feas, sparse_feas, scenario_feas, id_feas, scenario_fea_num,
            x, y, domain_num)


def convert_numeric(val):
    """
    Forced conversion
    """
    return int(val)


def main(dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device,  expert_num, proto_gamma, save_dir, seed):
    torch.manual_seed(seed)
    dataset_name = "amazon_5_core"
    dense_feas, sparse_feas, x, y, domain_num = get_amazon_data_dict(dataset_path)

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=1)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=1)
 
    domain_id = x_train["domain_indicator"].tolist()
    samples_num = [0 for _ in range(domain_num)]
    for i in range(len(domain_id)):
        samples_num[domain_id[i]] += 1
    scaled_batch = adjust_batch_num(samples_num, batch_size)
    
    dg = DataGenerator(x_train, y_train)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(x_val=x_val, y_val=y_val, x_test=x_test,
                                                                               y_test=y_test, batch_size=scaled_batch, domain_id=x_train["domain_indicator"])
    if model_name == "MMOE":
        model = MMOE_SDSP(dense_feas+sparse_feas, scaled_batch, domain_num, n_expert = expert_num, expert_params={"dims": [32]}, tower_params={"dims": [16]})
    elif model_name == "PLE":
        model = PLE_SDSP(dense_feas+sparse_feas, scaled_batch,domain_num, n_level=1, n_expert_specific=2, n_expert_shared=expert_num, expert_params={"dims": [16]}, tower_params={"dims": [8]})
    ctr_trainer = CTRTrainer(model, dataset_name, init_iter=2, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay},
                             n_epoch=epoch, earlystop_patience=5, proto_gamma=proto_gamma,device=device, model_path=save_dir,
                             scheduler_params={"step_size": 4, "gamma": 0.95})
    ctr_trainer.fit(train_dataloader, val_dataloader)
    domain_logloss, domain_auc, logloss, auc = ctr_trainer.evaluate_multi_domain_loss(ctr_trainer.model,
                                                                                      test_dataloader, domain_num, True)
    print(f'test auc: {auc} | test logloss: {logloss}')
    for d in range(domain_num):
        print(f'test domain {d} auc: {domain_auc[d]} | test domain {d} logloss: {domain_logloss[d]}|sim domain {d}: {model.sim_domain[d]}')



if __name__ == '__main__':
    import argparse
    import warnings

    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="./SDSP/data/amazon_5_core")
    parser.add_argument('--model_name', default='PLE')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4096)  #4096
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--device', default='cuda')  #cuda:0
    parser.add_argument('--expert_num', type=int,  nargs='+', help="A list of integers", default=[1, 1, 1])
    parser.add_argument('--proto_gamma', type=float, default=1e-04, help="prtotype learning parameter")
    parser.add_argument('--save_dir', default='./SDSP/results/amazon')
    parser.add_argument('--seed', type=int, default=2008)

    args = parser.parse_args()
    print("\n" + "="*40)  # 分割线
    print(__file__)
    print(f"Expert Num: {args.expert_num}")
    print(f"Base Model: {args.model_name}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Proto Learning Parameter: {args.proto_gamma}")
    print("="*40 + "\n")  # 分割线
    main(args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay,
            args.device, args.expert_num, args.proto_gamma, args.save_dir, args.seed)
"""
python run_amazon.py --model_name PLE
"""
