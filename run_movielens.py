import sys
sys.path.append(".")
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import pandas as pd
from SDSP.basic.features import DenseFeature, SparseFeature
from tqdm import tqdm
import numpy as np
import random
import time
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from SDSP.trainers.ctr_trainer import CTRTrainer
from SDSP.basic.data import DataGenerator, reduce_mem_usage
from SDSP.model.MMOE_SDSP import MMOE_SDSP
from SDSP.model.PLE_SDSP import PLE_SDSP

final_result = []




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


def get_movielens_data_rank_multidomain(data_path="data/ml-1m"):
    data = pd.read_csv(data_path+"/ml-1m.csv")
    data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])
    del data["genres"]

    group1 = {1, 18}
    group2 = {25}
    group3 = {35, 45, 50, 56}

    domain_num = 3

    data["domain_indicator"] = data["age"].apply(lambda x: map_group_indicator(x, [group1, group2, group3]))

    useless_features = ['title', 'timestamp']

    dense_features = ['age']
    scenario_features = ['domain_indicator']
    sparse_features = ['user_id', 'movie_id', 'gender', 'occupation', 'zip', "cate_id"]
    target = "rating"

    for feature in dense_features:
        data[feature] = data[feature].apply(lambda x: convert_numeric(x))
    if dense_features:
        sca = MinMaxScaler()  # scaler dense feature
        data[dense_features] = sca.fit_transform(data[dense_features])
    for feature in useless_features:
        del data[feature]
    for feat in tqdm(sparse_features):  # encode sparse feature
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    data[target] = data[target].apply(lambda x: convert_target(x))

    dense_feas = [DenseFeature(feature_name) for feature_name in dense_features]
    sparse_feas = [SparseFeature(feature_name, vocab_size=data[feature_name].nunique(), embed_dim=16) for feature_name
                   in sparse_features]
    scenario_feas = [SparseFeature(col, vocab_size=data[col].max() + 1, embed_dim=16) for col in scenario_features]
    y = data[target]
    del data[target]

    return dense_feas, sparse_feas, scenario_feas, data, y, domain_num

def map_group_indicator(age, list_group):
    l = len(list(list_group))
    for i in range(l):
        if age in list_group[i]:
            return i


def convert_target(val):
    v = int(val)
    if v > 3:
        return int(1)
    else:
        return int(0)


def convert_numeric(val):
    """
    Forced conversion
    """
    return int(val)


def df_to_dict_multi_domain(data, columns):
    """
    Convert the array to a dict type input that the network can accept
    Args:
        data (array): 3D datasets of type DataFrame (Length * Domain_num * feature_num)
        columns (list): feature name list
    Returns:
        The converted dict, which can be used directly into the input network
    """

    data_dict = {}
    for i in range(len(columns)):
        data_dict[columns[i]] = data[:, :, i]
    return data_dict





def main(dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device,  expert_num, proto_gamma, save_dir, seed):
    
    torch.manual_seed(seed)
    dataset_name = "Movielens"  
    dense_feas, sparse_feas, scenario_feas, x, y, domain_num= get_movielens_data_rank_multidomain(dataset_path)
    
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
        model = MMOE_SDSP(dense_feas+sparse_feas, scaled_batch, domain_num, n_expert = expert_num, expert_params={"dims": [16]}, tower_params={"dims": [8]})
    elif model_name == "PLE":
        model = PLE_SDSP(dense_feas+sparse_feas, scaled_batch,domain_num, n_level=1, n_expert_specific=2, n_expert_shared=expert_num, expert_params={"dims": [16]}, tower_params={"dims": [8]})
    ctr_trainer = CTRTrainer(model, dataset_name, init_iter=3, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay},
                             n_epoch=epoch, earlystop_patience=4, proto_gamma=proto_gamma, device=device,
                             model_path=save_dir,scheduler_params={"step_size": 2,"gamma": 0.85})
    ctr_trainer.fit(train_dataloader, val_dataloader)
    domain_logloss,domain_auc,logloss,auc = ctr_trainer.evaluate_multi_domain_loss(ctr_trainer.model, test_dataloader,domain_num,True)
    print(f'test auc: {auc} | test logloss: {logloss}')
    for d in range(domain_num):
        print(f'test domain {d} auc: {domain_auc[d]} | test domain {d} logloss: {domain_logloss[d]}|sim domain {d}: {model.sim_domain[d]}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="./SDSP/data/ml-1m")
    parser.add_argument('--model_name', default='PLE')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=4096)  #4096
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda')  #cuda:0
    parser.add_argument('--expert_num', type=int,  nargs='+', help="A list of integers", default=[1, 1, 1])
    parser.add_argument('--proto_gamma', type=float, default=1e-04, help="prtotype learning parameter")
    parser.add_argument('--save_dir', default='./SDSP/results/movielens')
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
