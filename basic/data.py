import random
import torch
import numpy as np
import pandas as pd
import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, mean_squared_error
from torch.utils.data import Dataset, DataLoader, random_split,Sampler


class TorchDataset(Dataset):

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x_dict_temp = {k: v.iloc[index] for k, v in self.x.items()}  # 获取 DataFrame 行
        y_temp = self.y.iloc[index]  # 获取 Series 值
        return x_dict_temp, y_temp

    def __len__(self):
        return len(self.y)


class PredictDataset(Dataset):

    def __init__(self, x):
        super().__init__()
        self.x = x

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.x.items()}

    def __len__(self):
        return len(self.x[list(self.x.keys())[0]])


class DataGenerator(object):

    def __init__(self, x, y):
        super().__init__()
        self.dataset = TorchDataset(x, y)
        self.length = len(self.dataset)

    def generate_dataloader(self, x_val=None, y_val=None, x_test=None, y_test=None, domain_id=None, split_ratio=None, batch_size=16,
                            num_workers=8):
        if split_ratio != None:
            train_length = int(self.length * split_ratio[0])
            val_length = int(self.length * split_ratio[1])
            test_length = self.length - train_length - val_length
            print("the samples of train : val : test are  %d : %d : %d" % (train_length, val_length, test_length))
            train_dataset, val_dataset, test_dataset = random_split(self.dataset,
                                                                    (train_length, val_length, test_length))
        else:
            train_dataset = self.dataset
            val_dataset = TorchDataset(x_val, y_val)
            test_dataset = TorchDataset(x_test, y_test)
        batch_size2 = sum(batch_size)
        batch_sampler = MultiDomainSampler(train_dataset, domain_id, batch_size)
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size2, sampler=batch_sampler, num_workers = num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size = batch_size2, shuffle = False, num_workers = num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size = batch_size2, shuffle = False, num_workers = num_workers)
        return train_dataloader, val_dataloader, test_dataloader


class MultiDomainSampler(Sampler):
    def __init__(self, data, domain_id, class_num):
        self.data = data
        self.domain_id = domain_id
        self.class_num = class_num
        self.indices = {
            dom: np.where(domain_id == dom)[0].tolist() for dom in range(len(class_num))
        }

    def __iter__(self):
        iter_len = []
        for i in range(len(self.class_num)):
            iter_len.append(len(self.indices[i])/self.class_num[i])
            np.random.shuffle(self.indices[i])
        iter_num = int(min(iter_len))
        final_indices = []
        for i in range(iter_num):
            indice_temp = []
            for d in range(len(self.class_num)):
                indice_temp.extend(self.indices[d][i*self.class_num[d]:((i+1)*self.class_num[d])]) 

            np.random.shuffle(indice_temp)
            final_indices.extend(indice_temp)
        return iter(final_indices)

    def __len__(self):
        return len(self.domain_id)

def get_auto_embedding_dim(num_classes):
    """ Calculate the dim of embedding vector according to number of classes in the category
    emb_dim = [6 * (num_classes)^(1/4)]
    reference: Deep & Cross Network for Ad Click Predictions.(ADKDD'17)
    Args:
        num_classes: number of classes in the category
    
    Returns:
        the dim of embedding vector
    """
    return np.floor(6 * np.pow(num_classes, 0.26))



def df_to_dict(data):
    """
    Convert the DataFrame to a dict type input that the network can accept
    Args:
        data (pd.DataFrame): datasets of type DataFrame
    Returns:
        The converted dict, which can be used directly into the input network
    """
    data_dict = data.to_dict('list')
    for key in data.keys():
        data_dict[key] = np.array(data_dict[key])
    return data_dict

def reduce_mem_usage(df):
    """Reduce memory.
    Args:
        df (pd.dataframe): a pandas dataframe
    Returns:
        df (pd.dataframe): a pandas dataframe
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('-----Memory compression starts-----')
    # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    # print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('-----Memory compression ends-----')
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))


    return df
