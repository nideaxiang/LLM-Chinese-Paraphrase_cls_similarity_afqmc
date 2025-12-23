from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer
import torch

class AFQMC(Dataset):
    def __init__(self,data_path,max_samples=5000):
        self.data = self.load_data(data_path,max_samples)
    def load_data(self,data_path,max_samples=5000):
        data = {}
        with open(data_path,'r',encoding='utf-8') as f:
            #以文本模式读取json 
            #单条加载
            for idx, line in enumerate(f):
                if idx >= max_samples:          # 只取前 max_samples 条
                    break
                #strip()方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
                #loads()传入的是一个字符串，返回一个python对象
                sample = json.loads(line.strip())
                data[idx] = sample
        return data
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        sample = self.data[idx]
        return sample
checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)  

def collate_fn(batch):
    batch_sen1=[]
    batch_sen2=[]
    batch_labels=[]
    for example in batch:
        batch_sen1.append(example["sentence1"])
        batch_sen2.append(example["sentence2"])
        batch_labels.append(int(example["label"]))#从字符串转换为整数
    tokenized_batch = tokenizer(
        batch_sen1,
        batch_sen2,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    tokenized_batch["labels"] = torch.tensor(batch_labels) #将列表转换为张量
    return tokenized_batch


    