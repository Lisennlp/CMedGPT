import os
import json
import random
from tqdm import tqdm
import copy
import pickle

import torch
from torch.utils.data import Dataset


class BaseDatasets(Dataset):
    def __init__(
        self,
        file_path,
        tokenizer,
        add_pos=False,
        add_ner=False,
        add_prompt: int = 0,
        max_len: int = 1024,
        add_cls_token: bool = False,
        seed=42,
        rank=0,
        world_size=1,
    ):
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.add_pos = add_pos
        self.add_ner = add_ner
        self.add_prompt = add_prompt
        self.add_cls_token = add_cls_token
        self.generate_prompt()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.iter_nums = 0
        random.seed(self.seed)

        if os.path.isdir(file_path):
            files = os.listdir(file_path)
            total_data = []
            for file in files:
                path = os.path.join(file_path, file)
                data = self.load_data(file_path=path)
                total_data.extend(data)
        else:
            total_data = self.load_data(file_path)
        random.shuffle(total_data)
        size = len(total_data)
        perrank_nums = size // self.world_size

        self.data = total_data[self.rank * perrank_nums: (self.rank + 1) * perrank_nums]
     

    def generate_prompt(self):
        self.prompt_tokens = [f"[unused{i}]" for i in range(1, self.add_prompt + 1)]

    def tokenize(self, line):
        if isinstance(line, str):
            token_ids = self.tokenizer.encode(line)
        elif isinstance(line, list):
            token_ids = self.tokenizer.convert_tokens_to_ids(line)
        else:
            raise ValueError("Unkonw input type!!!")
        if self.add_cls_token:
            token_ids = [self.tokenizer.cls_token_id] + token_ids
        return token_ids
    
    def combine_text(self, data):
        length, count = 0, 0
        tokens, ners, poss = [], [], []
        combine_data = []
        while count < len(data) - 1:
            token, pos, ner = data[count]
            length += len(token)
            if length > self.max_len - self.add_cls_token:
                combine_data.append([tokens, ners, poss])
                tokens, ners, poss = token, pos, ner
                length = len(token)
            tokens.extend(token)
            ners.extend(ner)
            poss.extend(pos)
            count += 1
        return combine_data

    def load_data(self, file_path):
        filename = file_path.rstrip('.jsonl') + f'{self.max_len}.pkl.other2'
        if os.path.exists(filename):
            print(f'’{filename}‘ is existed, starting to load data...')
            data = pickle.load(open(filename, 'rb'))
            return data
        data = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in tqdm(file, desc="Loading data"):
                poss, ners = [], []
                line = eval(line)
                assert isinstance(line, dict)
                tokens = self.prompt_tokens + list(line["tokens"])
               
                if self.add_pos:
                    poss = [2] * len(self.prompt_tokens) + list(line["poss"])
                    assert len(poss) == len(tokens)
                if self.add_ner:
                    ners = [2] * len(self.prompt_tokens) + line["ners"]
                    assert len(ners) == len(tokens)
                data.append([tokens, poss, ners])
        data = self.combine_text(data)
        print(f"Data nums: {len(data)}")
        pickle.dump(data, open(filename, 'wb'))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, poss, ners = self.data[idx]
        input_ids = self.tokenize(inputs)
        if self.add_cls_token and self.add_pos:
            poss = [0] + poss
            assert len(input_ids) == len(poss)
        if self.add_cls_token and self.add_ner:
            ners = [0] + ners
            assert len(input_ids) == len(ners)

        input_ids = input_ids[: self.max_len]
        poss = poss[: self.max_len]
        ners = ners[: self.max_len]

        labels = copy.deepcopy(input_ids)
        length = len(input_ids)
        while length < self.max_len:
            input_ids.append(0)
            labels.append(0)
            if self.add_pos:
                poss.append(0)
            if self.add_ner:
                ners.append(0)
            length += 1
        assert len(labels) == len(input_ids), print(len(labels), len(input_ids))
        input_ids = torch.tensor(input_ids).squeeze()
        labels = torch.tensor(labels).squeeze()
        poss = torch.tensor(poss).squeeze()
        ners = torch.tensor(ners).squeeze()
        # 打印前2个样本，查看样本是否正确
        if self.iter_nums < 2:
            print(f'prompt_tokens: {self.prompt_tokens}')
            print(f'prompt_ids: {input_ids[1 :len(self.prompt_tokens) + 1]}')
            print(f'input_ids: {input_ids} input_ids: {input_ids.shape}')
            print(f'labels: {labels} shape: {labels.shape}')
            if self.add_pos:
                print(f'poss: {poss} poss: {poss.shape} max: {poss.max()}')
            if self.add_ner:
                print(f'ners: {ners} ners: {ners.shape} max: {ners.max()}')
        self.iter_nums += 1
        return {"input_ids": input_ids, "labels": labels, "poss": poss, "ners": ners}
