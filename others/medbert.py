from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from tqdm import tqdm
from utils import BaseDatasets

data_path = '/nas2/lishengping/data/med_data/data_with_pos_ner/test.jsonl'

model_name = '/nas2/lishengping/models/pretrain_models/medbert'
add_pos = False
add_ner = False
add_prompt = False
max_len = 128
add_cls_token = True
seed = 42
world_size = 1
perrank_batch_size = 4


tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

model = AutoModelForMaskedLM.from_pretrained(model_name)


train_dataset = BaseDatasets(data_path,
                                tokenizer=tokenizer,
                                add_pos=add_pos,
                                add_ner=add_ner,
                                add_prompt=add_prompt,
                                max_len=max_len,
                                add_cls_token=add_cls_token,
                                seed=seed,
                                rank=0,
                                world_size=world_size)
train_loader = DataLoader(train_dataset, batch_size=perrank_batch_size, shuffle=False, drop_last=True)


epochs = 3
learning_rate = 2e-5

optimizer = AdamW(model.parameters(), lr=learning_rate)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
device = 'cuda:1'
model.to(device)
model.train()

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    total_loss = 0

    for index,  batch in enumerate(train_loader):
        batch_inputs = batch['input_ids'].long().to(device)
        batch_labels = batch['labels'].long().to(device)
 #       batch_poss = batch['poss'].long().to(device)
  #      batch_ners = batch['ners'].long().to(device)

        optimizer.zero_grad()
        outputs = model(batch_inputs, labels=batch_inputs)
        loss = outputs.loss
        if index % 100 == 0:
            print(f'step: {index} loss: {loss.item()}')
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Average Loss: {average_loss}")

model.save_pretrained("finetuned_medbert_model")

