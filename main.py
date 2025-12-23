from data import AFQMC,collate_fn
import torch
from torch.utils.data import DataLoader
from model import BertForPairwiseCLS2
from tqdm.auto import tqdm
from train import train_loop,test_loop
from torch.optim import AdamW
from transformers import get_scheduler
from transformers import AutoTokenizer,AutoModel,AutoConfig
import torch.nn as nn



traindata=AFQMC('E:/traeproject/llm/03实战项目/01Paraphrase Identification Task/afqmc_public/train.json',max_samples=8000)
valdata=AFQMC('E:/traeproject/llm/03实战项目/01Paraphrase Identification Task/afqmc_public/dev.json',max_samples=3000)

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

train_dataloader = DataLoader(
        traindata,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
    )
valid_dataloader= DataLoader(valdata, batch_size=4, shuffle=False, collate_fn=collate_fn)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
config=AutoConfig.from_pretrained(checkpoint)

model=BertForPairwiseCLS2(config)
model.to(device)

epochs = 3
num_training_steps = epochs * len(train_dataloader)
learning_rate = 1e-5
epoch_num = 3
optimizer = AdamW(model.parameters(), lr=learning_rate)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)



loss_fn = nn.CrossEntropyLoss()


total_loss = 0.
best_acc = 0.

for ep in range(epoch_num):
    print(f"Epoch {ep+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, ep+1, total_loss)
    test_acc = test_loop(valid_dataloader, model, mode='Valid')
    
    if best_acc < test_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), f'epoch_{ep+1}_valid_acc_{(100*test_acc):0.1f}_model_weights.bin')
print("Done!")
