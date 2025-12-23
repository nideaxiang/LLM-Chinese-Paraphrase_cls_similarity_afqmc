from tqdm.auto import tqdm
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_loop(dataloader,model,loss_fn,optimizer,scheduler,epoch,total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = (epoch-1)*len(dataloader)

    model.train()
    for step,batch in enumerate(dataloader,start=1):
        batch=batch.to(device)
        y=batch.pop('labels', None) 
        pred=model(batch)
        loss = loss_fn(pred, y)

        optimizer.zero_grad() # 清空梯度
        loss.backward() # 计算梯度
        optimizer.step() # 更新模型参数
        scheduler.step() # 更新学习率,动态调整学习率

        total_loss += loss.item() # 累加损失
        progress_bar.set_description(f'loss: {total_loss/(finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(dataloader, model, mode='Valid'):
    assert mode in ['Test', 'Valid'], "mode must be 'Test' or 'Valid'"
    size = len(dataloader.dataset)
    correct = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch=batch.to(device)
            y=batch.pop('labels', None) 
            pred=model(batch)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    print(f"{mode} Accuracy: {(100*correct):>0.1f}%")
    return correct
    
