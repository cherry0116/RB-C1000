import torch

def build_opt(model):
    if model.name=='resnet18':            
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    else:
        print('Invalid opt!')
        exit()
    return optimizer,scheduler