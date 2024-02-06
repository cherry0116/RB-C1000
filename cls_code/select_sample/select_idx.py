import torch
import numpy as np
import random

def select(main_opt,model,total_num,train_unlabidx_loader,train_unlabidx,device):
    select_num=int(total_num*(1-main_opt.first_select_ratio))
    if main_opt.hard_select=='yes':
        model.eval()
        scores = []

        for x in train_unlabidx_loader:
            x = x.type(torch.FloatTensor).to(device)
            with torch.no_grad():
                logits = model(x)
                softmax_confid = logits.softmax(1)
                score = softmax_confid.max(1)[0]
            scores.append(score.cpu().numpy())
        scores = np.concatenate(scores)

        score_idx_small2big=scores.argsort()
        train_unlabidx_score_small2big=train_unlabidx[score_idx_small2big]

        select_train_unidxs_idxs=train_unlabidx_score_small2big[:select_num]
        return  select_train_unidxs_idxs
    else:
        random_sample=random.sample(range(0, len(train_unlabidx)), select_num)
        select_train_unidxs_idxs=train_unlabidx[random_sample]
        return  select_train_unidxs_idxs


