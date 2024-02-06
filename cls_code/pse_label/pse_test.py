import torch
import numpy as np

def pseduo(model, train_unlabidx_loader,train_unlabidx,threshold,train10ratio,device,judge_threshold=0.5):
    model.eval()
    scores = []
    preds = []

    for x in train_unlabidx_loader:
        x = x.type(torch.FloatTensor).to(device)
        with torch.no_grad():
            logits = model(x)
            softmax_confid = logits.softmax(1)
            pred = softmax_confid[:,1]>judge_threshold
            score = softmax_confid.max(1)[0]
        scores.append(score.cpu().numpy())
        preds.append(pred.cpu().numpy())

    scores = np.concatenate(scores)
    preds = np.concatenate(preds)+0

    get_label_indexs = np.where(scores<threshold)
    preds[get_label_indexs]=-999
    train_unlabidx[get_label_indexs]=-999
    scores[get_label_indexs]=-999

    pse_labels,pse_idxs,pse_scores=preds,train_unlabidx,scores
    No_label_num=pse_labels.tolist().count(-999)
    print('pse_label num:%d'%(pse_labels.shape[0]-No_label_num))
    
    pse_testlabs,pse_testidxs=pse_testdata(pse_labels,pse_idxs,pse_scores,train10ratio)
    return  pse_testlabs,pse_testidxs


def pse_testdata(labs,idxs,scoes,train10ratio):     
    lab0_index=[i for i,x in enumerate(labs) if x==0]
    lab1_index=[i for i,x in enumerate(labs) if x==1]
    
    idxs1=idxs[lab1_index]
    labels1=labs[lab1_index]
    scores1=scoes[lab1_index]

    idxs0=idxs[lab0_index]
    labels0=labs[lab0_index]
    scores0=scoes[lab0_index]

    if len(lab1_index)>=int(len(lab0_index)*train10ratio/(1-train10ratio)):
        lab1_idx=scores1.argsort()[::-1][:int(len(lab0_index)*train10ratio/(1-train10ratio))]
        idxs1_we_need=idxs1[lab1_idx]
        labels1_we_need=labels1[lab1_idx]
        idxs10_we_need=idxs1_we_need.tolist()+idxs0.tolist()
        labels10_we_need=labels1_we_need.tolist()+labels0.tolist()
    else:
        lab0_idx=scores0.argsort()[::-1][:int(len(lab1_index)/train10ratio-len(lab1_index))]
        idxs0_we_need=idxs0[lab0_idx]
        labels0_we_need=labels0[lab0_idx]
        idxs10_we_need=idxs1.tolist()+idxs0_we_need.tolist()
        labels10_we_need=labels1.tolist()+labels0_we_need.tolist()

    shuffle_idx = np.arange(len(idxs10_we_need))
    np.random.shuffle(shuffle_idx)
    idxs10=np.array(idxs10_we_need)[shuffle_idx]
    labels10=np.array(labels10_we_need)[shuffle_idx]

    return labels10,idxs10
