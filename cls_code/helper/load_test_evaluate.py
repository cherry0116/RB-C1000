import torch
import numpy as np
from sklearn import metrics

def evalute(model, loader,device,judge_threshold=0.5):
    model.eval()
    scores = []
    preds = []
    labels = []
    for x,y in loader:
        x,y = x.type(torch.FloatTensor).to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            softmax_confid = logits.softmax(1)
            pred = softmax_confid[:,1]>judge_threshold
            score = softmax_confid.max(1)[0]
        scores.append(score.cpu().numpy())
        preds.append(pred.cpu().numpy())
        labels.append(y.cpu().numpy())
    scores = np.concatenate(scores)
    preds,labels = np.concatenate(preds)+0, np.concatenate(labels)
    confu_matrix=metrics.confusion_matrix(labels,preds)

    delta=1e-12   
    TN=confu_matrix[0][0]
    FP=confu_matrix[0][1]
    FN=confu_matrix[1][0]
    TP=confu_matrix[1][1]
    precision=float(TP)/float(TP+FP+delta)
    accuracy=float(TP+TN)/float(TP+FP+TN+FN)
    recall=float(TP)/float(TP+FN+delta)
    F1_score=(2*precision*recall)/(precision+recall+delta)
    MCC=float(TP*TN-FP*FN)/float(((TP+FP+delta)*(TP+FN+delta)*(TN+FP+delta)*(TN+FN+delta))**0.5)
    #scores.argsort() from small to big
    #scores.argsort()[::-1] from big to small
    #positive AP
    APs = []
    
    for cat in range(2):
        align_res, score = labels[preds==cat]==cat, scores[preds==cat]
        if align_res.tolist()==[]:
            APs.append(0)
        elif align_res.tolist()==[False]:
            APs.append(0)
        elif align_res.tolist()==[True]:
            APs.append(1)
        else:
            sorted_res = align_res[score.argsort()[::-1]]
            APs.append(np.stack([sorted_res[:tp_index+1].mean() for tp_index in np.where(sorted_res)[0]]).mean())
    mAP = np.array(APs).mean()
    results = {
        'Pre': precision,
        'Acc': accuracy,
        'Rec': recall,
        'F1': F1_score,
        'Mcc': MCC,
        'Ap': APs[1]
    }
    results['Sum'] = np.array([_ for _ in results.values()]).mean()
    con_matrix={
        'TN':TN,
        'FP':FP,
        'FN':FN,
        'TP':TP
    }
    return results,con_matrix