import torch
import numpy as np

def get_label_score(model, loader,device,judge_threshold=0.5):
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
    preds = np.concatenate(preds)+0
    labels = np.concatenate(labels)
    return scores,preds,labels