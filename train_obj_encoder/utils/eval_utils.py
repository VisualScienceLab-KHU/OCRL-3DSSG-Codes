import torch
import torch.nn.functional as F
import numpy as np
import clip

def evaluate_topk_object(objs_pred, objs_target, topk):
    res = []
    for obj in range(len(objs_pred)):
        obj_pred = objs_pred[obj]
        sorted_idx = torch.sort(obj_pred, descending=True)[1]
        gt = objs_target[obj]
        index = 1
        for idx in sorted_idx:
            if obj_pred[gt] >= obj_pred[idx] or index > topk:
                break
            index += 1
        res.append(index)
    return np.asarray(res)


def consine_classification(
    cls_matrix: torch.Tensor, # C X N_feat
    obj_feat: torch.Tensor,   # B X N_feat
    obj_gt: torch.Tensor
):
    # cls_matrix = F.normalize(cls_matrix, dim=-1)
    # obj_feat = F.normalize(obj_feat, dim=-1)
    sim_matrix = torch.mm(obj_feat, cls_matrix.T) # B X C
    # obj_pred = (sim_matrix + 1) * 0.5
    obj_pred = F.softmax(sim_matrix, dim=1)
    obj_label = torch.argmax(obj_gt, dim=1).long()
    return evaluate_topk_object(obj_pred, obj_label, topk=11)
