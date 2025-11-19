import numpy as np
import torch
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt

def get_gt(objs_target, rels_target, edges, multi_rel_outputs):
    gt_edges = []
    for edge_index in range(len(edges)):
        idx_eo = edges[edge_index][0]
        idx_os = edges[edge_index][1]
        target_eo = objs_target[idx_eo]
        target_os = objs_target[idx_os]
        target_rel = []
        if multi_rel_outputs:
            assert rels_target.ndim == 2
            for i in range(rels_target.shape[-1]):
                if rels_target[edge_index][i] == 1:
                    target_rel.append(i)
        else:
            assert rels_target.ndim == 1
            if rels_target[edge_index] > 0: # not None
                target_rel.append(rels_target[edge_index])
        gt_edges.append((target_eo, target_os, target_rel))
    return gt_edges


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


def evaluate_topk_predicate(rels_preds, gt_edges, multi_rel_outputs, topk, confidence_threshold=0.5, epsilon=0.02):
    res = []
    for rel in range(len(rels_preds)):
        rel_pred = rels_preds[rel]
        # make the 'none' confidence the highest, if none of the rel classes are bigger than confidence_threshold
        # which means 'none' prediction in the multi binary cross entropy approach.
        # if multi_rel_outputs:
        #     if rel_pred.max() < confidence_threshold:
        #         rel_pred[0] = rel_pred.max() + epsilon
        
        sorted_conf_matrix, sorted_idx = torch.sort(rel_pred, descending=True)
        temp_topk = []
        rels_target = gt_edges[rel][2]
        
        if len(rels_target) == 0: # no gt relation
            indices = torch.where(sorted_conf_matrix < confidence_threshold)[0]
            if len(indices) == 0:
                index = topk + 1
            else:
                index = sorted(indices)[0].item()+1
            
            temp_topk.append(index)

        for gt in rels_target:
            index = 1
            for idx in sorted_idx:
                if rel_pred[gt] >= rel_pred[idx] or index > topk:
                    break
                index += 1
            temp_topk.append(index)
        
        temp_topk = sorted(temp_topk)
        counter = 0
        for tmp in temp_topk:
            res.append(tmp - counter)
            counter += 1
        #res += temp_topk
    return np.asarray(res)


def evaluate_topk(objs_pred, rels_pred, gt_rel, edges, multi_rel_outputs, topk, confidence_threshold=0.5, epsilon=0.02):
    res, cls = [], []
    # convert the score from log_softmax to softmax
    objs_pred = np.exp(objs_pred)
    if not multi_rel_outputs:
        rels_pred = np.exp(rels_pred)
    
    for edge in range(len(edges)):
        edge_from = edges[edge][0]
        edge_to = edges[edge][1]
        rel_predictions = rels_pred[edge]
        obj = objs_pred[edge_from]
        sub = objs_pred[edge_to]

        # make the 'none' confidence the highest, if none of the rel classes are bigger than confidence_threshold
        # which means 'none' prediction in the multi binary cross entropy approach.
        # if multi_rel_outputs:
        #     if rel_predictions.max() < confidence_threshold:
        #         rel_predictions[0] = rel_predictions.max() + epsilon

        size_o = len(obj)
        size_r = len(rel_predictions)

        node_score = np.matmul(obj.reshape(size_o, 1), sub.reshape(1, size_o))
        conf_matrix = np.matmul(node_score.reshape(size_o, size_o, 1), rel_predictions.reshape(1, size_r))
        conf_matrix_1d = conf_matrix.reshape(-1)
        sorted_args_1d = torch.sort(conf_matrix_1d, descending=True)[1]

        subject = gt_rel[edge][0]
        obj = gt_rel[edge][1]
        temp_topk, tmp_cls = [], []

        for predicate in gt_rel[edge][2]:
            index = 1
            for idx_1d in sorted_args_1d:
                idx = np.unravel_index(idx_1d, (size_o, size_o, size_r))
                gt_conf = conf_matrix[subject, obj, predicate]
                if gt_conf >= conf_matrix[idx] or index > topk:
                    break
                index += 1
            temp_topk.append(index)
            tmp_cls.append(predicate)
        
        temp_topk = sorted(temp_topk)
        counter = 0
        for tmp in temp_topk:
            assert (tmp - counter) > 0
            res.append(tmp - counter)
            counter += 1
        #res += temp_topk
        cls += tmp_cls
    
    return np.asarray(res), np.array(cls)


def evaluate_triplet_topk(objs_pred, rels_pred, gt_rel, edges, multi_rel_outputs, topk, confidence_threshold=0.5, epsilon=0.02, use_clip=False, obj_topk=None):
    res, triplet = [], []
    if not use_clip:
        # convert the score from log_softmax to softmax
        objs_pred = np.exp(objs_pred)
    else:
        # convert the score to softmax
        objs_pred = F.softmax(objs_pred, dim=-1)
    
    if not multi_rel_outputs:
        rels_pred = np.exp(rels_pred)

    sub_scores, obj_scores, rel_scores = [],  [],  []
    
    for edge in range(len(edges)):
        edge_from = edges[edge][0]
        edge_to = edges[edge][1]
        rel_predictions = rels_pred[edge]
        sub = objs_pred[edge_from]
        obj = objs_pred[edge_to]
        
        if obj_topk is not None:
            sub_pred = obj_topk[edge_from]
            obj_pred = obj_topk[edge_to]

        node_score = torch.einsum('n,m->nm',sub,obj)
        conf_matrix = torch.einsum('nl,m->nlm',node_score,rel_predictions)
        conf_matrix_1d = conf_matrix.reshape(-1)
        sorted_conf_matrix, sorted_args_1d = torch.sort(conf_matrix_1d, descending=True)
        
        # just take topk
        sorted_conf_matrix = sorted_conf_matrix[:topk]
        sorted_args_1d = sorted_args_1d[:topk]

        sub_gt= gt_rel[edge][0]
        obj_gt = gt_rel[edge][1]
        rel_gt = gt_rel[edge][2]
        temp_topk, tmp_triplet = [], []

        if len(rel_gt) == 0: # no gt relation
            indices = torch.where(sorted_conf_matrix < confidence_threshold)[0]
            if len(indices) == 0:
                index = topk + 1
            else:
                index = sorted(indices)[0].item()+1
            temp_topk.append(index)
            if obj_topk is not None:
                tmp_triplet.append([sub_gt.cpu(),sub_pred, obj_gt.cpu(), obj_pred, -1])
            else:
                tmp_triplet.append([sub_gt.cpu(),obj_gt.cpu(),-1])
        
        for predicate in rel_gt: # for multi class case
            gt_conf = conf_matrix[sub_gt, obj_gt, predicate]
            indices = torch.where(sorted_conf_matrix == gt_conf)[0]
            if len(indices) == 0:
                index = topk + 1
            else:
                index = sorted(indices)[0].item()+1
            temp_topk.append(index)
            if obj_topk is not None:
                tmp_triplet.append([sub_gt.cpu(),sub_pred, obj_gt.cpu(), obj_pred, predicate])
            else:
                tmp_triplet.append([sub_gt.cpu(), obj_gt.cpu(), predicate])
            
            sub_scores.append(sub)
            obj_scores.append(obj)
            rel_scores.append(rel_predictions)
            
   
        temp_topk = sorted(temp_topk)
        counter = 0
        for tmp in temp_topk:
            res.append(tmp - counter)
            counter += 1
        triplet += tmp_triplet
    
    return np.asarray(res), np.array(triplet), sub_scores, obj_scores, rel_scores


def evaluate_topk_recall(objs_pred, rels_pred, objs_target, rels_target, edges):
    top_k_obj = evaluate_topk_object(objs_pred, objs_target, topk=10)
    gt_edges = get_gt(objs_target, rels_target, edges, topk=10)
    top_k_predicate = evaluate_topk_predicate(rels_pred, gt_edges, multi_rel_outputs=True, topk=5)
    top_k = evaluate_triplet_topk(objs_pred, rels_pred, rels_target, edges, multi_rel_outputs=True, topk=100)
    return top_k, top_k_obj, top_k_predicate


def get_mean_recall(triplet_rank, cls_matrix, topk=[50, 100]):
    if len(cls_matrix) == 0:
        return np.array([0,0])

    mean_recall = [[] for _ in range(len(topk))]
    cls_num = int(cls_matrix.max())
    for i in range(cls_num):
        cls_rank = triplet_rank[cls_matrix[:,-1] == i]
        if len(cls_rank) == 0:
            continue
        for idx, top in enumerate(topk):
            mean_recall[idx].append((cls_rank <= top).sum() * 100 / len(cls_rank))
    mean_recall = np.array(mean_recall, dtype=np.float32)
    return mean_recall.mean(axis=1)


def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output


def read_json(split):
    """
    Reads a json file and returns points with instance label.
    """
    selected_scans = set()
    if split == 'train' :
        selected_scans = selected_scans.union(read_txt_to_list('<Your Path>/data/3DSSG_subset/train_scans.txt'))
        with open("<Your Path>/data/3DSSG_subset/relationships_train.json", "r") as read_file:
            data = json.load(read_file)
    elif split == 'val':
        selected_scans = selected_scans.union(read_txt_to_list('<Your Path>/data/3DSSG_subset/validation_scans.txt'))
        with open("<Your Path>/data/3DSSG_subset/relationships_validation.json", "r") as read_file:
            data = json.load(read_file)
    else:
        raise RuntimeError('unknown split type:',split)

    return data

def get_zero_shot_recall(triplet_rank, cls_matrix, obj_names, rel_name):
   
    train_data = read_json('train')
    scene_data = dict()
    for i in train_data['scans']:
        objs = i['objects']
        for rel in i['relationships']:
            if str(rel[0]) not in objs.keys():
                #print(f'{rel[0]} not in objs in scene {i["scan"]} split {i["split"]}')
                continue
            if str(rel[1]) not in objs.keys():
                #print(f'{rel[1]} not in objs in scene {i["scan"]} split {i["split"]}')
                continue
            triplet_name = str(obj_names.index(objs[str(rel[0])])) + ' ' + str(obj_names.index(objs[str(rel[1])])) + ' ' + str(rel_name.index(rel[-1]))
            if triplet_name not in scene_data.keys():
                scene_data[triplet_name] = 1
            scene_data[triplet_name] += 1
    
    val_data = read_json('val')
    zero_shot_triplet = []
    count = 0
    for i in val_data['scans']:
        objs = i['objects']
        for rel in i['relationships']:
            count += 1
            triplet_name = str(obj_names.index(objs[str(rel[0])])) + ' ' + str(obj_names.index(objs[str(rel[1])])) + ' ' + str(rel_name.index(rel[-1]))
            if triplet_name not in scene_data.keys():
                zero_shot_triplet.append(triplet_name)
    
    # get valid triplet which not appears in train data
    valid_triplet = []
    non_zero_shot_triplet = []
    all_triplet = []

    for i in range(len(cls_matrix)):
        if cls_matrix[i, -1] == -1:
            continue
        if len(cls_matrix[i]) == 5:
            triplet_name = str(cls_matrix[i][0]) + ' ' + str(cls_matrix[i][2]) + ' ' + str(cls_matrix[i][-1])
        elif len(cls_matrix[i]) == 3:
            triplet_name = str(cls_matrix[i][0]) + ' ' + str(cls_matrix[i][1]) + ' ' + str(cls_matrix[i][-1])
        else:
            raise RuntimeError('unknown triplet length:', len(cls_matrix[i]))

        if triplet_name in zero_shot_triplet:
            valid_triplet.append(triplet_rank[i])
        else:
            non_zero_shot_triplet.append(triplet_rank[i])
        
        all_triplet.append(triplet_rank[i])
    
    valid_triplet = np.array(valid_triplet)
    non_zero_shot_triplet = np.array(non_zero_shot_triplet)
    all_triplet = np.array(all_triplet)

    zero_shot_50 = (valid_triplet <= 50).mean() * 100
    zero_shot_100 = (valid_triplet <= 100).mean() * 100

    non_zero_shot_50 = (non_zero_shot_triplet <= 50).mean() * 100
    non_zero_shot_100 = (non_zero_shot_triplet <= 100).mean() * 100

    all_50 = (all_triplet <= 50).mean() * 100
    all_100 = (all_triplet <= 100).mean() * 100

    return (zero_shot_50, zero_shot_100), (non_zero_shot_50, non_zero_shot_100), (all_50, all_100)

    
def get_head_body_tail(cls_matrix_list, topk_pred_list, relation_names):
    def cal_mA(cls_dict):
        predicate_mean_1, predicate_mean_3, predicate_mean_5 = [], [], []
        for i in cls_dict.keys():
            l = len(cls_dict[i])
            if l > 0:
                m_1 = (np.array(cls_dict[i]) <= 1).sum() / len(cls_dict[i])  # 
                m_3 = (np.array(cls_dict[i]) <= 3).sum() / len(cls_dict[i])
                m_5 = (np.array(cls_dict[i]) <= 5).sum() / len(cls_dict[i])
                predicate_mean_1.append(m_1)
                predicate_mean_3.append(m_3)
                predicate_mean_5.append(m_5) 
           
        predicate_mean_1 = np.mean(predicate_mean_1) if len(predicate_mean_1)>0 else 0.0
        predicate_mean_3 = np.mean(predicate_mean_3) if len(predicate_mean_3)>0 else 0.0
        predicate_mean_5 = np.mean(predicate_mean_5) if len(predicate_mean_5)>0 else 0.0
        return [predicate_mean_1 * 100, predicate_mean_3 * 100, predicate_mean_5 * 100]
    
    head_relation = ["left", "right", "front", "behind", "close by", "same as", "attached to", "standing on"]
    body_relation = ["bigger than", "smaller than", "higher than", "lower than", "lying on", "hanging on"]
    tail_relation = ["supported by", "inside", "same symmetry as", "connected to", "leaning against", "part of", "belonging to", "build in", "standing in", "cover", "lying in", "hanging in"]

    head_relation = [relation_names.index(i) for i in head_relation if i in relation_names]
    body_relation = [relation_names.index(i) for i in body_relation if i in relation_names]
    tail_relation = [relation_names.index(i) for i in tail_relation if i in relation_names]

    head_cls_dict = {i:[] for i in head_relation}
    body_cls_dict = {i:[] for i in body_relation}
    tail_cls_dict = {i:[] for i in tail_relation}

        
    for idx, j in enumerate(cls_matrix_list):
        if j[-1] in head_cls_dict.keys():
            head_cls_dict[j[-1]].append(topk_pred_list[idx])
        if j[-1] in body_cls_dict.keys():
            body_cls_dict[j[-1]].append(topk_pred_list[idx])
        if j[-1] in tail_cls_dict.keys():
            tail_cls_dict[j[-1]].append(topk_pred_list[idx])
    
    head_mA = cal_mA(head_cls_dict)
    body_mA = cal_mA(body_cls_dict)
    tail_mA = cal_mA(tail_cls_dict)
    return head_mA, body_mA, tail_mA

def get_per_rel_class(cls_matrix_list, topk_pred_list, relation_names):
    def cal_R(cls_dict):
        result = {}
        for i in cls_dict:
            if len(cls_dict[i]) > 0:
                preds = np.array(cls_dict[i])
                r1 = (preds <= 1).sum() / len(preds)
                r3 = (preds <= 3).sum() / len(preds)
                r5 = (preds <= 5).sum() / len(preds)
                result[i] = [r1 * 100, r3 * 100, r5 * 100]
            else:
                result[i] = [0.0, 0.0, 0.0]
        return result

    num_relations = len(relation_names)
    cls_dict = {i: [] for i in range(num_relations)}

    for idx, row in enumerate(cls_matrix_list):
        true_label = row[-1]
        if true_label in cls_dict:
            cls_dict[true_label].append(topk_pred_list[idx])

    label_wise_recall = cal_R(cls_dict)
    return label_wise_recall

def get_per_obj_class(gt_obj_label, topk_obj_list, object_names):
    def cal_R(cls_dict):
        result = {}
        for i in cls_dict:
            if len(cls_dict[i]) > 0:
                objs = np.array(cls_dict[i])
                r1 = (objs <= 1).sum() / len(objs)
                r3 = (objs <= 3).sum() / len(objs)
                r5 = (objs <= 5).sum() / len(objs)
                result[i] = [r1 * 100, r3 * 100, r5 * 100]
            else:
                result[i] = [0.0, 0.0, 0.0]
        return result

    num_relations = len(object_names)
    cls_dict = {i: [] for i in range(num_relations)}

    for idx, row in enumerate(gt_obj_label):
        true_label = row
        if true_label in cls_dict:
            cls_dict[true_label].append(topk_obj_list[idx])

    label_wise_recall = cal_R(cls_dict)
    return label_wise_recall

def compute_predicate_acc_per_class(cls_matrix_list, topk_pred_list, rel_label_list, epoch):
    cls_dict = {}
    for i in range(26):
        cls_dict[i] = []
    
    total_cnt=0
    for idx, j in enumerate(cls_matrix_list):
        if j[-1] != -1:
            cls_dict[j[-1]].append(topk_pred_list[idx])
            total_cnt+=1
    
    predicate_mean = []
    for i in range(26):
        l = len(cls_dict[i])
        if l > 0:
            m_1 = (np.array(cls_dict[i]) <= 1).sum() / len(cls_dict[i])
            m_3 = (np.array(cls_dict[i]) <= 3).sum() / len(cls_dict[i])
            m_5 = (np.array(cls_dict[i]) <= 5).sum() / len(cls_dict[i])
            
            cls=rel_label_list[i]
            freq=len(cls_dict[i])
            predicate_mean.append([cls,freq,[m_1,m_3,m_5]])
    predicate_mean.sort(key=lambda x: x[1],reverse=True)
    for i in range(len(predicate_mean)):
        predicate_mean[i][1]
    
    fig1=draw_graph(predicate_mean,0)
    print("--------------------------------")
    print(predicate_mean)
    print("--------------------------------")
    #fig2=draw_graph(predicate_mean,1)
    #fig3=draw_graph(predicate_mean,2)
    #wandb.log({f"Accuracy@1 (epoch {epoch})": wandb.Image(fig1)}, step=epoch)
    plt.close(fig1)
    #plt.close(fig2)
    #plt.close(fig3)
    
def draw_graph(predicate_mean, topk_index):
    fig, ax1 = plt.subplots(figsize=(13,10))

    ax1.set_xlabel("Class")
    ax1.set_ylabel("Frequency", color="blue")
    ax1.plot([predicate_mean[i][0] for i in range(len(predicate_mean))], [predicate_mean[i][1] for i in range(len(predicate_mean))], marker="o", linestyle="-", color="blue", label="Frequency")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.tick_params(axis="x", rotation=90)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Acc", color="orange")
    ax2.bar([predicate_mean[i][0] for i in range(len(predicate_mean))], [predicate_mean[i][2][topk_index] for i in range(len(predicate_mean))], alpha=0.6, color="orange", label="Acc")
    ax2.tick_params(axis="y", labelcolor="orange")
    
    fig.tight_layout()
    # 그래프 제목
    tmp=[1,3,5]
    fig.suptitle(f"rel_acc_per_cls@{tmp[topk_index]}")
    plt.subplots_adjust(top=0.85)

    return fig