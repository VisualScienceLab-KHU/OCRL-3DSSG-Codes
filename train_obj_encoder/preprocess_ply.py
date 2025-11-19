import numpy as np
import torch
from tqdm import tqdm
from itertools import product
from glob import glob
import pickle

from dataset.preprocess import compute, load_mesh, dataset_loading_3RScan
from config import dataset_config, DATA_PATH, PREPROCESS_PATH, config_system
import h5py
import os
import cv2
import argparse
from multiprocessing import Pool

"""
Preprocessing for 3DSSG Dataset: Better Feature w. Simsiam-like SSL settings

Things to use
 - Multi-view images
 - Entire Scan Point Cloud

Things to store:
 - point cloud per object instance
 - instance label (for zero-shot text prompt)
 - multi-view image with properly cropped

Output Datastructure
 h5py
 - numpy point cloud
 - numpy image
 - labels
"""

parser = argparse.ArgumentParser(description="Example script for argparse")
parser.add_argument("--split", type=str, default="train_scans", choices=["train_scans", "validation_scans"])
args = parser.parse_args()


def read_relationship_json(mconfig, data, selected_scans:list):
    rel, objs, scans = dict(), dict(), []

    for scan_i in data['scans']:
        if scan_i["scan"] == 'fa79392f-7766-2d5c-869a-f5d6cfb62fc6':
            if mconfig["label_file"] == "labels.instances.align.annotated.v2.ply":
                '''
                In the 3RScanV2, the segments on the semseg file and its ply file mismatch. 
                This causes error in loading data.
                To verify this, run check_seg.py
                '''
                continue
        if scan_i['scan'] not in selected_scans:
            continue
            
        relationships_i = []
        for relationship in scan_i["relationships"]:
            relationships_i.append(relationship)
            
        objects_i = {}
        for id, name in scan_i["objects"].items():
            objects_i[int(id)] = name

        rel[scan_i["scan"] + "_" + str(scan_i["split"])] = relationships_i
        objs[scan_i["scan"]+"_"+str(scan_i['split'])] = objects_i
        scans.append(scan_i["scan"] + "_" + str(scan_i["split"]))

    return rel, objs, scans

def zero_mean(point):
    mean = torch.mean(point, dim=0)
    point -= mean.unsqueeze(0)
    ''' without norm to 1  '''
    # furthest_distance = point.pow(2).sum(1).sqrt().max() # find maximum distance for each n -> [n]
    # point /= furthest_distance
    return point  

def retrieval_rgb_img(
    points, instances, split_id,
    scan_id, multiview_path, 
    instance2labelName, num_points, 
    all_edge, rel_json, padding=0.2
):
    all_instance = list(np.unique(instances))
    nodes_all = list(instance2labelName.keys())
    
    if 0 in all_instance: # remove background
        all_instance.remove(0)
    
    nodes = []
    for i, instance_id in enumerate(nodes_all):
        if instance_id in all_instance:
            nodes.append(instance_id)
    
    # get edge (instance pair) list, which is just index, nodes[index] = instance_id
    if all_edge:
        edge_indices = list(product(list(range(len(nodes))), list(range(len(nodes)))))
        edge_indices = [i for i in edge_indices if i[0] != i[1]] # filter out (i,i)
    else:
        edge_indices = [(nodes.index(r[0]), nodes.index(r[1])) for r in rel_json if r[0] in nodes and r[1] in nodes]
    
    dim_point = points.shape[-1]
    instances_box, label_node = dict(), []
    
    for i, instance_id in enumerate(nodes):
        assert instance_id in all_instance, "invalid instance id"
        # get node label name
        instance_name = instance2labelName[instance_id]
        label_node.append(classNames.index(instance_name))
        obj_pointset = points[np.where(instances == instance_id)[0]]
        min_box = np.min(obj_pointset[:,:3], 0) - padding
        max_box = np.max(obj_pointset[:,:3], 0) + padding
        instances_box[instance_id] = (min_box, max_box)  
        
        num_point_sample = len(obj_pointset) if num_points > len(obj_pointset) else num_points
        choice = np.random.choice(len(obj_pointset), num_point_sample, replace=True)
        obj_pointset = obj_pointset[choice, :]
        obj_pointset = torch.from_numpy(obj_pointset.astype(np.float32))
        obj_pointset[:,:3] = zero_mean(obj_pointset[:,:3])
        
        save_path = PREPROCESS_PATH if args.split == "train_scans" else PREPROCESS_PATH + "_val"
        if not os.path.isdir(f"{save_path}/{scan_id}/{split_id}"):
            os.makedirs(f"{save_path}/{scan_id}/{split_id}", exist_ok=True)
        with h5py.File(f"{save_path}/{scan_id}/{split_id}/{instance_name}_{instance_id}_data.h5", "w") as f:
            image_path_list = glob(f"{multiview_path}/instance_{instance_id}_class_{instance_name}_view*_*_*.jpg")
            for i, _p in enumerate(image_path_list):
                multi_view_img_np = cv2.imread(_p)
                f.create_dataset(f"rgb_view_{i}", data=multi_view_img_np, dtype=np.float32)
            f.create_dataset("obj_point", data=obj_pointset.numpy(), dtype=np.float32)
            f.attrs["semantic_name"] = instance_name
            f.attrs["semantic_id"] = instance_id
            f.attrs["dim_point"] = dim_point
            f.attrs["num_points"] = num_points
    
    with open(f"{save_path}/{scan_id}/{split_id}/instance_box.json", "wb") as f:
        pickle.dump(instances_box, f)
    

if __name__ == "__main__":
    
    config = config_system
    mconfig = dataset_config
    root = mconfig["root"]
    split = args.split
    root_3rscan = f"{DATA_PATH}/3RScan/data/3RScan"
    label_type = '3RScan160'
    scans = []
    multi_rel_outputs = True
    use_rgb = True
    use_normal = True
    max_edges=-1
    use_descriptor = config["MODEL"]["use_descriptor"]
    use_data_augmentation = mconfig["use_data_augmentation"]
    use_2d_feats = config["MODEL"]["use_2d_feats"]
    
    if mconfig["selection"] == "":
        mconfig["selection"] = root
    classNames, relationNames, data, selected_scans = dataset_loading_3RScan(root, mconfig["selection"], split)        
    
    # for multi relation output, we just remove off 'None' relationship
    if multi_rel_outputs:
        relationNames.pop(0)
            
    wobjs, wrels, o_obj_cls, o_rel_cls = compute(classNames, relationNames, data, selected_scans, False)
    w_cls_obj = torch.from_numpy(np.array(o_obj_cls)).float().to(config["DEVICE"])
    w_cls_rel = torch.from_numpy(np.array(o_rel_cls)).float().to(config["DEVICE"])
    
    # for single relation output, we set 'None' relationship weight as 1e-3
    if not multi_rel_outputs:
        w_cls_rel[0] = w_cls_rel.max()*10
    
    w_cls_obj = w_cls_obj.sum() / (w_cls_obj + 1) /w_cls_obj.sum()
    w_cls_rel = w_cls_rel.sum() / (w_cls_rel + 1) /w_cls_rel.sum()
    w_cls_obj /= w_cls_obj.max()
    w_cls_rel /= w_cls_rel.max()
    
    # print some info
    print('=== {} classes ==='.format(len(classNames)))
    for i in range(len(classNames)):
        print('|{0:>2d} {1:>20s}'.format(i,classNames[i]),end='')
        if w_cls_obj is not None:
            print(':{0:>1.3f}|'.format(w_cls_obj[i]),end='')
        if (i+1) % 2 ==0:
            print('')
    print('')
    print('=== {} relationships ==='.format(len(relationNames)))
    for i in range(len(relationNames)):
        print('|{0:>2d} {1:>20s}'.format(i,relationNames[i]),end=' ')
        if w_cls_rel is not None:
            print('{0:>1.3f}|'.format(w_cls_rel[i]),end='')
        if (i+1) % 2 ==0:
            print('')
    print('')
    
    # compile json file
    relationship_json, objs_json, scans = read_relationship_json(mconfig, data, selected_scans)
    print('num of data:',len(scans))
    assert(len(scans)>0)
    
    dim_pts = 3
    if use_rgb:
        dim_pts += 3
    if use_normal:
        dim_pts += 3
    
    for scan_id in tqdm(scans):
        scan_id_no_split = scan_id.rsplit('_',1)[0]
        split_id = scan_id.rsplit('_',1)[1]
        map_instance2labelName = objs_json[scan_id]
        scan_path = os.path.join(root_3rscan, scan_id_no_split)
        data = load_mesh(scan_path, mconfig["label_file"], use_rgb, use_normal)
        
        points = data['points']
        instances = data['instances']
        
        rel_json = relationship_json[scan_id] # List[ Triplet[ Subject Id, Object Id, Predicate Id ] ]
        multi_view_path = f"{scan_path}/multi_view"
        retrieval_rgb_img(
            points, instances, split_id,
            scan_id_no_split, multi_view_path, 
            map_instance2labelName, mconfig["num_points"], 
            all_edge=True, rel_json=rel_json
        )
    
    
## Multi-processing은 깔끔하게 포기
## scan에 대해서 split별 분할 저장되어 있는데, 이러면 synchronization을 해줘야함
## 그거 하는데 쓰는 비용을 넣을 바에는 차라리 걍 single로 굴린다. 더럽고 치사해서 진짜
#     save_args = []
#     for scan_id in scans:
#         scan_id_no_split = scan_id.rsplit('_',1)[0]
#         map_instance2labelName = objs_json[scan_id]
#         scan_path = os.path.join(root_3rscan, scan_id_no_split)
#         rel_json = relationship_json[scan_id] # List[ Triplet[ Subject Id, Object Id, Predicate Id ] ]
#         save_args.append((
#             scan_id,
#             scan_path,
#             map_instance2labelName,
#             rel_json,
#             mconfig,
#             use_rgb, use_normal
#         ))
    
#     with Pool(processes=8) as pool:
#         results = list(tqdm(pool.imap(save_scan, save_args), total=len(save_args)))

# def save_scan(arg):
#     scan_id, scan_path, map_instance2labelName, rel_json, mconfig, use_rgb, use_normal = arg
#     scan_id_no_split = scan_id.rsplit('_', 1)[0]
#     data = load_mesh(scan_path, mconfig["label_file"], use_rgb, use_normal)
#     points = data['points']
#     instances = data['instances']
    
#     multi_view_path = f"{scan_path}/multi_view"
#     retrieval_rgb_img(
#         points, instances, 
#         scan_id_no_split, multi_view_path, 
#         map_instance2labelName, mconfig["num_points"], 
#         all_edge=True, rel_json=rel_json
#     )
#     return True