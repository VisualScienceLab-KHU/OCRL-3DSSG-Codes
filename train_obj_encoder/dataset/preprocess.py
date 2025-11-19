if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
import numpy as np
import os
import argparse
import json
from utils import util
from utils import util, util_ply
import trimesh
import torch
import math

def Parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../data/example_data', help="rio path")
    parser.add_argument('--type', type=str, default='train', choices=['train', 'test', 'validation'], help="allow multiple rel pred outputs per pair",required=False)
    parser.add_argument('--txt', type=str, default='"../data/train_scans.txt"', help="path to the txt file contain scan ids",required=False)
    return parser

def compute_weights(labels, classes, count, verbose=False):
    if verbose: print("-------------")    
    sum_weights = 0
    for c in range(len(classes)):
        if classes[c] / count > 0:        
            sum_weights += count / classes[c]

    sum_weight_norm = 0
    weights=list()
    for c in range(len(classes)):
        if classes[c] / count > 0:
            weight = count / classes[c] / sum_weights
            sum_weight_norm += weight
            if verbose: print('{0:>20s} {1:>1.3f} {2:>6d}'.format(labels[c], weight, int(classes[c])))
            weights.append(weight)
        else:
            if verbose: print('{0:>20s} {1:>1.3f} {2:>6d}'.format(labels[c], 0.0, int(classes[c])))
            weights.append(0)
    if verbose: print("-------------")
    return weights

def compute(classNames,relationNames, relationship_data, selections:list = None, verbose=False):
    o_rel_cls = np.zeros((len(relationNames)))
    o_obj_cls = np.zeros((len(classNames)))
    classes_count = 0
    counter = 0
    
    exceed_ids = dict()
    scene_analysis = dict()
    cnn=0
    for scan in relationship_data['scans']:
        scan_id = scan["scan"]
        if selections is not None:
            if scan_id not in selections:
                continue
        instance2LabelName = {}
        
        for k, v in scan["objects"].items():
            instance2LabelName[int(k)] = v
            if v not in classNames:
                if verbose: print(v,'not in classNames')
            o_obj_cls[classNames.index(v)] += 1

        nnk=dict()
        for relationship in scan["relationships"]:
            if relationship[3] not in relationNames:
                if verbose: print(relationship[3],'not in relationNames')
                continue

            obj = relationship[0] # id object
            sub = relationship[1] # id subject
            rel = relationship[2] # id relationship
            
            if obj == 0 or sub == 0:
                raise RuntimeError('found obj or sub is 0')
            
            if not obj in instance2LabelName:
                RuntimeWarning('key not found:',obj)
                continue

            if not sub in instance2LabelName:
                RuntimeWarning('key not found:',sub)
                continue
            
            if relationNames.index(relationship[3]) >= len(relationNames): 
                if rel not in exceed_ids:
                    exceed_ids[relationNames.index(relationship[3])]=0
                else:
                    exceed_ids[relationNames.index(relationship[3])]+=1
                continue
            o_rel_cls[relationNames.index(relationship[3])] += 1
            classes_count += 1
            
            nn = str(obj)+'_'+str(sub)
            if nn not in nnk:
                nnk[nn] = 0
            nnk[str(obj)+'_'+str(sub)] +=1
        for v in nnk.values():
            if v > 1:
                cnn+=1
                
        counter += 1
        
        scene_analysis[scan_id] = dict()
        scene_analysis[scan_id]['num objects'] = len(scan["objects"].items())
        scene_analysis[scan_id]['num relationships'] = len(scan['relationships'])
    if verbose: print('num multi predicates:',cnn)
        
    if len(exceed_ids)>1:
        print('exceed map')
        for id, count in exceed_ids.items():
            print('\t',id,count)

    if verbose: print("objects:")
    wobjs = compute_weights(classNames, o_obj_cls, classes_count,verbose)
    if verbose: print("relationships:")
    wrels = compute_weights(relationNames, o_rel_cls, classes_count,verbose)
    return wobjs,wrels,o_obj_cls,o_rel_cls

def read_relationships_json(args):
    catfile = os.path.join(args.root, 'classes.txt')
    classNames = util.read_classes(catfile)
    relationNames = util.read_relationships(os.path.join(args.root,'relationships.txt'))
    pth_relationships_json = os.path.join(os.path.join(args.root,'relationships_'+str(args.type)+'.json'))

    with open(pth_relationships_json, "r") as read_file:
        data = json.load(read_file)

    wobjs, wrels, o_obj_cls, o_rel_cls = compute(classNames, relationNames, data, verbose=True)

def dataset_loading_3RScan(root:str, pth_selection:str,split:str,class_choice:list=None):  
    # read object class
    pth_catfile = os.path.join(pth_selection, 'classes.txt')
    classNames = util.read_txt_to_list(pth_catfile)
    # read relationship class
    pth_relationship = os.path.join(pth_selection, 'relationships.txt')
    util.check_file_exist(pth_relationship)
    relationNames = util.read_relationships(pth_relationship)
    # read relationship json
    selected_scans=set()
    if split == 'train_scans' :
        selected_scans = selected_scans.union(util.read_txt_to_list(os.path.join(pth_selection,'train_scans.txt')))
        with open(os.path.join(root, 'relationships_train.json'), "r") as read_file:
            data = json.load(read_file)
    elif split == 'validation_scans':
        selected_scans = selected_scans.union(util.read_txt_to_list(os.path.join(pth_selection,'validation_scans.txt')))
        with open(os.path.join(root, 'relationships_validation.json'), "r") as read_file:
            data = json.load(read_file)
    else:
        raise RuntimeError('unknown split type:',split)
    return  classNames, relationNames, data, selected_scans
                        
def load_mesh(path,label_file,use_rgb,use_normal):
    result=dict()
    if label_file == 'labels.instances.align.annotated.v2.ply' or label_file == 'labels.instances.align.annotated.ply':
        
        plydata = trimesh.load(os.path.join(path,label_file), process=False)
        points = np.array(plydata.vertices)
        instances = util_ply.read_labels(plydata).flatten()
        
        if use_rgb:
            rgbs = np.array(plydata.visual.vertex_colors.tolist())[:,:3]
            points = np.concatenate((points, rgbs / 255.0), axis=1)
            
        if use_normal:
            normal = plydata.vertex_normals[:,:3]
            points = np.concatenate((points, normal), axis=1)
        
        result['points']=points
        result['instances']=instances
    else:
        raise NotImplementedError('')
    return result

def print_class_info(classNames, relationNames, data, selected_scans, multi_rel_outputs):
    wobjs, wrels, o_obj_cls, o_rel_cls = compute(classNames, relationNames, data,selected_scans, False)
    w_cls_obj = torch.from_numpy(np.array(o_obj_cls)).float().to("cuda")
    w_cls_rel = torch.from_numpy(np.array(o_rel_cls)).float().to("cuda")
    
    # for single relation output, we set 'None' relationship weight as 1e-3
    if not multi_rel_outputs:
        w_cls_rel[0] = w_cls_rel.max() * 10
    
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

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                      [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                      [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def main():
    args = Parser().parse_args()
    read_relationships_json(args)


if __name__ == "__main__": main()
