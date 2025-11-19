import torch
import numpy as np
import math
import os,json

def set_random_seed(seed):
    import random,torch
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def check_file_exist(path):
    if not os.path.exists(path):
            raise RuntimeError('Cannot open file. (',path,')')

def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

def to_gpu(*tensors):
    device = "cuda"
    c_tensor = [ t.to(device) for t in tensors ]
    return c_tensor


def read_classes(read_file):
    obj_classes = [] 
    with open(read_file, 'r') as f: 
        for line in f: 
            obj_class = line.rstrip().lower() 
            obj_classes.append(obj_class) 
    return obj_classes 


def read_relationships(read_file):
    relationships = [] 
    with open(read_file, 'r') as f: 
        for line in f: 
            relationship = line.rstrip().lower() 
            relationships.append(relationship) 
    return relationships 



def load_semseg(json_file, name_mapping_dict=None, mapping = True):    
    '''
    Create a dict that maps instance id to label name.
    If name_mapping_dict is given, the label name will be mapped to a corresponding name.
    If there is no such a key exist in name_mapping_dict, the label name will be set to '-'

    Parameters
    ----------
    json_file : str
        The path to semseg.json file
    name_mapping_dict : dict, optional
        Map label name to its corresponding name. The default is None.
    mapping : bool, optional
        Use name_mapping_dict as name_mapping or name filtering.
        if false, the query name not in the name_mapping_dict will be set to '-'
    Returns
    -------
    instance2labelName : dict
        Map instance id to label name.

    '''
    instance2labelName = {}
    with open(json_file, "r") as read_file:
        data = json.load(read_file)
        for segGroups in data['segGroups']:
            # print('id:',segGroups["id"],'label', segGroups["label"])
            # if segGroups["label"] == "remove":continue
            labelName = segGroups["label"]
            if name_mapping_dict is not None:
                if mapping:
                    if not labelName in name_mapping_dict:
                        labelName = 'none'
                    else:
                        labelName = name_mapping_dict[labelName]
                else:
                    if not labelName in name_mapping_dict.values():
                        labelName = 'none'

            instance2labelName[segGroups["id"]] = labelName.lower()#segGroups["label"].lower()
    return instance2labelName


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
    
    
def gen_descriptor(pts:torch.tensor):
    '''
    centroid_pts,std_pts,segment_dims,segment_volume,segment_lengths
    [3, 3, 3, 1, 1]
    '''
    assert pts.ndim==2
    assert pts.shape[-1]==3
    # centroid [n, 3]
    centroid_pts = pts.mean(0) 
    # # std [n, 3]
    std_pts = pts.std(0)
    # dimensions [n, 3]
    segment_dims = pts.max(dim=0)[0] - pts.min(dim=0)[0]
    # volume [n, 1]
    segment_volume = (segment_dims[0]*segment_dims[1]*segment_dims[2]).unsqueeze(0)
    # length [n, 1]
    segment_lengths = segment_dims.max().unsqueeze(0)
    return torch.cat([centroid_pts,std_pts,segment_dims,segment_volume,segment_lengths],dim=0)
