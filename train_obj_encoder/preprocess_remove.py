from config import dataset_config, config_system, PREPROCESS_PATH
import h5py
from glob import glob
from tqdm import tqdm
import numpy as np
import os
import PIL.Image as Image
import pickle

n_pts = dataset_config["num_points_union"]

def __remove_small(obj_h5_list):
    remove_candidate = []
    for idx, _p in tqdm(enumerate(obj_h5_list), total=len(obj_h5_list)):
        with h5py.File(_p, "r") as f:
            pts = np.array(f["obj_point"])
            n_p = pts.shape[0]
            if n_p < dataset_config["num_points_union"]: remove_candidate.append(idx)
    print(
        "# of outlier of small point cloud:", 
        len(remove_candidate), 
        "of total", 
        len(obj_h5_list)
    )
    obj_h5_list = [ x for i, x in enumerate(obj_h5_list) if not i in remove_candidate ]
    return obj_h5_list

def __random_sample(points: np.array):
    N_r = points.shape[0]
    choice = np.random.choice(N_r, n_pts, replace=True)
    obj_pointset = points[choice, :]
    return obj_pointset

if __name__ == "__main__":
    
    t_obj_h5_list = __remove_small(glob(f"{PREPROCESS_PATH}/*/*/*.h5"))
    v_obj_h5_list = __remove_small(glob(f"{PREPROCESS_PATH}_val/*/*/*.h5"))
    
    if not os.path.exists(f'{PREPROCESS_PATH}/final_objs_train'):
        os.makedirs(f'{PREPROCESS_PATH}/final_objs_train')
    if not os.path.exists(f'{PREPROCESS_PATH}/final_objs_validation'):
        os.makedirs(f'{PREPROCESS_PATH}/final_objs_validation')
    
    ## 저장 어떻게 할지 생각좀...
    print("## Accumulating training scans...")
    t_obj_data_list = []
    t_idx = 5
    for idx, _p in tqdm(enumerate(t_obj_h5_list[5000 * t_idx:]), total=len(t_obj_h5_list[5000 * t_idx:])):
        _data = {}
        with h5py.File(_p, "r") as f:
            pts = np.array(f["obj_point"])
            _data["obj_point"] = __random_sample(pts)
            _data["mv_rgb"] = []
            rgb = Image.fromarray(np.array(f["rgb_view_0"], dtype=np.uint8)).transpose(Image.ROTATE_270)
            _data["mv_rgb"].append(rgb)
            _data["instance_id"] = f.attrs["semantic_id"]
            _data["instance_name"] = f.attrs["semantic_name"]
        t_obj_data_list.append(_data)
        if len(t_obj_data_list) > 5000 or len(t_obj_h5_list[5000 * t_idx:]) == idx + 1:
            with open(f"{PREPROCESS_PATH}/final_objs_train/train_obj_pc_{t_idx}.pkl", "wb") as f:
                pickle.dump(t_obj_data_list, f)
            t_obj_data_list.clear()
            t_idx += 1
    print("## Training data saved")
    
    v_obj_data_list = []
    val_idx = 0
    print("## Accumulating validation scans...")
    for idx, _p in tqdm(enumerate(v_obj_h5_list), total=len(v_obj_h5_list)):
        _data = {}
        with h5py.File(_p, "r") as f:
            pts = np.array(f["obj_point"])
            _data["obj_point"] = __random_sample(pts)
            _data["mv_rgb"] = []
            _data["instance_id"] = f.attrs["semantic_id"]
            _data["instance_name"] = f.attrs["semantic_name"]
        v_obj_data_list.append(_data)
        if len(v_obj_data_list) > 5000 or len(v_obj_h5_list) == idx + 1:
            with open(f"{PREPROCESS_PATH}/final_objs_validation/val_obj_pc_{val_idx}.pkl", "wb") as f:
                pickle.dump(v_obj_data_list, f)
            v_obj_data_list.clear()
            val_idx += 1