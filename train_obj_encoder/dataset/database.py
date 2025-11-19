import numpy as np
import torch
import torch.utils.data as data

import PIL.Image as Image
from dataset.preprocess import rotation_matrix
from config import dataset_config, config_system, PREPROCESS_PATH
from utils.util import read_txt_to_list
import clip
import h5py
from glob import glob
from tqdm import tqdm
import pickle

class SSGCMFeatDataset(data.Dataset):
    def __init__(self, 
            split,
            use_rgb,
            use_normal,
            device
        ):
        super(SSGCMFeatDataset, self).__init__()
        assert split in ["train_scans", "validation_scans"], "Not the valid split"
        self.config = config_system
        self.mconfig = dataset_config
        self.root = self.mconfig["root"]
        self.split = split
        self.use_rgb = use_rgb
        self.use_normal = use_normal
        self.obj_file_list = PREPROCESS_PATH if split == "train_scans" else PREPROCESS_PATH + "_val"
        self.mode = "train" if split == "train_scans" else "validation"
        self.n_pts = self.mconfig["num_points_union"]
        self.top_k_num = self.mconfig["top_k_img_num"]
        self.device = device
        _, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.obj_cls_list = read_txt_to_list(f"{dataset_config['root']}/classes.txt")
        # Read directly from h5py
        # self.obj_h5_list = glob(f"{self.obj_file_list}/*-*-*-*-*/*/*.h5")
        # self.__remove_small()
        self.obj_bin_list = glob(f"{PREPROCESS_PATH}/final_objs_{self.mode}/*.pkl")
        
        print("Getting h5py object files... please wait...")
        self.obj_data_list = []
        for _p in tqdm(self.obj_bin_list, total=len(self.obj_bin_list)):
            self.obj_data_list.extend(self.__read_compressed_bin_file(_p))
        
        # All activated on training
        self.dim_pts = 3
        if self.use_rgb:
            self.dim_pts += 3
        if self.use_normal:
            self.dim_pts += 3
    
    def to(self, device):
        self.device = device
    
    ## Random sampling for uniform tensor shape
    ## Input points: N_r X 9, N_r: random size
    ## Ouptput points: N_f X 9, N_f: fixed size 
    def __random_sample(self, points: torch.Tensor):
        N_r = points.shape[0]
        selection_mask = torch.randperm(N_r)[:self.n_pts]
        return points[selection_mask, :]
    
    def __data_augmentation(self, points):
        # random rotate
        matrix= np.eye(3)
        matrix[0:3,0:3] = rotation_matrix([0, 0, 1], np.random.uniform(0, 2*np.pi, 1))
        matrix = torch.from_numpy(matrix).float()
        
        centroid = points[:, :3].mean(0)
        points[:, :3] -= centroid
        points[:, :3] = points[:, :3] @ matrix.T
        if self.use_normal:
            ofset = 3
            if self.use_rgb:
                ofset += 3
            points[:, ofset: 3 + ofset] = points[:, ofset: 3 + ofset] @ matrix.T
        return points
    
    def __len__(self):
        return len(self.obj_data_list)
    
    def __to_torch(self, x):
        return torch.from_numpy(np.array(x, dtype=np.float32))
    
    def __read_compressed_bin_file(self, _path: str):
        obj_data = []
        with open(_path, "rb") as f:
            obj = pickle.load(f)
            for con in obj:
                _data = {}
                _data["obj_point"] = self.__to_torch(con["obj_point"])
                if self.mode == "train":
                    _data["mv_rgb"] = torch.stack([ self.preprocess(img) for img in con["mv_rgb"][:self.top_k_num] ], dim=0)
                else :
                    _data["mv_rgb"] = []
                _data["instance_id"] = con["instance_id"]
                _data["instance_name"] = con["instance_name"]
                obj_data.append(_data)
        return obj_data
    
    def __read_compressed_h5_file(self, _path: str):
        _data = {}
        with h5py.File(_path, "r") as f:
            pts = self.__to_torch(f["obj_point"])
            _data["obj_point"] = self.__random_sample(pts)
            _data["mv_rgb"] = []
            if self.split == "train_scans":
                rgb = Image.fromarray(np.array(f["rgb_view_0"], dtype=np.uint8)).transpose(Image.ROTATE_270)
                _data["mv_rgb"].append(self.preprocess(rgb))
            # For now, Single image is used
            # rgb_keys = [ x for x in list(f.keys()) if x.startswith("rgb_view") ]
            # for k in rgb_keys:
            #     rgb = Image.fromarray(np.array(f[k], dtype=np.uint8)).transpose(Image.ROTATE_270)
            #     _data["mv_rgb"].append(self.preprocess(rgb))
            _data["instance_id"] = f.attrs["semantic_id"]
            _data["instance_name"] = f.attrs["semantic_name"]
        return _data
    
    ## There exist small object point cloud with 256 points
    ## For now, we need to remove that shit
    def __remove_small(self):
        print("Removing small point clouds...")
        remove_candidate = []
        for idx, _p in tqdm(enumerate(self.obj_h5_list), total=len(self.obj_h5_list)):
            with h5py.File(_p, "r") as f:
                pts = np.array(f["obj_point"])
                n_p = pts.shape[0]
                if n_p < self.n_pts: remove_candidate.append(idx)
        print(
            "# of outlier of small point cloud:", 
            len(remove_candidate), 
            "of total", 
            len(self.obj_h5_list)
        )
        self.obj_h5_list = [ x for i, x in enumerate(self.obj_h5_list) if not i in remove_candidate ]
    
    def __getitem__(self, index):
        """
        Instance Point Cloud를 sampling하는 방법에 대한 조사가 필요함.
        Simsiam 방법론으로 어떻게 sampling 및 augmentation을 수행하지?
        
        - 25/01/23: First Experiment, Single Image view
        - ??/??/??: TODO: Multi-View pair settings
        """
        obj_data = self.obj_data_list[index] # self.__read_compressed_file(obj_path)
        
        obj_pos_1 = self.__data_augmentation(obj_data["obj_point"])
        obj_pos_2 = self.__data_augmentation(obj_data["obj_point"])
        text_feature = clip.tokenize(f"A point cloud of a {obj_data['instance_name']}")
        text_id = self.obj_cls_list.index(obj_data['instance_name'])
        target_text_id = torch.zeros(len(self.obj_cls_list), dtype=torch.float32)
        target_text_id[text_id] = 1.
        
        if self.split == "train_scans":    
            ## Zero Masking for variable length Multi-View image
            N, C, H, W = obj_data["mv_rgb"].shape
            zero_mask = torch.ones((self.top_k_num), dtype=torch.float32)
            if N < self.top_k_num:
                _d = self.top_k_num - N
                zero_img = torch.zeros((_d, C, H, W), dtype=torch.float32)
                obj_data["mv_rgb"] = torch.cat([ obj_data["mv_rgb"], zero_img ], dim=0)
                zero_mask[N:] = 0.
            return obj_pos_1, obj_pos_2, obj_data["mv_rgb"], zero_mask, text_feature, target_text_id
        else:
            return obj_data["obj_point"], obj_data['instance_name'], target_text_id