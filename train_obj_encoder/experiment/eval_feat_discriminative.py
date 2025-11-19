from model.frontend.pointnet import PointNetEncoder
from config import dataset_config, config_system, PREPROCESS_PATH
from utils.util import read_txt_to_list
from dataset.database import SSGCMFeatDataset
from torch.utils.data import DataLoader
import argparse
import torch
from tqdm import tqdm
import pickle
import os
import numpy as np
import parser
import clip

parser = argparse.ArgumentParser(description="Evaluation for feature space discriminative")
parser.add_argument("--exp_dir", type=str, required=True)
args = parser.parse_args()

if __name__ == "__main__":
    
    model_path = args.exp_dir
    exp_name = model_path.split("/")[1] + "_" + model_path.split("/")[3].split(".")[0]
    device = "cuda"
    encoder_model = PointNetEncoder(device, channel=9).to(device)
    encoder_model.load_state_dict(torch.load(model_path))
    encoder_model = encoder_model.eval()
    t_dataset = SSGCMFeatDataset(split="train_scans", use_rgb=True, use_normal=True, device=device)
    t_loader = DataLoader(t_dataset, batch_size=256, shuffle=True, drop_last=True)
    
    if not os.path.exists(f"<Your Path>/bfeat_object_experiments/{exp_name}"):
        os.makedirs(f"<Your Path>/bfeat_object_experiments/{exp_name}")
    obj_cls_list = read_txt_to_list(f"{dataset_config['root']}/classes.txt")
    
    with torch.no_grad():
        for i, (data_t1, data_t2, obj_data, zero_mask, text_feat, label) in enumerate(t_loader):
            feat_per_labels = {}
            data_t1, data_t2 = \
                    data_t1.to(device), data_t2.to(device)
            batch_size = data_t1.size()[0]
            
            data = torch.cat((data_t1, data_t2))
            data = data.transpose(2, 1).contiguous()
            point_feats, _, _ = encoder_model(data)
            
            point_t1_feats = point_feats[:batch_size, :]
            point_t2_feats = point_feats[batch_size:, :]
            z = torch.stack([point_t1_feats, point_t2_feats]).mean(dim=0)
            
            print(f"Processing {i}-th batch...")
            for idx, text in tqdm(enumerate(label)):
                label_idx = text.argmax(dim=-1).item()
                text_labels = obj_cls_list[label_idx]
                if not text in feat_per_labels.keys():
                    feat_per_labels[text_labels] = [ z[idx].unsqueeze(0).cpu().numpy() ]
                else:
                    feat_per_labels[text_labels].append(z[idx].unsqueeze(0).cpu().numpy())
        
            for k in feat_per_labels.keys():
                feat_per_labels[k] = np.concatenate(feat_per_labels[k], axis=0)

            with open(f"<Your Path>/bfeat_object_experiments/{exp_name}/batch_{i}.pkl", "wb") as f:
                pickle.dump(feat_per_labels, f)