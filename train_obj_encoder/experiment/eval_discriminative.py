from model.frontend.pointnet import PointNetEncoder
from model.frontend.dgcnn import DGCNN
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
    v_dataset = SSGCMFeatDataset(split="validation_scans", use_rgb=True, use_normal=True, device=device)
    v_loader = DataLoader(v_dataset, batch_size=256, shuffle=True, drop_last=True)
    
    with torch.no_grad():
        feat_per_labels = {}
        for i, (data, text_feat, label) in enumerate(v_loader):
            data = data.to(device)
            data = data.transpose(2, 1).contiguous()
            z, _, _ = encoder_model(data)
            
            print(f"Processing {i}-th batch...")
            for idx, text in tqdm(enumerate(text_feat)):
                if not text in feat_per_labels.keys():
                    feat_per_labels[text] = [ z[idx].unsqueeze(0).cpu().numpy() ]
                else:
                    feat_per_labels[text].append(z[idx].unsqueeze(0).cpu().numpy())
        
        for k in feat_per_labels.keys():
            feat_per_labels[k] = np.concatenate(feat_per_labels[k], axis=0)
    
    if not os.path.exists("./experiment/results"):
        os.makedirs("./experiment/results")
    
    with open(f"./experiment/results/{exp_name}.pkl", "wb") as f:
        pickle.dump(feat_per_labels, f)