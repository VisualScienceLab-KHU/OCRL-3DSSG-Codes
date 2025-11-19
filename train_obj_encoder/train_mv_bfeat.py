import os
import argparse
import clip.model
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from dataset.database import SSGCMFeatDataset
from model.frontend.pointnet import PointNetEncoder
from model.loss import IntraModalBarlowTwinLoss, SupervisedCrossModalInfoNCE, CrossModalInfoNCE
from lightly.loss.ntx_ent_loss import NTXentLoss
from model.frontend.dgcnn import DGCNN
from config import config_system, dataset_config
from einops import rearrange
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
from datetime import datetime
import torch.optim as optim
from utils.logger import AverageMeter, Progbar
from utils.util import read_txt_to_list, to_gpu
from utils.eval_utils import consine_classification
import clip
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
import numpy as np

parser = argparse.ArgumentParser(description="Example script for argparse")
parser.add_argument("--encoder", "-e", type=str, default="pointnet", choices=["pointnet", "dgcnn"])
parser.add_argument("--exp_name", type=str, default="better_feature")
parser.add_argument("--resume", "-r", type=str)
args = parser.parse_args()

now = datetime.now()
exp_name = f"{args.exp_name}_{args.encoder}_{now.strftime('%Y-%m-%d_%H')}"

def __setup():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + exp_name):
        os.makedirs('checkpoints/' + exp_name)
    if not os.path.exists('checkpoints/' + exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + exp_name + '/' + 'models')

def build_classifier(clip_model: clip.model, device):
    clip_model = clip_model.eval()
    obj_cls = read_txt_to_list(f"{dataset_config['root']}/classes.txt")
    obj_tokens = torch.cat([ clip.tokenize(f"A point cloud of a {obj}") for obj in obj_cls ], dim=0).to(device)
    text_gt_matrix = clip_model.encode_text(obj_tokens)
    return text_gt_matrix.float()

def validation(validation_loader:DataLoader, model: torch.nn.Module, text_cls_matrix: torch.Tensor):
    n_iters = len(validation_loader)
    progbar = Progbar(n_iters, width=40, stateful_metrics=['Misc/epo', 'Misc/it'])
    topk_obj_list = np.array([])
    model.eval()
    with torch.no_grad():
        for i, (data_obj, _, gt_label) in enumerate(validation_loader):
            data_obj, gt_label = to_gpu(data_obj, gt_label)
            data = data_obj.transpose(2, 1).contiguous()
            point_feats, _, _ = model(data)
            obj_topk_list = consine_classification(text_cls_matrix, point_feats, gt_label)
            topk_obj_list = np.concatenate((topk_obj_list, obj_topk_list))
            
            eval_batch = [ 100 * (topk_obj_list <= i).sum() / len(topk_obj_list) for i in [1, 5, 10] ]
            logs = [
                ("validation/Obj_R1", eval_batch[0]),
                ("validation/Obj_R5", eval_batch[1]),
                ("validation/Obj_R10", eval_batch[2]),
            ]
            progbar.add(1, values=logs)
    obj_acc = [ 100 * (topk_obj_list <= i).sum() / len(topk_obj_list) for i in [1, 5, 10] ]
    return obj_acc
    
def train(rank): # , world_size
    # if rank == 0:
    wandb.init(project="<Your Project>", name=exp_name)
    
    # dist.init_process_group(
    #     "nccl", 
    #     init_method="tcp://127.0.0.1:55136",
    #     rank=rank, 
    #     world_size=world_size
    # )
    torch.manual_seed(config_system["SEED"])
    ## Training Parameter config
    device = rank
    bsz = config_system["Batch_Size"]
    lr = config_system["LR"]
    epoches = config_system["MAX_EPOCHES"]
    save_interval = config_system["SAVE_INTERVAL"]
    log_interval = config_system["LOG_INTERVAL"]
    evaluate_interval = config_system["VALID_INTERVAL"]

    t_dataset = SSGCMFeatDataset(split="train_scans", use_rgb=True, use_normal=True, device=rank)
    # sampler = DistributedSampler(t_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_loader = DataLoader(t_dataset, batch_size=bsz, shuffle=True, drop_last=True)
    v_dataset = SSGCMFeatDataset(split="validation_scans", use_rgb=True, use_normal=True, device=device)
    validation_loader = DataLoader(v_dataset, batch_size=bsz, shuffle=True, drop_last=True)
    
    model, _ = clip.load("ViT-B/32", device=rank)
    model = model.eval()
    with torch.no_grad():
        text_cls_matrix = build_classifier(model, device)
    encoder_model = None
    if args.encoder == "pointnet":
        encoder_model = PointNetEncoder(device, channel=9).to(device)
    elif args.encoder == "dgcnn":
        encoder_model = DGCNN({
            "k": 10,
            "emb_dims": 512
        }).to(device)
    else:
        raise NotImplementedError
    if args.resume:
        encoder_model.load_state_dict(torch.load(args.resume))
        
    # ddp_model = DDP(encoder_model, device_ids=[rank]) 
    optimizer = optim.Adam(encoder_model.parameters(), lr=lr, weight_decay=1e-6)
    
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epoches, eta_min=0, last_epoch=-1)
    # CyclicLR( # 
    #     optimizer, base_lr=lr / 10, 
    #     step_size_up=epoches, max_lr=lr * 5, 
    #     gamma=0.8, mode='exp_range', cycle_momentum=False
    # )
    intra_criterion = IntraModalBarlowTwinLoss()
    cm_visual_criterion = SupervisedCrossModalInfoNCE(device, temperature=0.07) # magnify temperature to 0.2
    cm_text_criterion = SupervisedCrossModalInfoNCE(device, temperature=0.07) 
    # NTXentLoss(temperature = 0.07).to(device) # For NX-Tent loss, calculate self-negativity.
    n_iters = len(train_loader)
    
    best_val = -987654321
    for epoch in range(epoches):
        encoder_model.train()
        
        ####################
        # Train
        ####################
        progbar = Progbar(n_iters, width=40, stateful_metrics=['Misc/epo', 'Misc/it'])
        train_losses = AverageMeter(name="Train Total Loss")
        train_imid_losses = AverageMeter(name="Train Intra-Modal Loss")
        train_cm_visual_losses = AverageMeter(name="Train 3D-Visual Cross-Modal Loss")
        train_cm_text_losses = AverageMeter(name="Train 3D-Text Cross-Modal Loss")
        wandb_log = {}
        topk_obj_list = np.array([])
        for i, (data_t1, data_t2, rgb_imgs, zero_mask, text_feat, gt_label) in enumerate(train_loader):
            data_t1, data_t2, rgb_imgs, zero_mask, text_feat, gt_label = \
                data_t1.to(rank), \
                data_t2.to(rank), \
                rgb_imgs.to(rank), \
                zero_mask.to(rank), \
                text_feat.to(rank), \
                gt_label.to(rank)
            batch_size = data_t1.size()[0]
            
            rgb_feat_list = []
            with torch.no_grad():
                _, M, _, _, _ = rgb_imgs.shape
                for m in range(M):
                    rgb_feat = model.encode_image(rgb_imgs[:, m, ...]).unsqueeze(1).to(device).float()
                    rgb_feat_list.append(rgb_feat)
                text_feat = model.encode_text(text_feat.squeeze(1)).to(device).float()
                rgb_feats = torch.cat(rgb_feat_list, dim=1)
            
            optimizer.zero_grad()
            data = torch.cat((data_t1, data_t2))
            data = data.transpose(2, 1).contiguous()
            point_feats, _, _ = encoder_model(data)
            
            point_t1_feats = point_feats[:batch_size, :]
            point_t2_feats = point_feats[batch_size:, :]
            
            loss_imbt = intra_criterion(point_t1_feats, point_t2_feats)        
            point_feats = torch.stack([point_t1_feats, point_t2_feats]).mean(dim=0)
            loss_cm_visual = cm_visual_criterion(point_feats, rgb_feats, gt_label, zero_mask)
            loss_cm_text = cm_text_criterion(point_feats, text_feat, gt_label)
            # gt_label must be added
            
            total_loss = 0.1 * loss_imbt + loss_cm_visual + loss_cm_text
            total_loss.backward()
            # clip_grad_norm_(encoder_model.parameters(), 0.5)
            optimizer.step()
            
            train_losses.update(total_loss.detach().cpu().item(), batch_size)
            train_imid_losses.update(loss_imbt.detach().cpu().item(), batch_size)
            train_cm_visual_losses.update(loss_cm_visual.detach().cpu().item(), batch_size)
            train_cm_text_losses.update(loss_cm_text.detach().cpu().item(), batch_size)
            
            # if rank == 0:
            t_log = [
                ("train/total_loss", total_loss.detach().cpu().item()),
                ("train/imid_loss", loss_imbt.detach().cpu().item()),
                ("train/cm_visual_loss", loss_cm_visual.detach().cpu().item()),
                ("train/cm_text_loss", loss_cm_text.detach().cpu().item()),
                ("Misc/epo", int(epoch)),
                ("Misc/it", int(i)),
                ("lr", lr_scheduler.get_last_lr()[0])
            ]
            if epoch % log_interval == 0:
                obj_topk_list = consine_classification(text_cls_matrix.detach(), point_feats.detach(), gt_label)
                topk_obj_list = np.concatenate((topk_obj_list, obj_topk_list))
                eval_batch = [ 100 * (topk_obj_list <= i).sum() / len(topk_obj_list) for i in [1, 5, 10] ]
                wandb_log["Train/Obj_R@1"] = eval_batch[0]
                wandb_log["Train/Obj_R@5"] = eval_batch[1]
                wandb_log["Train/Obj_R@10"] = eval_batch[2]
                logs = [
                    ("train/Obj_R1", eval_batch[0]),
                    ("train/Obj_R5", eval_batch[1]),
                    ("train/Obj_R10", eval_batch[2]),
                ]
                t_log += logs
            progbar.add(1, values=t_log)
        
        lr_scheduler.step()
        # if rank == 0:
        if epoch % evaluate_interval == 0:
            obj_topk = validation(validation_loader, encoder_model, text_cls_matrix.detach())
            val_metric = sum(obj_topk) / 3.
            wandb_log["Validation/Obj_R@1"] = obj_topk[0]
            wandb_log["Validation/Obj_R@5"] = obj_topk[1]
            wandb_log["Validation/Obj_R@10"] = obj_topk[2]
            if val_metric > best_val:
                best_val = val_metric
                print('==> Saving Best Model...')
                save_file = os.path.join(f'checkpoints/{exp_name}/models/', 'best_model.pth'.format(epoch=epoch))
                torch.save(encoder_model.state_dict(), save_file)

        if epoch % save_interval == 0:
            print('==> Saving...')
            save_file = os.path.join(f'checkpoints/{exp_name}/models/', 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(encoder_model.state_dict(), save_file)
        
        wandb_log["Train/Learning_Rate"] = lr_scheduler.get_last_lr()[0]
        wandb_log['Train/Loss'] = train_losses.avg
        wandb_log['Train/IMBT_Loss'] = train_imid_losses.avg
        wandb_log['Train/CM_Visual_Loss'] = train_cm_visual_losses.avg
        wandb_log['Train/CM_Text_Loss'] = train_cm_text_losses.avg
        wandb.log(wandb_log)
    
    # dist.destroy_process_group()

if __name__ == "__main__":
    __setup()
    # world_size = torch.cuda.device_count()  # 사용 가능한 GPU 개수
    train(rank="cuda")
    # mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)