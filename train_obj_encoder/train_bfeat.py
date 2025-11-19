import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from dataset.database import SSGCMFeatDataset
from model.frontend.pointnet import PointNetEncoder
from lightly.loss.ntx_ent_loss import NTXentLoss
from model.frontend.dgcnn import DGCNN
from config import config_system
import numpy as np
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
from datetime import datetime
import torch.optim as optim
from utils.logger import AverageMeter, IOStream

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import clip

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

def train(rank, world_size):
    if rank == 0:
        wandb.init(project="<Your Project>", name=exp_name)
        io = IOStream('checkpoints/' + exp_name + '/run.log')
    
    dist.init_process_group(
        "nccl", 
        init_method="tcp://127.0.0.1:55136",
        rank=rank, 
        world_size=world_size
    )
    torch.manual_seed(config_system["SEED"])
    ## Training Parameter config
    bsz = config_system["Batch_Size"]
    lr = config_system["LR"]
    epoches = config_system["MAX_EPOCHES"]
    save_interval = config_system["SAVE_INTERVAL"]
    log_interval = config_system["LOG_INTERVAL"]
    
    
    t_dataset = SSGCMFeatDataset(split="train_scans", use_rgb=True, use_normal=True, device=rank)
    sampler = DistributedSampler(t_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_loader = DataLoader(t_dataset, batch_size=bsz, sampler=sampler)
    
    # v_dataset = SSGCMFeatDataset(split="validation_scans", use_rgb=True, use_normal=True, device=rank)
    # v_loader = DataLoader(v_dataset, batch_size=bsz, shuffle=True, drop_last=True)
    
    model, _ = clip.load("ViT-B/32", device=rank)
    model = model.eval()
    
    encoder_model = None
    if args.encoder == "pointnet":
        encoder_model = PointNetEncoder(rank, channel=9).to(rank)
    elif args.encoder == "dgcnn":
        encoder_model = DGCNN({
            "k": 10,
            "emb_dims": 512
        }).to(rank)
    if args.resume:
        encoder_model.load_state_dict(torch.load(args.resume))
    
    ddp_model = DDP(encoder_model, device_ids=[rank]) 
    optimizer = optim.Adam(encoder_model.parameters(), lr=lr, weight_decay=1e-6)
    
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epoches, eta_min=0, last_epoch=-1)
    criterion = NTXentLoss(temperature = 0.07).to(rank)
    
    best_loss = 987654321
    for epoch in range(epoches):
        lr_scheduler.step()
        ddp_model.train()
            
        ####################
        # Train
        ####################
        train_losses = AverageMeter(name="Train Total Loss")
        train_imid_losses = AverageMeter(name="Train Intra-Modal Loss")
        train_cmid_losses = AverageMeter(name="Train Cross-Modal Loss")
        wandb_log = {}
        print(f'Start training epoch: ({epoch}/{epoches})')
        for i, (data_t1, data_t2, rgb_img, text_feat, _) in enumerate(train_loader):
            data_t1, data_t2, rgb_img, text_feat = \
                data_t1.to(rank), data_t2.to(rank), rgb_img.to(rank).float(), text_feat.to(rank)
            batch_size = data_t1.size()[0]
            
            with torch.no_grad():
                rgb_feat = model.encode_image(rgb_img).to(rank).float()
                text_feat = model.encode_text(text_feat.squeeze(1)).to(rank).float()
            
            optimizer.zero_grad()
            data = torch.cat((data_t1, data_t2))
            data = data.transpose(2, 1).contiguous()
            point_feats, _, _ = ddp_model(data)
            
            point_t1_feats = point_feats[:batch_size, :]
            point_t2_feats = point_feats[batch_size:, :]
            
            loss_imid = criterion(point_t1_feats, point_t2_feats)        
            point_feats = torch.stack([point_t1_feats, point_t2_feats]).mean(dim=0)
            loss_cmid = criterion(point_feats, rgb_feat.squeeze(1)) + criterion(point_feats, text_feat)
            
            total_loss = loss_imid + loss_cmid
            total_loss.backward()
            clip_grad_norm_(encoder_model.parameters(), 0.5)
            optimizer.step()
            
            train_losses.update(total_loss.item(), batch_size)
            train_imid_losses.update(loss_imid.item(), batch_size)
            train_cmid_losses.update(loss_cmid.item(), batch_size)
            
            if epoch % log_interval == 0:
                print('Epoch (%d), Batch(%d/%d), loss: %.6f, imid loss: %.6f, cmid loss: %.6f ' % (
                    epoch, i, 
                    len(train_loader), train_losses.avg, 
                    train_imid_losses.avg, train_cmid_losses.avg
                ))
                wandb_log['Train Loss'] = train_losses.avg
                wandb_log['Train IMID Loss'] = train_imid_losses.avg
                wandb_log['Train CMID Loss'] = train_cmid_losses.avg
                
        if rank == 0:
            outstr = 'Train %d, loss: %.6f' % (epoch, train_losses.avg)
            io.cprint(outstr)  
            
            if train_losses.avg < best_loss:
                best_loss = train_losses.avg
                print('==> Saving Best Model...')
                save_file = os.path.join(f'checkpoints/{exp_name}/models/', 'best_model.pth'.format(epoch=epoch))
                torch.save(encoder_model.state_dict(), save_file)
    
            if epoch % save_interval == 0:
                print('==> Saving...')
                save_file = os.path.join(f'checkpoints/{exp_name}/models/', 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                torch.save(encoder_model.state_dict(), save_file)
            
            wandb.log(wandb_log)
    dist.destroy_process_group()


if __name__ == "__main__":
    __setup()
    world_size = torch.cuda.device_count()  # 사용 가능한 GPU 개수
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    