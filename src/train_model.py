"""
# Ensore inside MSNet folder
% pwd
/opt/files/maio2022/SAT/NSNet

# Run (this) script -> specify output folder
% python src/train_model.py sat-solving NSNet /opt/files/maio2022/SAT/NSNet/SATSolving/SATLIB --model NSNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import argparse
import numpy as np
import random
import math

from tqdm import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

# pytorch-scatter ported to pytorch main
# https://github.com/rusty1s/pytorch_scatter/issues/241#issuecomment-1336116049
# from torch_scatter import scatter_sum
# from torch.scatter_reduce import scatter_sum
from utils.scatter import scatter_sum

from utils.options import add_model_options
from utils.logger import Logger
from utils.dataloader import get_dataloader
from utils.utils import safe_log
from models.nsnet import NSNet
from models.neurosat import NeuroSAT

from utils.options import ArgOpts


def main(opts: ArgOpts = None):
    opts = opts or ArgOpts()            # empty -> uses defaults passed below
    parser = argparse.ArgumentParser()
    # parser.add_argument('task', type=str, choices=['model-counting', 'sat-solving'], help='Experiment task')
    # parser.add_argument('exp_id', type=str, help='Experiment id')
    # parser.add_argument('train_dir', type=str, help='Directory with training data')
    parser.add_argument(
        '--task', type=str, choices=['model-counting', 'sat-solving'], help='Experiment task',
        default=opts.get("task", "sat-solving"))
    parser.add_argument(
        '--exp_id', type=str, help='Experiment id',
        default=opts.get("exp_id", "NSNet"))
    parser.add_argument(
        '--train_dir', type=str, help='Directory with training data',
        default=opts.get("train_dir", "/opt/files/maio2022/SAT/NSNet/SATSolving/sr/train"))

    parser.add_argument('--train_size', type=int, default=opts.get("train_size", None), help='Number of training data')
    parser.add_argument('--valid_dir', type=str, default=opts.get("valid_dir", None), help='Directory with validating data')
    parser.add_argument('--loss', type=str, choices=['assignment', 'marginal'], default=opts.get("loss", 'marginal'), help='Loss type for SAT solving')
    parser.add_argument('--restore', type=str, default=opts.get("restore", None), help='Continue training from a checkpoint')
    parser.add_argument('--save_model_epochs', type=int, default=opts.get("save_model_epochs", 1), help='Number of epochs between model savings')
    parser.add_argument('--num_workers', type=int, default=opts.get("num_workers", 8), help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=opts.get("batch_size", 128), help='Batch size')
    parser.add_argument('--epochs', type=int, default=opts.get("epochs", 200), help='Number of epochs during training')
    parser.add_argument('--lr', type=float, default=opts.get("lr", 1e-4), help='Learning rate')
    parser.add_argument('--weight_dacay', type=float, default=opts.get("weight_decay", 1e-10), help='L2 regularization weight')
    parser.add_argument('--scheduler', type=str, default=opts.get("scheduler", None), help='Scheduler')
    parser.add_argument('--lr_step_size', type=int, default=opts.get("lr_step_size", 20), help='Learning rate step size')
    parser.add_argument('--lr_factor', type=float, default=opts.get("lr_factor", 0.5), help='Learning rate factor')
    parser.add_argument('--lr_patience', type=int, default=opts.get("lr_patience", 20), help='Learning rate patience')
    parser.add_argument('--clip_norm', type=float, default=opts.get("clip_norm", 0.65), help='Clipping norm')
    parser.add_argument('--seed', type=int, default=opts.get("seed", 0), help='Random seed')
    parser.add_argument('--valid_epochs', type=int, default=opts.get("valid_epochs", 5), help='Run validation every x epochs')

    parser.add_argument('--device', type=str, default=opts.get("device", None), help='Device: cpu, cuda or mps')

    add_model_options(parser, opts=opts)

    # opts = parser.parse_args()
    # Parse arguments when using Jupyter Notebook: https://stackoverflow.com/a/72670647/11750716
    opts, _unknown = parser.parse_known_args()

    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    opts.log_dir = os.path.join('runs', opts.exp_id)
    opts.checkpoint_dir = os.path.join(opts.log_dir, 'checkpoints')

    os.makedirs(opts.log_dir, exist_ok=True)
    os.makedirs(opts.checkpoint_dir, exist_ok=True)

    opts.log = os.path.join(opts.log_dir, 'log.txt')
    sys.stdout = Logger(opts.log, sys.stdout)
    sys.stderr = Logger(opts.log, sys.stderr)

    if not opts.device:
        opts.device = "cpu"
        if torch.cuda.is_available():
            opts.device = "cuda"
        elif torch.backends.mps.is_available():
            # Check that MPS is available (MacBook with M1 / M2 chip)
            # https://pytorch.org/docs/stable/notes/mps.html
            opts.device = "mps"

    # opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(opts)

    models = {
        'NSNet': NSNet,
        'NeuroSAT': NeuroSAT,
    }

    model = models[opts.model](opts)
    model.to(opts.device)

    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_dacay)
    train_loader = get_dataloader(opts.train_dir, opts, 'train', opts.train_size)

    if opts.valid_dir is not None:
        valid_loader = get_dataloader(opts.valid_dir, opts, 'valid')
    else:
        valid_loader = None

    if opts.scheduler is not None:
        if opts.scheduler == 'ReduceLROnPlateau':
            assert opts.valid_dir is not None
            scheduler = ReduceLROnPlateau(optimizer, factor=opts.lr_factor, patience=opts.lr_patience)
        else:
            assert opts.scheduler == 'StepLR'
            scheduler = StepLR(optimizer, step_size=opts.lr_step_size, gamma=opts.lr_factor)

    best_loss = float('inf')

    start_epoch = 0
    if opts.restore != None:
        print('Loading model checkpoint from %s..' % opts.restore)
        if opts.device == 'cpu':
            checkpoint = torch.load(opts.restore, map_location='cpu')
        else:
            checkpoint = torch.load(opts.restore)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        model.to(opts.device)

    for epoch in range(start_epoch, start_epoch + opts.epochs):
        print('EPOCH #%d' % epoch)
        print('Training...')
        train_loss = 0
        train_tot = 0
        train_rmse = 0
        train_cnt = 0

        model.train()
        for data in tqdm(train_loader):
            optimizer.zero_grad()
            data = data.to(opts.device)
            batch_size = data.c_size.shape[0]

            if opts.task == 'model-counting':
                preds = model(data)
                labels = data.y
                loss = F.mse_loss(preds, labels)
                mse = loss.item()
                train_rmse += mse * batch_size
            else:
                v_prob = model(data)
                c_size = data.c_size.sum().item()
                c_batch = data.c_batch
                l_edge_index = data.l_edge_index
                c_edge_index = data.c_edge_index

                if opts.loss == 'assignment':
                    preds = v_prob[:, 0]
                    labels = data.y
                    # The mean reduction divides the total loss by both the batch size and the support size.
                    # However, batchmean divides only by the batch size
                    # and aligns with the KL div math definition.
                    # This means that batchmean is a more consistent choice
                    # for loss reduction when training a model with a batch size greater than one.
                    loss = F.binary_cross_entropy(preds, labels)  #, reduction="batchmean")
                else:
                    preds = v_prob
                    labels = data.y
                    labels = torch.stack([labels, 1-labels], dim=1)
                    loss = F.kl_div(safe_log(preds), labels)  #, reduction="batchmean")

                v_assign = (v_prob > 0.5).float()
                l_assign = v_assign.reshape(-1)
                c_sat = torch.clamp(scatter_sum(l_assign[l_edge_index], c_edge_index, dim=0, dim_size=c_size), max=1)
                sat_batch = (scatter_sum(c_sat, c_batch, dim=0, dim_size=batch_size) == data.c_size).float()
                train_cnt += sat_batch.sum().item()

            train_loss += loss.item() * batch_size
            train_tot += batch_size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip_norm)
            optimizer.step()

        train_loss /= train_tot
        print('Training LR: %f, Training loss: %f' % (optimizer.param_groups[0]['lr'], train_loss))

        if opts.task == 'model-counting':
            train_rmse = math.sqrt(train_rmse / train_tot)
            print('Training RMSE: %f' % train_rmse)
        else:
            train_acc = train_cnt / train_tot
            print('Training accuracy: %f' % train_acc)

        if epoch % opts.save_model_epochs == 0:
            torch.save({
                'state_dict': model.state_dict(), 
                'epoch': epoch,
                'optimizer': optimizer.state_dict()}, 
                os.path.join(opts.checkpoint_dir, 'model_%d.pt' % epoch)
            )

        if opts.valid_dir is not None and epoch % opts.valid_epochs == 0:
            # Validate every x number of epochs
            print('Validating...')

            valid_loss = 0
            valid_tot = 0
            valid_rmse = 0
            valid_cnt = 0

            model.eval()
            for data in valid_loader:
                data = data.to(opts.device)
                batch_size = data.c_size.shape[0]
                with torch.no_grad():
                    if opts.task == 'model-counting':
                        preds = model(data)
                        labels = data.y
                        loss = F.mse_loss(preds, labels)
                        mse = loss.item()
                        valid_rmse += mse * batch_size
                    else:
                        v_prob = model(data)
                        c_size = data.c_size.sum().item()
                        c_batch = data.c_batch
                        l_edge_index = data.l_edge_index
                        c_edge_index = data.c_edge_index

                        if opts.loss == 'assignment':
                            preds = v_prob[:, 0]
                            labels = data.y
                            loss = F.binary_cross_entropy(preds, labels)  #, reduction="batchmean")
                        else:
                            preds = v_prob
                            labels = data.y
                            labels = torch.stack([labels, 1-labels], dim=1)
                            loss = F.kl_div(safe_log(preds), labels)   #, reduction="batchmean")

                        v_assign = (v_prob > 0.5).float()
                        preds = v_assign[:, 0]
                        l_assign = v_assign.reshape(-1)
                        c_sat = torch.clamp(scatter_sum(l_assign[l_edge_index], c_edge_index, dim=0, dim_size=c_size), max=1)
                        sat_batch = (scatter_sum(c_sat, c_batch, dim=0, dim_size=batch_size) == data.c_size).float()
                        valid_cnt += sat_batch.sum().item()

                valid_loss += loss.item() * batch_size
                valid_tot += batch_size

            valid_loss /= valid_tot
            print('Validating loss: %f' % valid_loss)

            if opts.task == 'model-counting':
                valid_rmse = math.sqrt(valid_rmse / valid_tot)
                print('Validating RMSE: %f' % valid_rmse)
            else:
                valid_acc = valid_cnt / valid_tot
                print('Validating accuracy: %f' % valid_acc)

            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch, 
                    'optimizer': optimizer.state_dict()}, 
                    os.path.join(opts.checkpoint_dir, 'model_best.pt')
                )

            if opts.scheduler is not None:
                if opts.scheduler == 'ReduceLROnPlateau':
                    scheduler.step(valid_loss)
                else:
                    scheduler.step()
        else:
            if opts.scheduler is not None:
                scheduler.step()

    # RETURN THE MODEL!!!
    return model, optimizer


if __name__ == '__main__':
    main()
