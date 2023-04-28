"""
# Ensore inside MSNet folder
% pwd
/opt/files/maio2022/SAT/NSNet

# Run (this) script -> specify output folder
% python src/run_model.py /opt/files/maio2022/SAT/NSNet/SATSolving/SATLIB --model NSNet
"""

import os
import argparse
import random
import glob
import time
import warnings
from itertools import accumulate

import numpy as np
import torch
from tqdm import tqdm

from utils.options import add_model_options
from utils.dataloader import get_dataloader
from models.bp import BP
from models.nsnet import NSNet
from models.neurosat import NeuroSAT


def run_model(opts, verbose: bool = False):
    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print(opts)

    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    t0 = time.time()

    models = {
        'BP': BP,
        'NSNet': NSNet,
        'NeuroSAT': NeuroSAT,
    }

    model = models[opts.model](opts)
    model.to(opts.device)

    test_loader = get_dataloader(opts.test_dir, opts, 'test')
    all_files = sorted(glob.glob(opts.test_dir + '/**/*.cnf', recursive=True))
    all_files = [os.path.abspath(f) for f in all_files]

    if opts.checkpoint is not None:
        if verbose:
            print('Loading model checkpoint from %s..' % opts.checkpoint)
        if opts.device.type == 'cpu':
            checkpoint = torch.load(opts.checkpoint, map_location='cpu')
        else:
            checkpoint = torch.load(opts.checkpoint)

        model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.to(opts.device)

    out_name = f"{opts.model}" + (f"_{opts.out_name}" if getattr(opts, "out_name", None) else "")

    # --- Silence warning ---
    # /opt/miniconda3/envs/pt10/lib/python3.10/site-packages/torch_geometric/data/collate.py:145: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
    # with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)

    if verbose:
        print(f'Running... (saving assignment to `[cnf filename]_{out_name}.out`)')

    model.eval()
    i = 0
    for data in tqdm(test_loader):
        data = data.to(opts.device)
        with torch.no_grad():
            v_probs = model(data)
            v_assigns = (v_probs > 0.5).float()
            assignments = v_assigns[:, 0]
            v_sizes = (data.l_size / 2).int()
            for offset, v_size in zip(accumulate(v_sizes), v_sizes):
                assignment = assignments[offset-v_size:offset]
                f = all_files[i]
                filename = os.path.splitext(os.path.basename(f))[0]
                tmp_filepath = os.path.join(os.path.dirname(f), filename + '_' + out_name + '.out')
                with open(tmp_filepath, 'w') as f:
                    for v in assignment:
                        f.write('%d ' % v)
                    f.write('\n')
                i += 1

    assert i == len(all_files)

    t = time.time() - t0
    if verbose:
        print('Running Time: %f' % t)

    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dir', type=str, help='Directory with testing data')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to be tested')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--model', type=str, default='NSNet', help='Model to be tested')
    # Add argument that adds name to .out filename
    parser.add_argument('--out_name', type=str, default=None, help='Name to be added to .out filename')

    add_model_options(parser)

    opts = parser.parse_args()
    opts.task = 'sat-solving'

    t = run_model(opts, verbose=True)


if __name__ == '__main__':
    main()
