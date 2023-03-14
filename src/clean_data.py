"""
# Ensore inside MSNet folder
% pwd
/opt/files/maio2022/SAT/NSNet

# Run download (this) script -> specify output folder
% python src/clean_data.py /opt/files/maio2022/SAT/NSNet/SATSolving/SATLIB

"""


import os
import argparse
import glob

from tqdm import tqdm

from utils.utils import write_dimacs_to
from utils.utils import preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Directory with sat data')
    opts = parser.parse_args()
    print(opts)

    print('Cleaning files...')
    all_files = sorted(glob.glob(opts.data_dir + '/**/*.cnf', recursive=True))
    all_files = [os.path.abspath(f) for f in all_files]

    for f in tqdm(all_files):
        n_vars, clauses = preprocess(f)
        write_dimacs_to(n_vars, clauses, f)


if __name__ == '__main__':
    main()
