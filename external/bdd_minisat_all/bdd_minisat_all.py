"""
# ------------------------------------------------------------------
# --- First ensure bdd_minisat_all -> AllSAT solver is compiled ---
# ------------------------------------------------------------------
- http://www.sd.is.uec.ac.jp/toda/code/cnf2obdd.html#download
- https://arxiv.org/abs/1510.00523

Subdirectory "external/bdd_minisat_all" should contain:
- bdd_minisat_all.py
- bdd_minisat_all       # executable & compiled for current OS/machine

# Inside MSNet/external/bdd_minisat_all directory:
% rm bdd_minisat_all         # delete executable compiled on different OS/machine
% wget http://www.sd.is.uec.ac.jp/toda/code/bdd_minisat_all-1.0.2.tar.gz
% tar -xvzf bdd_minisat_all-1.0.2.tar.gz
% cd bdd_minisat_all-1.0.2
% make
% cp bdd_minisat_all ./../bdd_minisat_all
% cd ..
% rm -rf bdd_minisat_all-1.0.2
% rm bdd_minisat_all-1.0.2.tar.gz
# ------------------------------------------------------------------
"""


import os
import argparse
import subprocess
import pickle
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='Input file')
    parser.add_argument('tmp_output_file', type=str, help='Temporary output file')
    parser.add_argument('output_file', type=str, help='Output file')
    parser.add_argument("--max_obdd_nodes", type=int, default=100_000_000, help="Maximum number of OBDD nodes")
    opts = parser.parse_args()

    cmd_line = ['./bdd_minisat_all', opts.input_file, opts.tmp_output_file]

    if opts.max_obdd_nodes is not None:
        cmd_line.append(f"-n{opts.max_obdd_nodes}")

    # may also finished by linux oom killer
    subprocess.run(cmd_line)

    with open(opts.tmp_output_file, 'r') as f:
        # may also finished by linux oom killer
        lines = f.readlines()
        counting = len(lines)
        n_vars = len(lines[0].strip().split()) - 1
        marginal = np.zeros(n_vars)
        for line in lines:
            assignment = np.array([int(s) for s in line.strip().split()[:-1]])
            assignment = assignment > 0
            marginal += assignment
        marginal /= counting

    with open(opts.output_file, 'wb') as f:
        pickle.dump(marginal, f)


if __name__ == '__main__':
    main()
