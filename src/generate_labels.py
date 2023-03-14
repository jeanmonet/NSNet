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

# Ensore inside MSNet folder
% pwd
/opt/files/maio2022/SAT/NSNet

# Run label generator (this) script -> specify output folder
# Generates "marginals" using Minisat ALLSAT solver
#    (see page 6, point 4.1 in https://arxiv.org/pdf/2211.03880.pdf)
% python src/generate_labels.py marginal /opt/files/maio2022/SAT/NSNet/SATSolving/SATLIB --n_process 2
"""

import os
import argparse
import glob
import pickle
import shutil

from concurrent.futures.process import ProcessPoolExecutor

from tqdm import tqdm

from utils.solvers import MCSolver
from utils.solvers import SATSolver
from utils.solvers import MESolver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['model-counting', 'assignment', 'marginal'], help='Task')
    parser.add_argument('data_dir', type=str, help='Directory with sat data')
    parser.add_argument('--out_dir', type=str, default=None, help='Output Directory with sat data')
    parser.add_argument('--n_process', type=int, default=8, help='Number of processes')
    parser.add_argument('--timeout', type=int, default=5000, help='Timeout')

    opts = parser.parse_args()
    print(opts)

    if opts.task == 'model-counting':
        opts.solver = 'DSHARP'
    elif opts.task == 'assignment':
        opts.solver = 'CaDiCaL'
    else:
        # Ex. opts.task = "margianal"
        opts.solver = 'bdd_minisat_all'

    if opts.out_dir is not None:
        os.makedirs(opts.out_dir, exist_ok=True)

    if opts.task == 'model-counting':
        solver = MCSolver(opts)
    elif opts.task == 'assignment':
        solver = SATSolver(opts)
    else:
        # external/bdd_minisat_all
        solver = MESolver(opts)

    labels = []

    print('Generating labels...')
    all_files = sorted(glob.glob(opts.data_dir + '/**/*.cnf', recursive=True))
    all_files = [os.path.abspath(f) for f in all_files]

    with ProcessPoolExecutor(max_workers=opts.n_process) as pool:
        # results = pool.map(solver.run, tqdm(all_files))
        try:
            results = pool.map(solver.run, tqdm(all_files))
        except KeyboardInterrupt:
            print("Terminating processes")
            pool.shutdown(wait=True)
            raise

    print("DONE RUNNING PARALLEL SOLVERS. RESULTS:", len(results))

    # # !!! RUN SOLVER SEQUENTIALLY INSTEAD OF USING MULTI-THREADING !!!
    # results = []
    # print("RUNNING SOLVER SEQUENTIALLY")
    # for file in tqdm(all_files):
    #     # Run solver sequentially
    #     results.append(solver.run(file))
    # print("DONE RUNNING SOLVER SEQUENTIALLY. RESULTS:", len(results))

    tot = len(all_files)
    cnt = 0

    for i, result in enumerate(tqdm(results)):
        if opts.task == 'model-counting':
            complete, counting, t = result
        elif opts.task == 'assignment':
            complete, assignment, _, t = result
        else:
            complete, marginal, t = result

        if complete:
            cnt += 1
            if opts.task == 'model-counting':
                ln_counting = float(counting.ln())
                labels.append(ln_counting)
            elif opts.task == 'assignment':
                labels.append(assignment)
            else:
                labels.append(marginal)

            if opts.out_dir is not None:
                shutil.copyfile(all_files[i], os.path.join(opts.out_dir, '%.5d.cnf' % (cnt)))
        else:
            if opts.out_dir is None:
                os.remove(all_files[i])

    r = cnt / tot
    print('Total: %d, Labeled: %d, Ratio: %.4f.' % (tot, cnt, r))

    if opts.out_dir is not None:
        if opts.task == 'model-counting':
            labels_file = os.path.join(opts.out_dir, 'countings.pkl')
        elif opts.task == 'assignment':
            labels_file = os.path.join(opts.out_dir, 'assignments.pkl')
        else:
            labels_file = os.path.join(opts.out_dir, 'marginals.pkl')
    else:
        if opts.task == 'model-counting':
            labels_file = os.path.join(opts.data_dir, 'countings.pkl')
        elif opts.task == 'assignment':
            labels_file = os.path.join(opts.data_dir, 'assignments.pkl')
        else:
            # Task: "marginal"
            labels_file = os.path.join(opts.data_dir, 'marginals.pkl')

    with open(labels_file, 'wb') as f:
        pickle.dump(labels, f)


if __name__ == '__main__':
    main()
