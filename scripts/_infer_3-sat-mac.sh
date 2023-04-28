MODEL_PATH="src/runs/nsnet_3sat/checkpoints/model_200.pt"

# --- Infer marginals wiht NSNet and save in .out file for each CNF ---
python src/run_model.py SATSolving/3-sat/test_hard --checkpoint $MODEL_PATH --model NSNet --n_rounds 10 --batch_size 32

# --- Run Sparrow solver with initial marginal assignment from NSNet (above) ---
python src/test_sat_solver.py SATSolving/3-sat/test_hard --solver Sparrow --max_flips 100 --n_process 8 --trial 0 --model NSNet
python src/test_sat_solver.py SATSolving/3-sat/test_hard --solver Sparrow --max_flips 100 --n_process 8 --trial 1 --model NSNet
python src/test_sat_solver.py SATSolving/3-sat/test_hard --solver Sparrow --max_flips 100 --n_process 8 --trial 2 --model NSNet

# --- Show results ---
python src/show_sat_result.py runs/Sparrow/evaluations/ 3-sat --model NSNet
