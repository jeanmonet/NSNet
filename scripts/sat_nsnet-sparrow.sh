# --- RUN ALL THREE EXPERIMENTS WITH NSNet + Sparrow SOLVER ---

# export MODEL_PATH="src/runs/nsnet_base/checkpoints/model_150.pt"

./scripts/sat_nsnet-sparrow_ca.sh
./scripts/sat_nsnet-sparrow_3-sat.sh
./scripts/sat_nsnet-sparrow_sr.sh
