python src/train_model.py sat-solving --exp_id sat_nsnet_ca_marginal --train_dir /opt/files/maio2022/SAT/NSNet/SATSolving/ca/train/ --valid_dir /opt/files/maio2022/SAT/NSNet/SATSolving/ca/valid/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss marginal --batch_size 64 --save_model_epochs 5

python src/test_model.py sat-solving /opt/files/maio2022/SAT/NSNet/SATSolving/ca/test/ --checkpoint runs/sat_nsnet_ca_marginal/checkpoints/model_best.pt --batch-size 64

python src/test_model.py sat-solving /opt/files/maio2022/SAT/NSNet/SATSolving/ca/test_hard/ --checkpoint runs/sat_nsnet_ca_marginal/checkpoints/model_best.pt --batch-size 64


# python src/train_model.py sat-solving sat_nsnet_ca_marginal ~/scratch/NSNet/SATSolving/ca/train/ --valid_dir ~/scratch/NSNet/SATSolving/ca/valid/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss marginal
# python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/ca/test/ --checkpoint runs/sat_nsnet_ca_marginal/checkpoints/model_best.pt
# python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/ca/test_hard/ --checkpoint runs/sat_nsnet_ca_marginal/checkpoints/model_best.pt

# python src/train_model.py sat-solving sat_nsnet_ca_assignment ~/scratch/NSNet/SATSolving/ca/train/ --valid_dir ~/scratch/NSNet/SATSolving/ca/valid/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss assignment
# python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/ca/test/ --checkpoint runs/sat_nsnet_ca_assignment/checkpoints/model_best.pt
# python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/ca/test_hard/ --checkpoint runs/sat_nsnet_ca_assignment/checkpoints/model_best.pt