python src/train_model.py sat-solving --exp_id sat_nsnet_sr_marginal --train_dir /opt/files/maio2022/SAT/NSNet/SATSolving/sr/train/ --valid_dir /opt/files/maio2022/SAT/NSNet/SATSolving/sr/valid/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss marginal --batch_size 64 --save_model_epochs 1

python src/test_model.py sat-solving /opt/files/maio2022/SAT/NSNet/SATSolving/sr/test/ --checkpoint runs/sat_nsnet_sr_marginal/checkpoints/model_best.pt --batch_size 32

python src/test_model.py sat-solving /opt/files/maio2022/SAT/NSNet/SATSolving/sr/test_hard/ --checkpoint runs/sat_nsnet_sr_marginal/checkpoints/model_best.pt --batch_size 32


# python src/train_model.py sat-solving sat_nsnet_sr_marginal ~/scratch/NSNet/SATSolving/sr/train/ --valid_dir ~/scratch/NSNet/SATSolving/sr/valid/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss marginal
# python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/sr/test/ --checkpoint runs/sat_nsnet_sr_marginal/checkpoints/model_best.pt
# python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/sr/test_hard/ --checkpoint runs/sat_nsnet_sr_marginal/checkpoints/model_best.pt

# python src/train_model.py sat-solving sat_nsnet_sr_assignment ~/scratch/NSNet/SATSolving/sr/train/ --valid_dir ~/scratch/NSNet/SATSolving/sr/valid/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss assignment
# python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/sr/test/ --checkpoint runs/sat_nsnet_sr_assignment/checkpoints/model_best.pt
# python src/test_model.py sat-solving ~/scratch/NSNet/SATSolving/sr/test_hard/ --checkpoint runs/sat_nsnet_sr_assignment/checkpoints/model_best.pt