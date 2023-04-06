python src/train_model.py sat-solving --exp_id sat_nsnet_3-sat_marginal --train_dir /opt/files/maio2022/SAT/NSNet/SATSolving/3-sat/train/ --valid_dir /opt/files/maio2022/SAT/NSNet/SATSolving/3-sat/valid/ --epochs 200 --scheduler ReduceLROnPlateau --lr_step_size 20 --loss marginal --batch_size 64 --save_model_epochs 1

python src/test_model.py sat-solving /opt/files/maio2022/SAT/NSNet/SATSolving/3-sat/test/ --checkpoint runs/sat_nsnet_3-sat_marginal/checkpoints/model_best.pt --batch_size 32

python src/test_model.py sat-solving /opt/files/maio2022/SAT/NSNet/SATSolving/3-sat/test_hard/ --checkpoint runs/sat_nsnet_3-sat_marginal/checkpoints/model_best.pt --batch_size 32
