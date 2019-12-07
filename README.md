# cycleGAN

```
python train.py --dataroot ../project_data/horse2zebra --save_path ../project_runs/horse2zebra --scheduler_step_size 20 --train_epochs 100
```

continue training
```
python train.py --dataroot ..\project_data\horse2zebra\ --save_path ..\project_runs\horse2zebra\ --load_model ..\project_runs\horse2zebra\project_epoch_15.model --start_epochs 15 --train_epochs 50
```