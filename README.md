# cycleGAN

```
python train.py --dataroot ../project_data/horse2zebra --save_path ../project_runs/horse2zebra --scheduler_step_size 20 --train_epochs 100
```

continue training
```
python train.py --dataroot ../project_data/horse2zebra/ --save_path ../project_runs/horse2zebra/ --load_model ../project_runs/horse2zebra/project_epoch_15.model --start_epochs 15 --train_epochs 50
```

use linear decay instead. LR is unchanged for first scheduler_step_size then decay to 0 in next scheduler_step_size
```
python train.py --dataroot ../project_data/horse2zebra --save_path ../project_runs/horse2zebra_linear --scheduler_step_size 50 --train_epochs 100 --scheduler_type linear_decay
```

dataroot: data folder which contains trainA and trainB

batch_size: should be 1 otherwise there is not enough RAM

lambda_A, lambda_B, lambda_idt: described in the paper

save_path, save_prefix: result path and prefix

lr: initial learning rate

beta1: Adam optimizer beta1

train_epochs: number of epochs to train

start_epochs: used in continue training.

max_length: maximum length of the dataset, can be lowered to a small number to test code

save_freq, img_freq: save model or print sample img every several epochs

scheduler_step_size: step size when using StepLR; constant and decay length when using linear_decay

scheduler_gamma: gamma when using StepLR

load_model: load model path when continue training

reset_optimizer: turn on if trying to reset optimizer when continue training

ngf: generator channel base count

ndf: discriminator channel base coung

g_blocks: generator number of blocks

n_layers: discriminator number of layers

use_dropout: use dropout or not

g_n_downsampling: generator downsampling layer number

dropout: dropout probabilty when use_dropout is set to true

scheduler_type: StepLR or linear_decay

