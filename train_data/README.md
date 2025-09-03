## Script to train and evaliate E3SM outputs

* Train E3SM output training data produced using **extract_train_info.py**
* Generate best trained model, Train & Validation losses, and a bias-variance table.

Usage
-----

```console
python trainingMAM.py -h
```
```bash
usage: trainingMAM.py [-h] [--natmos NATMOS] [--batch_size BATCH_SIZE] [--depths DEPTHS [DEPTHS ...]]
                      [--widths WIDTHS [WIDTHS ...]] [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY]
                      [--n_models N_MODELS] [--n_epochs N_EPOCHS] [--loss_func LOSS_FUNC] [--varlist VARLIST [VARLIST ...]]
                      [--transformation_type TRANSFORMATION_TYPE]
                      data_path

Train a model using ModelTrainer

positional arguments:
  data_path             Path to the data directory

options:
  -h, --help            show this help message and exit
  --natmos NATMOS       Number of atmospheric variables
  --batch_size BATCH_SIZE
                        Batch size for training
  --depths DEPTHS [DEPTHS ...]
                        List of model depths
  --widths WIDTHS [WIDTHS ...]
                        List of model widths
  --learning_rate LEARNING_RATE
                        Learning rate for optimizer
  --weight_decay WEIGHT_DECAY
                        Weight decay for optimizer
  --n_models N_MODELS   Number of models to train
  --n_epochs N_EPOCHS   Number of epochs for training
  --loss_func LOSS_FUNC
                        Loss function to use (default: MSE)
  --varlist VARLIST [VARLIST ...]
                        List of variables
  --transformation_type TRANSFORMATION_TYPE
                        Type of transformation (default: standardization)
```

Example use
-----------

```console
python trainingMAM.py /Users/hass877/Work/data_analysis --natmos 20 --n_epochs 10 --depths 3 --widths 128
```
```
Loss Function:  MSE
Training Data Shapes:

(224764, 40) (224764, 20)
Validation Data Shapes:

(218154, 40) (218154, 20)
Testing Data Shapes:

(218155, 40) (218155, 20)

Applying transformation:  standardization

Training models with depth: 3, width: 128

Number of parameters: 49152

Model Run:  1
100%|███████████████████████████████████████████████████████████████████████████████████████| 878/878 [00:04<00:00, 178.97it/s]
Epoch [1/10], Training Loss: 0.489182, Validation Loss: 0.367898
100%|███████████████████████████████████████████████████████████████████████████████████████| 878/878 [00:02<00:00, 314.51it/s]
Epoch [2/10], Training Loss: 0.346363, Validation Loss: 0.329220
100%|███████████████████████████████████████████████████████████████████████████████████████| 878/878 [00:02<00:00, 311.85it/s]
Epoch [3/10], Training Loss: 0.316491, Validation Loss: 0.314689
100%|███████████████████████████████████████████████████████████████████████████████████████| 878/878 [00:02<00:00, 309.37it/s]
Epoch [4/10], Training Loss: 0.295509, Validation Loss: 0.296138
100%|███████████████████████████████████████████████████████████████████████████████████████| 878/878 [00:02<00:00, 319.61it/s]
Epoch [5/10], Training Loss: 0.282081, Validation Loss: 0.281795
100%|███████████████████████████████████████████████████████████████████████████████████████| 878/878 [00:02<00:00, 314.98it/s]
Epoch [6/10], Training Loss: 0.268901, Validation Loss: 0.270303
100%|███████████████████████████████████████████████████████████████████████████████████████| 878/878 [00:02<00:00, 310.62it/s]
Epoch [7/10], Training Loss: 0.250892, Validation Loss: 0.268200
100%|███████████████████████████████████████████████████████████████████████████████████████| 878/878 [00:02<00:00, 317.80it/s]
Epoch [8/10], Training Loss: 0.251187, Validation Loss: 0.351005
patience_counter: 1
100%|███████████████████████████████████████████████████████████████████████████████████████| 878/878 [00:02<00:00, 312.19it/s]
Epoch [9/10], Training Loss: 0.234974, Validation Loss: 0.238122
100%|███████████████████████████████████████████████████████████████████████████████████████| 878/878 [00:02<00:00, 309.28it/s]
Epoch [10/10], Training Loss: 0.216744, Validation Loss: 0.223775
Training finished.
bc_a1: R2 = 0.9817214326448855
bc_a4: R2 = 0.9816318737925327
pom_a1: R2 = 0.9742863768377038
pom_a4: R2 = 0.9734894583230497
so4_a1: R2 = 0.9930498890189144
so4_a2: R2 = 0.9532570408566492
so4_a3: R2 = 0.9432513382837113
mom_a1: R2 = 0.05611388880152912
mom_a2: R2 = 0.047788480376773745
mom_a4: R2 = 0.3851340681574029
ncl_a1: R2 = -0.9452838837729705
ncl_a2: R2 = -0.9683945565797871
soa_a1: R2 = 0.7712646039815026
soa_a2: R2 = -0.37163880368389357
soa_a3: R2 = 0.5881786721600462
num_a1: R2 = 0.9057875647820443
num_a2: R2 = 0.7117522009442359
num_a4: R2 = 0.9716111502390882
H2SO4: R2 = 0.993427273027802
SOAG: R2 = 0.7715640615622699
```
