# Outliers exposure + mixup + GANs

This repository contains implementation code for the porject that exposes an anomaly detector to interpolations between normal and anomalous data generated using GANs architecture. 

## Setup
Install the libraries via:
```bash
pip install -r requirements.txt
```

If you are using Windows CMD first set the environment variable PTYHONPATH 
```bash
set PYTHONPATH=.
```

Otherwise add ```PYTHONPATH=.``` at the beginning of your command.

## Training on CIFAR dataset

To train the model run:
```
python src/run_cifar.py [OPTIONS]
```

### Available Options:

- `--g_lr`: Generator learning rate (default: 0.001)
- `--d_lr`: Discriminator learning rate (default: 0.001)
- `--n_epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 128)
- `--lambda_one`: Lambda 1 for loss function (default: 0.5)
- `--lambda_two`: Lambda 2 for loss function (default: 0.5)
- `--lambda_three`: Gradient penalty coefficient (default: 0.5)
- `--outliers_num`: Number of outliers to include (default: 1)
- `--normal_dataset_size`: Number of normal samples in the dataset (default: 128)
- `--interpolations_num`: Number of interpolations to use (default: 100)
- `--device`: Device for computation (`cpu` or `cuda`, default: `cpu`)
- `--normal_labels`: Labels for normal data (e.g., `1,2,3`)
- `--outliers_labels`: Labels for outlier data (e.g., `4,5`)
- `--g_milestones`: Learning rate scheduler milestones for the generator (default: empty)
- `--d_milestones`: Learning rate scheduler milestones for the discriminator (default: empty)
- `--description`: Brief description of the model or training
- `--save_checkpoint`: Save model every `n` epochs (default: 10)
- `--n_critic`: Number of discriminator updates before generator update (default: 5)
- `--cuda_no`: CUDA device number (if using GPU)
- `--num_workers`: Number of workers for data loading (default: 1)
- `--init_epochs`: Number of initialization epochs (default: 300)
- `--seed`: Random seed (default: 42)
- `--init_batch_size`: Initial batch size for generator training (default: 16)
- `--encoder_name`: Name of the encoder for image-to-latent space conversion
- `--vector_dim`: Dimension of the latent vector (default: 256)

### Example Usage
```bash
python run_cifar.py --g_lr 0.0005 --d_lr 0.0005 --n_epochs 200 --batch_size 64 --device cuda --normal_labels "0,1,2" --outliers_labels "8,9" 
```

For storing the results an SQLITE database is used.

Running this script will create a new file oe.db if it doesn't exist. If it does it will create a new entry in table ```models``` of the database oe.db. Every row has an `id` which we can later use for continuing training or evaluating. For every `save_checkpoint` a new entry into trainingresult is added with training loss, validation loss and epoch.


## Evaluating the model
Run the script
```bash
python src/run_evaluation.py [OPTIONS]
```
### Argument Descriptions:

- `--models`: Model ID and training results  
  **Type**: `str`  
  **Description**: The identifier of the model along with the training results. This helps in keeping track of which model is being used or referenced during testing or evaluation. The format is `model1:epoch1,model2:epoch2,model3:epoch3`. We provide multiple models that have the same set of parameters but were trained on a different seed.

- `--test_label`: Test labels  
  **Type**: `int`  
  **Description**: The labels used for the test dataset. This specifies which subset of data will be utilized for testing the model's performance.

- `--num_workers`: Number of workers (default: 1)  
  **Type**: `int`  
  **Description**: The number of worker threads used for data loading. Increasing the number of workers can speed up data loading, especially for larger datasets.

- `--device`: Device for computation (`cpu` or `cuda`, default: `cpu`)  
  **Type**: `str`  
  **Description**: The hardware device on which the model will run. It defaults to `'cpu'`, but if you have a GPU, you can set it to `'cuda'` for faster computation.

- `--batch_size`: Batch size (default: 128)  
  **Type**: `int`  
  **Description**: The number of samples per batch during training or evaluation. 

### Usage example:
```
python src/run_evaluation.py --models "1:100,2:110,3:120" --test_label 5 --num_workers 0 --device cpu --batch_size 64
```

