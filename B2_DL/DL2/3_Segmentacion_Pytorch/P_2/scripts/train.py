# Author: Antonio Martínez González
import argparse
import os
import torch
from torch import nn
import albumentations as alb
from albumentations.pytorch import ToTensorV2
import tqdm
from model import UNET
from dataset import carvanaDataset
from utils import evaluate, savePredicionsAsComposedImage
from time import time
from callbacks import earlyStopping, modelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('-itrd', '--image_train_directory', type=str, required=True, default=None, help="String or path-like object. Path of the directory containing the images that will be used as training set")
parser.add_argument('-mtrd', '--mask_train_directory', type=str, required=True, default=None, help="String or path-like object. Path of the directory containing the masks that will be used as training set. Each mask filename must be a modified version the corresponding image where the '.jpg' termination is substituted by a '_mask.gif' termination.")
parser.add_argument('-itsd', '--image_test_directory', type=str, required=True, default=None, help="String or path-like object. Path of the directory containing the images that will be used as validation/test set")
parser.add_argument('-mtsd', '--mask_test_directory', type=str, required=True, default=None, help="String or path-like object. Path of the directory containing the masks that will be used as validation/test set. Each mask filename must be a modified version the corresponding image where the '.jpg' termination is substituted by a '_mask.gif' termination.")
parser.add_argument('-mpt', '--mixed_precision_training', type=bool, required=False, default=False, help="Boolean. Flag that determines wether to use mixed precision during training. Often leads to less power and memory consumption when training on GPU devices.")
parser.add_argument('-ic', '--in_channels', type=int, required=False, default=3, help="Integer. Number of channels of input image.")
parser.add_argument('-oc', '--out_channels', type=int, required=False, default=1, help="Integer. Number of binary masks per image (equals to the number of classes in the semantic segmentation problem)")
parser.add_argument('-bn', '--batch_norm', type=bool, required=False, default=True, help="Boolean. Flag that determines the application of batch normalization after each convolutional block and before the activation function.")
parser.add_argument('-ne', '--n_epochs', type=int, required=False, default=5, help="Integer. Number of epochs during training stage.")
parser.add_argument('-lr', '--learning_rate', type=float, required=False, default=0.0001, help="Float. Optimizer's learning rate.")
parser.add_argument('-bs', '--batch_size', type=int, required=False, default=16, help="Integer. Batch size. Number of samples that will be used to update model weigths on each iteration of the backpropagation algorithm.")
parser.add_argument('-nwk', '--n_workers', type=int, required=False, default=1, help="Integer. Number of workers or threads for multiprocessing. Setting values greater than one may lead to less execution times if the data loading/preprocessing step is expensive in compute terms.")
parser.add_argument('-sh', '--shuffle', type=bool, required=False, default=True, help="Boolean. Flags that determines wether to shuffle the training set between epochs. Setting this parameter to True often leads to a more stable stable training phase and better generalization.")
parser.add_argument('-cr', '--crop', type=bool, required=False, default=True, help="Boolean. Flag that determines wether to crop the skip connnections for UNET network or to apply zero padding. Only applies if input image heigth or width are not divisible by 2**n_blocks, where n_blocks is the number of convolutional blocks in the UNET network (assuming a max pooling operation with square kernel and stride=2 after each convolutional block).")
parser.add_argument('-ih', '--image_height', type=int, required=False, default=160, help="Integer. Desired image/mask heigth for both training and test sets.")
parser.add_argument('-iw', '--image_width', type=int, required=False, default=240, help="Integer. Desired image/mask width for both training and test sets.")
parser.add_argument('-lm', '--load_pretrained', type=str, required=False, default=None, help="String or path-like object. Path of the .pth file obtained from a previous training session.")
parser.add_argument('-mp', '--model_path', type=str, required=False, default="models/model.pth", help="String or path-like object. Path where best model weigths and optimizer state will be saved.")
parser.add_argument('-es', '--early_stopping', type=bool, required=False, default=False, help="Boolean. Flag that determines the application of early stopping to the training phase")
parser.add_argument('-pt', '--patience', type=int, required=False, default=3, help="Integer. Number of epochs without improvement in performance before stopping training.")
parser.add_argument('-md', '--min_delta', type=float, required=False, default=0.0, help="Float. Change in loss less than or equal to min_delta is considered as no improvement.")
parser.add_argument('-mod', '--mode', type=str, required=False, default="max", help="String. Determines if metrics must be minimized or maximized for callbacks.")
parser.add_argument('-pdir', '--predictions_dir', type=str, required=False, default="predictions", help="String. Path of the directory where predicted masks will be saved as a concatenated image.")
args = parser.parse_args()

# TODO: handle incorrect cases of args.image_width and args.image_height for resizing operation
# TODO: select monitoring metric used for callbacks, from the set {loss, acc, dice}
# TODO: select albumentations transform (and parameters) from a set of pre-defined values
# TODO: select which metrics to use (and losses)

if __name__ == "__main__":

    # Setting best device for training according to our system capabilities
    device = (
        "cuda" # Nvidia CUDA-compatible GPU
        if torch.cuda.is_available()
        else "mps" # MPS-compatible MAC computer
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Creates the directory (if specified and if it does not exists) where model weigths and optimizer state will be saved
    if args.model_path:
        dirname = os.path.dirname(args.model_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

    # Build UNET model and move it to the chosen device
    model = UNET(in_channels=args.in_channels, out_channels=args.out_channels, batch_norm=args.batch_norm, crop=args.crop).to(device)
    # Loss function. Logits are needed because no sigmoid is applied at the end of the forward pass due to compatibility issues with mixed precision training
    loss_fn = nn.BCEWithLogitsLoss()
    # Build Adam optimizer and set learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Build trainig set with data augmentation to increase the generalization performance
    train_dataset = torch.utils.data.DataLoader(
        carvanaDataset(
            image_dir=args.image_train_directory,
            mask_dir=args.mask_train_directory,
            transform=alb.Compose([
                alb.Resize(height=args.image_height, width=args.image_width),
                alb.Rotate(limit=35, p=0.9),
                alb.HorizontalFlip(p=0.5),
                alb.VerticalFlip(p=0.2),
                ToTensorV2(),
            ]),
            scale=True,
        ),
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        shuffle=args.shuffle,
    )

    # Build test set. No data augmentation/shuffling is needed.
    test_dataset = torch.utils.data.DataLoader(
        carvanaDataset(
            image_dir=args.image_test_directory,
            mask_dir=args.mask_test_directory,
            transform=alb.Compose([
                alb.Resize(height=args.image_height, width=args.image_width),
                ToTensorV2(),
            ]),
            scale=True,
        ),
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        shuffle=False,
    )

    # If pretrained checkpoint is given, set model weigths and optimizer state and evaluate model performance on training and test set
    if args.load_pretrained:
        # carga del fichero pth
        ckpt = torch.load(args.load_pretrained)
        # carga los pesos al modelo
        model.load_state_dict(ckpt["model_state_dict"])
        # carga el estado del optimizador (importante!!)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # evalua el rendimiento del modelo pre-entrenado
        train_loss, train_acc, train_dice = evaluate(train_dataset, model, loss_fn, device)
        test_loss, test_acc, test_dice = evaluate(test_dataset, model, loss_fn, device)
        # muestra por pantalla las metricas obtenidas
        print(f"Performance of pretrained model --> Train loss : {train_loss:.4f} || Test loss : {test_loss:.4f} || Train Acc : {train_acc:.4f} || Test Acc : {test_acc:.4f} || Train DICE : {train_dice:.4f} || Test DICE : {test_dice:.4f}")
    
    # If mixed precision training is specified initialize the gradient scaler
    if args.mixed_precision_training:
        gradscaler = torch.cuda.amp.GradScaler()

    # Model checkpoint
    model_checkpoint_callback = modelCheckpoint(path=args.model_path, min_delta=args.min_delta, mode=args.mode)
        
    # Early stopping
    if args.early_stopping:
        early_stopping_callback = earlyStopping(patience=args.patience, min_delta=args.min_delta, mode=args.mode)

    
    # Set model to 'training' mode
    model.train()
    # Start training phase. Iterate over the number of epochs
    for epoch in range(args.n_epochs):
        t0 = time() # For computing training time per epoch
        # Iterate over the training dataset
        for input_image, target_mask in train_dataset:
            # Move input image and target mask to chosen device
            input_image, target_mask = input_image.to(device), target_mask.to(device)
            # Set gradients to zero
            optimizer.zero_grad()
            if args.mixed_precision_training:
                with torch.cuda.amp.autocast():
                    predicted_mask = model(input_image).squeeze(1)
                    loss = loss_fn(predicted_mask, target_mask)
                # ...
                gradscaler.scale(loss).backward()
                # Update weigths
                gradscaler.step(optimizer)
                gradscaler.update()
            else:
                predicted_mask = model(input_image).squeeze(1)
                loss = loss_fn(predicted_mask, target_mask)
                # Compute gradients
                loss.backward()
                # Update weigths
                optimizer.step()
        t1 = time() # For computing training time per epoch

        # Compute accuracy and dice score over training set
        train_loss, train_acc, train_dice = evaluate(train_dataset, model, loss_fn, device)
        # Compute accuracy and dice score over test set
        test_loss, test_acc, test_dice = evaluate(test_dataset, model, loss_fn, device)
        # Display results
        print(f"Train loss : {train_loss:.4f} || Test loss : {test_loss:.4f} || Train Acc : {train_acc:.4f} || Test Acc : {test_acc:.4f} || Train DICE : {train_dice:.4f} || Test DICE : {test_dice:.4f} || Epoch time : {t1-t0:.2f} s")

        # Model checkpoint
        model_checkpoint_callback(model, optimizer, test_dice)

        if args.early_stopping:
            early_stopping_callback(test_dice)
            if early_stopping_callback.stop:
                print(f"Early stopping on epoch {epoch+1}")
                break

    ckpt = torch.load(args.model_path)
    # carga los pesos al modelo
    model.load_state_dict(ckpt["model_state_dict"])
    savePredicionsAsComposedImage(test_dataset, model, device, dir=args.predictions_dir)