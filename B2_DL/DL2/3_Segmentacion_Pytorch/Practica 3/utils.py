import os
import torch
import numpy as np
from dataset import cityscapesDataset
from torchvision.transforms import ToPILImage
import albumentations as A
from albumentations.pytorch import ToTensorV2
torch_to_pil = ToPILImage()

COLORMAP = {
    0 : [0,0,0],
    1 : [128,64,128],
    2 : [244,35,232],
    3 : [70,70,70],
    4 : [102,102,156],
    5 : [190,153,153],
    6 : [153,153,153],
    7 : [250,170,30],
    8 : [220,220,0],
    9 : [107,142,35],
    10 : [152,251,152],
    11 : [70,130,180],
    12 : [220,20,60],
    13 : [255,0,0],
    14 : [0,0,142],
    15 : [0,0,70],
    16 : [0,60,100],
    17 : [0,80,100],
    18 : [0,0,230],
    19 : [119,11,32],
}

def mask_to_cityscapes_rgb(mask):
    """ Convierte una mascara del dataset Cityscapes, donde cada elemento
    contiene el índice de la clase a la que pertenece, en una imágen RGB respetando
    la correspondencia definida en el diccionario COLORMAP """

    if not isinstance(mask, np.ndarray) or not mask.ndim == 2:
      raise ValueError("Input must be an 2D np.array")

    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8) # Inicializacion a cero
    for class_index in np.unique(mask).astype(int):
        mask_rgb[mask == class_index] = COLORMAP[class_index]

    return mask_rgb

def savePredicionsAsComposedImage(
        dataset,
        image_dir,
        mask_dir,
        image_height,
        image_width,
        batch_size,
        num_workers,
        pin_memory,
        model,
        device,
        aux=False, 
        dir=os.path.join("dataset", "predictions"), 
        n_samples=10
    ):
    # Crea el directorio 'dir' si no existe
    if not os.path.exists(dir):
        os.makedirs(dir)

    model = model.to(device)

    original_dataset = torch.utils.data.DataLoader(
    cityscapesDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=A.Compose([
            A.Resize(height=image_height, width=image_width),
            ToTensorV2()
        ]),
        scale=True,
    ),
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=pin_memory,
    )

    # Pasa el modelo a modo evaluacion
    model.eval()
    # Evitar el calculo de gradientes
    with torch.no_grad():
        # Iterar sobre el dataset
        counter = 0
        for (original, _), (x, y) in zip(original_dataset, dataset):
            # Mover los tensores al device
            x, y = x.to(device), y.to(device)
            preds = model(x)
            if aux:
                preds = preds["out"]
            preds = torch.softmax(preds, dim=1)
            preds = torch.argmax(preds, dim=1)
            for i in range(x.size(0)):
                concatenated = torch.cat(
                    [
                        original[i], 
                        torch.from_numpy(np.transpose(mask_to_cityscapes_rgb(y[i].cpu().numpy()), (2,0,1))), 
                        torch.from_numpy(np.transpose(mask_to_cityscapes_rgb(preds[i].cpu().numpy()), (2,0,1)))
                    ], 
                    dim=2)                
                image = torch_to_pil(concatenated)
                image.save(os.path.join(dir, f"{counter}.jpg"))
                counter += 1
                if counter == n_samples:
                    # Pasa el modelo a modo entrenamiento
                    model.train()
                    return None
    model.train()
