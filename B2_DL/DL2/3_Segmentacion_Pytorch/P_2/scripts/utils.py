# Author: Antonio Martínez González
import torch
import torchvision
import os
import random
import shutil

def extractCarvanaSamples(image_dir, mask_dir, new_image_dir, new_mask_dir, ratio, shuffle=True):

    if os.path.exists(image_dir) and not os.path.isdir(image_dir):
        raise ValueError(f"The value passed to the parameter image_dir must be the path to an exiting. Received {image_dir}")
    elif not os.path.exists(image_dir):
        raise ValueError(f"The value passed to the paremeter image_dir must be the path to a directory containing Carvana images. Received {image_dir}") 
    
    if os.path.exists(mask_dir) and not os.path.isdir(mask_dir):
        raise ValueError(f"The value passed to the parameter mask_dir must be the path to an exiting. Received {mask_dir}")
    elif not os.path.exists(mask_dir):
        raise ValueError(f"The value passed to the paremeter mask_dir must be the path to a directory containing Carvana masks. Received {mask_dir}") 
    
    if os.path.exists(new_image_dir) and not os.path.isdir(new_image_dir):
        raise ValueError(f"The value passed to the parameter new_image_dir must be the path to an exiting or non existing directory. Received {new_image_dir}")
    elif not os.path.exists(new_image_dir):
        os.makedirs(new_image_dir)
    
    if os.path.exists(new_mask_dir) and not os.path.isdir(new_mask_dir):
        raise ValueError(f"The value passed to the parameter new_mask_dir must be the path to an exiting or non existing directory. Received {new_mask_dir}")
    elif not os.path.exists(new_mask_dir):
        os.makedirs(new_mask_dir)

    if ratio <= 0.0 or ratio >= 1.0:
        raise ValueError(f"Ratio must be a float value between 0 and 1. Received {ratio}")
    
    
    codes = [i.replace(".jpg", "") for i in os.listdir(image_dir)]
    if shuffle:
        random.shuffle(codes)

    n_elements_to_extract = int(len(codes) * ratio)
    to_extract = codes[:n_elements_to_extract]

    for code in to_extract:

        image_pathname = code + ".jpg"
        mask_pathname = code + "_mask.gif"

        shutil.copy(os.path.join(image_dir, image_pathname), os.path.join(new_image_dir, image_pathname))
        shutil.copy(os.path.join(mask_dir, mask_pathname), os.path.join(new_mask_dir, mask_pathname))

        os.remove(os.path.join(image_dir, image_pathname))
        os.remove(os.path.join(mask_dir, mask_pathname))



# Evalua el rendimiento de un modelo para un problema de segmentación semantica binaria
# Asume que el dataset devuelve como ground truth una mascara binaria 2D y que el modelo da como predicciones los logits
def evaluate(dataset, model, loss_fn, device):
    # Inicializar metricas a cero
    loss = acc = dice = 0
    # Poner el modelo en modo evaluacion
    model.eval()
    # Evita el calculo de los gradientes
    with torch.no_grad():
        # Itera sobre el dataset
        for x, y in dataset:
            # Mover los tensores al device elegido
            x, y = x.to(device), y.to(device)
            # Calcular predicciones y descartar la dimension vacía
            preds = model(x).squeeze(1)
            # Carcular función de coste y actualizar variable
            loss += loss_fn(preds, y)
            # Conversion a probabilidades
            preds = torch.sigmoid(preds)
            # Binariza la probabilidad predicha para cada pixel
            preds = (preds > 0.5).float()
            # Calcular accuracy y actualizar variable
            acc += (preds == y).sum() / torch.numel(preds)
            # Calcular DICE y actualizar variable
            dice += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-6)
    # Cambiar el modelo a modo entrenamiento
    model.train()
    # Normalizar metricas por la longitud del dataloader
    loss = loss / len(dataset)
    acc = acc / len(dataset)
    dice = dice / len(dataset)
    return loss, acc, dice



def savePredicionsAsComposedImage(dataset, 
                                  model, 
                                  device, 
                                  dir=os.path.join("dataset", "test", "predictions"),
                                  n_samples=10,
                                  ):
    # Crea el directorio 'dir' si no existe
    if not os.path.exists(dir):
        os.makedirs(dir)
    to_pil = torchvision.transforms.ToPILImage()
    model = model.to(device)
    # Pasa el modelo a modo evaluacion
    model.eval()
    # Evitar el calculo de gradientes
    with torch.no_grad():
        # Iterar sobre el dataset
        counter = 0
        for x, y in dataset:
            # Mover los tensores al device
            x, y = x.to(device), y
            preds = (torch.sigmoid(model(x).squeeze(1)) > 0.5).float()
            for i in range(x.size(0)):
                concatenated = torch.cat([x[i], y[i].repeat(3, 1, 1).to(device), preds[i].repeat(3, 1, 1).to(device)], dim=2)
                image = to_pil(concatenated)
                image.save(os.path.join(dir, f"{counter}.jpg"))
                counter += 1
                if counter == n_samples:
                    model.train()
                    return None
    # Pasa el modelo a modo entrenamiento
    model.train()