import torch
import os
import numpy as np
from PIL import Image

class cityscapesDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, scale=False, to_torch_tensor=False):
        self.image_dir = image_dir # Ruta del directorio que contiene las imagenes de entrada
        self.mask_dir = mask_dir # Ruta del directorio que contiene las mascaras de salida
        self.codes = [i.replace(".png", "") for i in os.listdir(image_dir)] # Lista que contiene los codigos identificadores de cada imagen
        self.transform = transform
        self.scale = scale
        self.to_torch_tensor = to_torch_tensor

    def __len__(self): # Devuelve el numero de muestras del dataset (una por cada codigo identificador de la imagen)
        return len(self.codes)

    def __getitem__(self, index): # Importación y preprocesado de una muestra
        code = self.codes[index]
        img_path = os.path.join(self.image_dir, code+".png") # Ruta de la imagen de entrada
        mask_path = os.path.join(self.mask_dir, code+"cleaned_labels.png") # Ruta de la mascara
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        if self.scale:
            image = image / 255.0
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = np.transpose(image, (2,0,1))
            self.to_torch_tensor = True
        if self.to_torch_tensor:
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)        
        return image, mask