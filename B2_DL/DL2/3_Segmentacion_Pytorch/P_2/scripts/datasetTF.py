# Author: Antonio Martínez González
import os
from PIL import Image
import tensorflow as tf
import numpy as np
import random

class carvanaDatasetTF(tf.keras.utils.Sequence):
    def __init__(self, image_dir, mask_dir, transform=None, scale=False, batch_size=4, shuffle=True, to_tf_tensor=False,):
        self.image_dir = image_dir # Ruta del directorio que contiene las imagenes de entrada
        self.mask_dir = mask_dir # Ruta del directorio que contiene las mascaras de salida
        self.codes = [i.replace(".jpg", "") for i in os.listdir(image_dir)] # Lista que contiene los codigos de cada imagen
        self.transform = transform
        self.scale = scale
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.to_tf_tensor = to_tf_tensor

    def __len__(self): # Devuelve el numero de batches
        return len(self.codes) // self.batch_size

    def __getitem__(self, index): # Importación y preprocesado de un batcch
        inbatch = []
        outbatch = []
        codes = self.codes[index*self.batch_size:(index+1)*self.batch_size]
        for code in codes:
            img_path = os.path.join(self.image_dir, code+".jpg") # Ruta de la imagen de entrada
            mask_path = os.path.join(self.mask_dir, code+"_mask.gif") # Ruta de la mascara
            image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask[mask == 255.0] = 1.0
            if self.scale:
                image = image / 255.0
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            inbatch.append(image)
            outbatch.append(mask) 
        inbatch, outbatch = np.array(inbatch), np.array(outbatch)
        if self.to_tf_tensor:
            inbatch, outbatch = tf.convert_to_tensor(inbatch), tf.convert_to_tensor(outbatch)
        return inbatch, outbatch
    
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.codes)