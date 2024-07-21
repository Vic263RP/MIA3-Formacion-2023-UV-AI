import torch

# callback early stopping para terminar el entrenamiento cuando la metrica en el conjunto de validacion se estanca, empeora o no mejora lo suficiente conforme aumentan las epocas
class earlyStopping:
    def __init__(self, patience=3, min_delta=0.0, mode="min"):
        """
        Args:
            patience (int): El entrenamiento para si en 'patience' numero de epocas la metrica de monitorizacion no ha mejorado
            min_delta (float): Cambio minimo en valor absoluto en la metrica de monitorizacion para considerar el resultado como una mejora respecto de la epoca anterior
            mode (str): Uno de los valores 'max' o 'min'. Determina si el objetivo es minimizar o maximizar la metrica de monitorizacion
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0 # inicializar el contador de epocas en la que la metrica no mejora
        self.stop = False # boolean flag para que determina cuando ha de parar el entrenamiento
        self.mode = mode

        if self.mode not in {"min", "max"}:
            raise ValueError(f"The value passed to the parameter 'mode' must be either 'max' or 'min'. Received {self.mode}")
        
        if self.min_delta < 0.0:
            raise ValueError(f"The value passed to the parameter min_delta must be greater than or equal to zero. Received {self.min_delta}")
        
        if self.mode == "max":
            self.best_score = 0.0
        else:
            self.best_score = 1e10

    def __call__(self, new_score):

        if self.mode == "max":
            metric_improved = (new_score - self.best_score > self.min_delta)
        elif self.mode == "min":
            metric_improved = (self.best_score - new_score > self.min_delta)

        if metric_improved:
            self.best_score = new_score
            self.counter = 0
        else:
            # si la metrica no ha mejorado añadimos una unidad al contador de epocas sin mejora
            self.counter += 1
            # comprobamos si hay que parar el entrenamiento
            if self.counter == self.patience:
                self.stop = True


# callback model checkpoint para guardar el mejor modelo obtenido hasta la fecha (pesos del modelo y estado del optimizador)
class modelCheckpoint:
    def __init__(self, path, min_delta=0.0, mode="min"):
        """
        Args:
            path (str): ruta donde almacenar los pesos del modelo y el estado del optimizador
        """
        if not path:
            path = "unerCarvana.pth"
        self.path = path
        self.mode = mode
        self.min_delta = min_delta

        if self.mode not in {"min", "max"}:
            raise ValueError(f"The value passed to the parameter 'mode' must be either 'max' or 'min'. Received {self.mode}")
        
        if self.min_delta < 0.0:
            raise ValueError(f"The value passed to the parameter min_delta must be greater than or equal to zero. Received {self.min_delta}")
        
        if self.mode == "max":
            self.best_score = 0.0
        else:
            self.best_score = 1e10

    def __call__(self, model, optimizer, new_score):

        if self.mode == "max":
            metric_improved = (new_score - self.best_score > self.min_delta)
        elif self.mode == "min":
            metric_improved = (self.best_score - new_score > self.min_delta)

        if metric_improved:
            torch.save({
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                }, self.path)
            self.best_score = new_score