import os
import torch
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
from PIL import Image
import keras

# Cargar modelo
model = keras.saving.load_model('modelo_regresion.keras')
print("Modelo cargado")

# Cargar y preprocesar imagen
img = Image.open('data/9.jpeg')
img_crop = img.crop((1000, 1100, 1512, 1612))  # Mismo ROI
img_array = np.array(img_crop, dtype=np.float32) / 255.0
img_input = np.expand_dims(img_array, axis=0)  # Batch dimension

# Inferencia
prediccion = model.predict(img_input, verbose=0)[0]

# Mostrar resultados
print(f"Biomasa_estimad_g_L_(Spirulina_maxima): {prediccion[0]:.4f}")
print(f"Oxigeno_disuelto_DO_mg_L_estimado: {prediccion[1]:.4f}")