import os
import torch
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers, models, ops
import pandas as pd
import numpy as np
from PIL import Image

# Configurar backend PyTorch y GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando: {device}")

# Cargar datos
df = pd.read_excel('data/correlacion_clorofila_biomasa_spirulina.xlsx')
y = df[['Biomasa_estimad_g_L_(Spirulina_maxima)', 'Oxigeno_disuelto_DO_mg_L_estimado']].values

img_files = sorted([f for f in os.listdir('data/') if f.endswith(('.jpg', '.png', '.jpeg'))])
X = []

# Preprocesamiento: crop y normalización
for img_file in img_files:
    img = Image.open(f'data/{img_file}')
    img_crop = img.crop((1000, 1100, 1512, 1612))  # x, y, x+512, y+512
    img_array = np.array(img_crop, dtype=np.float32) / 255.0
    X.append(img_array)

X = np.array(X)
print(f"Shape datos: {X.shape}")

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.GaussianNoise(0.1)
])

# Modelo CNN
inputs = layers.Input(shape=(512, 512, 3))
x = data_augmentation(inputs)

# Capas convolucionales
x = layers.Conv2D(35, 3, activation='relu', padding='same')(x)
x = layers.MaxPooling2D(4)(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.MaxPooling2D(4)(x)
x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
x = layers.MaxPooling2D(4)(x)
x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
x = layers.MaxPooling2D(4)(x)
# x = layers.GlobalAveragePooling2D()(x)

# Capas densas
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)


# Salida regresión
outputs = layers.Dense(2)(x)

model = models.Model(inputs, outputs)

# Compilación
model.compile(optimizer='Adagrad', loss='mse', metrics=['mae'])
model.summary()

# Entrenamiento
history = model.fit(X, y, epochs=500, batch_size=1)

# Guardar historial de entrenamiento en CSV
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)
print("Historial de entrenamiento guardado en training_history.csv")

# Guardar modelo en formatos Keras y PyTorch
model.save('modelo_regresion.keras')
print("Modelo guardado en modelo_regresion.keras")

# Convertir a PyTorch nativo
torch_model = model  # El modelo ya es PyTorch internamente
state_dict = {}
for layer in model.layers:
    if hasattr(layer, 'weights') and len(layer.weights) > 0:
        for weight in layer.weights:
            state_dict[weight.name] = torch.from_numpy(np.array(weight))

torch.save({'model_state_dict': state_dict, 'model_config': model.get_config()}, 'modelo_regresion.pt')
print("Modelo guardado en modelo_regresion.pt")