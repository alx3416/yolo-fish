import cv2
from ultralytics import YOLO
import torch

# --- 1. CONFIGURACIÓN INICIAL ---

# Carga del modelo YOLOv8
model = YOLO('models/yolov8x_21sp_5364img.pt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Usando dispositivo: {device}")

# Inicializar la webcam (0 = cámara por defecto)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: No se pudo abrir la webcam.")
    exit()

print("Webcam iniciada. Presiona ESC para salir.")

# --- 2. PROCESAMIENTO EN TIEMPO REAL ---

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error al leer el frame de la webcam.")
        break

    # Realizar la detección
    results = model.predict(
        frame,
        conf=0.4,
        verbose=False
    )

    # Obtener el frame con las cajas delimitadoras dibujadas
    annotated_frame = results[0].plot()

    # Mostrar el frame en una ventana de OpenCV
    cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

    # Salir si se presiona ESC
    if cv2.waitKey(1) & 0xFF == 27:
        print("Procesamiento detenido.")
        break

# --- 3. LIMPIEZA Y LIBERACIÓN DE RECURSOS ---

cap.release()
cv2.destroyAllWindows()
