import cv2
import os
from ultralytics import YOLO

# --- 1. CONFIGURACIÓN INICIAL ---

# Crear la carpeta de salida si no existe
output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Carga del modelo YOLOv8.
model = YOLO('models/yolov8x_21sp_5364img.pt')

# Ruta del archivo de video de entrada.
video_path = 'data/test.mp4'
cap = cv2.VideoCapture(video_path)

# Obtener propiedades del video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = 0  # Contador para nombrar los frames de salida

# Configuración del escritor de video de salida (Mantengo el VideoWriter para el flujo de datos)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# Puedes comentar la siguiente línea si SOLO quieres guardar frames y no el video completo:
out = cv2.VideoWriter('output_tracking.mp4', fourcc, fps, (frame_width, frame_height))

# --- 2. PROCESAMIENTO DEL VIDEO ---

while cap.isOpened():
    success, frame = cap.read()

    if success:
        frame_count += 1

        # Realizar la detección y seguimiento en el frame.
        results = model.track(
            frame,
            persist=True,
            tracker='bytetrack.yaml',
            # classes=[15],  # Clase 'fish' en COCO.
            conf=0.1,
            verbose=False  # Suprime los mensajes en la línea de comandos
        )

        # Obtener el frame con las cajas delimitadoras y los IDs de seguimiento dibujados.
        annotated_frame = results[0].plot()

        # Guardar el frame anotado en la carpeta 'output'
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, annotated_frame)

        # Escribir el frame anotado en el archivo de video (si no lo quieres, comenta esta línea)
        out.write(annotated_frame)

        # Mostrar el frame en una ventana de OpenCV
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # El final del video o un error de lectura
        break

# --- 3. LIMPIEZA Y LIBERACIÓN DE RECURSOS ---

cap.release()
out.release()
cv2.destroyAllWindows()