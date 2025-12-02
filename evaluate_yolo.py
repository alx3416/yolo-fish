import cv2
import os
from ultralytics import YOLO
import glob

# --- 1. CONFIGURACIÓN INICIAL ---

# Crear la carpeta de salida si no existe
output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Carga del modelo YOLOv8.
model = YOLO('models/yolov8x_21sp_5364img.pt')

# Ruta de la carpeta con las imágenes de entrada
images_path = 'data/23sp_4120img_34945annots_2688res/train/images'

# Obtener lista de todas las imágenes jpg en la carpeta
image_files = sorted(glob.glob(os.path.join(images_path, '*.jpg')))

print(f"Total de imágenes encontradas: {len(image_files)}")

# Contador para nombrar los frames de salida
frame_count = 0

# --- 2. PROCESAMIENTO DE LAS IMÁGENES ---

for image_path in image_files:
    # Leer la imagen
    frame = cv2.imread(image_path)

    if frame is not None:
        frame_count += 1

        # Mostrar progreso cada 100 imágenes
        if frame_count % 100 == 0:
            print(f"Procesando imagen {frame_count}/{len(image_files)}...")

        # Realizar la detección y seguimiento en el frame.
        results = model.track(
            frame,
            persist=True,
            tracker='bytetrack.yaml',
            # classes=[15],  # Clase 'fish' en COCO.
            conf=0.4,
            verbose=False  # Suprime los mensajes en la línea de comandos
        )

        # Obtener el frame con las cajas delimitadoras y los IDs de seguimiento dibujados.
        annotated_frame = results[0].plot()

        # Guardar el frame anotado en la carpeta 'output'
        frame_filename = os.path.join(output_folder, f"yolo_frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, annotated_frame)

        # Mostrar el frame en una ventana de OpenCV
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Procesamiento interrumpido por el usuario.")
            break
    else:
        print(f"Error al leer la imagen: {image_path}")

# --- 3. LIMPIEZA Y LIBERACIÓN DE RECURSOS ---

cv2.destroyAllWindows()
print(f"Procesamiento completado. {frame_count} imágenes procesadas.")
print(f"Imágenes anotadas guardadas en la carpeta: {output_folder}")