import cv2
import os
from ultralytics import YOLO
import glob
import time

# --- 1. CONFIGURACIÓN INICIAL ---

# Crear la carpeta de salida si no existe
output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Carga del modelo YOLOv8
model = YOLO('models/yolov8x_21sp_5364img.pt')

# Ruta de la carpeta con las imágenes de entrada
images_path = 'data/23sp_4120img_34945annots_2688res/train/images'

# Obtener lista de todas las imágenes jpg en la carpeta
image_files = sorted(glob.glob(os.path.join(images_path, '*.jpg')))

print(f"Total de imágenes encontradas: {len(image_files)}")

# Contador para nombrar los frames de salida
frame_count = 0
start_time = time.time()

# --- 2. PROCESAMIENTO DE LAS IMÁGENES (SIN TRACKING) ---

for image_path in image_files:
    # Leer la imagen
    frame = cv2.imread(image_path)

    if frame is not None:
        frame_count += 1

        # Mostrar progreso cada 100 imágenes
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / frame_count
            remaining = (len(image_files) - frame_count) * avg_time
            print(f"Procesando imagen {frame_count}/{len(image_files)} | "
                  f"Tiempo promedio: {avg_time:.3f}s | "
                  f"Tiempo restante estimado: {remaining/60:.1f}min")

        # Realizar SOLO la detección (sin tracking)
        results = model.predict(
            frame,
            conf=0.4,
            verbose=False
        )

        # Obtener el frame con las cajas delimitadoras dibujadas
        annotated_frame = results[0].plot()

        # Guardar el frame anotado en la carpeta 'output'
        frame_filename = os.path.join(output_folder, f"yolo_frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, annotated_frame)

        # Mostrar el frame en una ventana de OpenCV
        cv2.imshow("YOLOv8 Detection", annotated_frame)

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Procesamiento interrumpido por el usuario.")
            break
    else:
        print(f"Error al leer la imagen: {image_path}")

# --- 3. LIMPIEZA Y LIBERACIÓN DE RECURSOS ---

cv2.destroyAllWindows()

# Estadísticas finales
total_time = time.time() - start_time
avg_time_per_image = total_time / frame_count if frame_count > 0 else 0

print(f"\n{'='*60}")
print(f"Procesamiento completado")
print(f"{'='*60}")
print(f"Imágenes procesadas: {frame_count}/{len(image_files)}")
print(f"Tiempo total: {total_time/60:.2f} minutos")
print(f"Tiempo promedio por imagen: {avg_time_per_image:.3f} segundos")
print(f"FPS promedio: {1/avg_time_per_image:.2f}")
print(f"Imágenes anotadas guardadas en: {output_folder}")
print(f"{'='*60}")