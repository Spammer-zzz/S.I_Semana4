import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen en escala de grises y color
imagen = cv2.imread('snk1.png', cv2.IMREAD_GRAYSCALE)
imagen_color = cv2.imread('snk1.png')
imagen_color = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2RGB)

# Definir los filtros y tamaños de vecindad a comparar
filtros = {
    'Original': imagen,
    'Media 3x3': cv2.blur(imagen, (3, 3)),
    'Media 7x7': cv2.blur(imagen, (7, 7)),
    'Mediana 3x3': cv2.medianBlur(imagen, 3),
    'Mediana 7x7': cv2.medianBlur(imagen, 7)
}

fig, axs = plt.subplots(len(filtros), 2, figsize=(10, 14))
fig.suptitle('Comparación de filtrados y vecindad en Harris y Hough')

for idx, (nombre, img_filtrada) in enumerate(filtros.items()):
    # Harris
    img_color_harris = cv2.imread('snk1.png')
    img_color_harris = cv2.cvtColor(img_color_harris, cv2.COLOR_BGR2RGB)
    esquinas = cv2.cornerHarris(np.float32(img_filtrada), 2, 3, 0.04)
    img_color_harris[esquinas > 0.01 * esquinas.max()] = [255, 0, 0]
    axs[idx, 0].imshow(img_color_harris)
    axs[idx, 0].set_title(f'Harris - {nombre}')
    axs[idx, 0].axis('off')
    # Hough
    bordes = cv2.Canny(img_filtrada, 50, 150)
    lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    img_color_hough = cv2.imread('snk1.png')
    img_color_hough = cv2.cvtColor(img_color_hough, cv2.COLOR_BGR2RGB)
    if lineas is not None:
        for linea in lineas:
            x1, y1, x2, y2 = linea[0]
            cv2.line(img_color_hough, (x1, y1), (x2, y2), (0, 255, 0), 2)
    axs[idx, 1].imshow(img_color_hough)
    axs[idx, 1].set_title(f'Hough - {nombre}')
    axs[idx, 1].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()