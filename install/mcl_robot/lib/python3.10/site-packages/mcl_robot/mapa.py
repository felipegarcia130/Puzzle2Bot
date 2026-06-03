"""
pgm_to_npy.py
-------------
Convierte un mapa .pgm/.yaml de ROS2 a slam_map.npy
para usar con el nodo MCL.

Uso:
    python3 pgm_to_npy.py

Genera:
    /tmp/slam_map.npy
    Imprime las constantes MAP_X_MIN/MAX, MAP_Y_MIN/MAX a usar en mcl_node.py
"""

import numpy as np
import cv2

# ── Parámetros del yaml ───────────────────────────────────────────────────────
PGM_PATH    = '/home/felipe/puzzlebot/maps/cartographer_real.pgm'
RESOLUTION  = 0.01          # metros/pixel
ORIGIN      = [-1.07, -4.89]  # [x, y] esquina inferior izquierda del mapa
FREE_THRESH = 0.25
OCC_THRESH  = 0.65
NEGATE      = 0

OUTPUT_PATH = '/tmp/slam_map.npy'

# ── Cargar imagen ─────────────────────────────────────────────────────────────
img = cv2.imread(PGM_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f'No se encontró: {PGM_PATH}')

H, W = img.shape
print(f'Mapa cargado: {W}x{H} px')

# ── Convertir a occupancy (0=libre, 100=ocupado, -1=desconocido) ──────────────
# ROS2 map_server: valor pixel normalizado a [0,1]
# Si negate=0: prob_ocupado = 1 - pixel/255
img_float = img.astype(np.float32) / 255.0

if NEGATE == 0:
    occ_prob = 1.0 - img_float
else:
    occ_prob = img_float

occ_grid = np.full((H, W), -1, dtype=np.int8)
occ_grid[occ_prob < FREE_THRESH] = 0    # libre
occ_grid[occ_prob > OCC_THRESH]  = 100  # ocupado

# ── Guardar como npy ──────────────────────────────────────────────────────────
# El MCL espera: 0 = libre (navigable), cualquier otro = pared
# La convención del MCL usa 0=libre en grid binaria
slam_map = np.zeros((H, W), dtype=np.int8)
slam_map[occ_grid == 100] = 100   # ocupado
slam_map[occ_grid == -1]  = -1    # desconocido

np.save(OUTPUT_PATH, slam_map)
print(f'Guardado: {OUTPUT_PATH}')

# ── Calcular constantes para mcl_node.py ─────────────────────────────────────
map_x_min = ORIGIN[0]
map_x_max = ORIGIN[0] + W * RESOLUTION
map_y_min = ORIGIN[1]
map_y_max = ORIGIN[1] + H * RESOLUTION

print()
print('══ Copia estas constantes en mcl_node.py ══')
print(f'MAP_X_MIN  = {map_x_min:.2f}')
print(f'MAP_X_MAX  = {map_x_max:.2f}')
print(f'MAP_Y_MIN  = {map_y_min:.2f}')
print(f'MAP_Y_MAX  = {map_y_max:.2f}')
print(f'RESOLUTION = {RESOLUTION}')
print()
print(f'Dimensiones: {W}x{H} px  →  {W*RESOLUTION:.2f}x{H*RESOLUTION:.2f} m')