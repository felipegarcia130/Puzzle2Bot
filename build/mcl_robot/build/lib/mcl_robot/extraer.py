import numpy as np
import cv2

occ = np.load('/home/felipe/ros2_ws/src/mcl_robot/maps/slam_map.npy')
print(f'Shape: {occ.shape}')

# Muestra el npy RAW sin transformaciones
vis = np.where(occ == 0, 255, np.where(occ == 100, 0, 128)).astype(np.uint8)
vis = cv2.resize(vis, (540, 540), interpolation=cv2.INTER_NEAREST)
cv2.imshow('RAW - sin transformaciones', vis)
cv2.waitKey(0)