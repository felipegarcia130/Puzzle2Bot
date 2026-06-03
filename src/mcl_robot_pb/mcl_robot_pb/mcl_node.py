
"""
import cv2
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped

N          = 500   # partículas totales — suficiente para localización global
N_RANDOM   = 50    # 5% aleatorias en cada ciclo (kidnapped robot)
N_LOCAL    = N - N_RANDOM

MAP_X_MIN  = -4.0
MAP_X_MAX  =  5.0
MAP_Y_MIN  = -3.0
MAP_Y_MAX  =  6.0
RESOLUTION =  0.05


def ray_casting_vectorized(particulas, grid, angles, max_range, step=2):
    H, W = grid.shape

    filas  = particulas[:, 0]
    cols   = particulas[:, 1]
    thetas = particulas[:, 2]

    abs_angles = thetas[:, None] + angles[None, :] + math.pi
    sin_a = np.sin(abs_angles)
    cos_a = np.cos(abs_angles)

    steps = np.arange(step, max_range + step, step, dtype=np.float32)

    f_idx = (filas[:, None, None] - sin_a[:, :, None] * steps[None, None, :]).astype(np.int32)
    c_idx = (cols[:, None, None]  + cos_a[:, :, None] * steps[None, None, :]).astype(np.int32)

    out_of_bounds = (f_idx < 0) | (f_idx >= H) | (c_idx < 0) | (c_idx >= W)
    f_safe = np.clip(f_idx, 0, H - 1)
    c_safe = np.clip(c_idx, 0, W - 1)

    hit_wall = (grid[f_safe, c_safe] == 0)
    hit = out_of_bounds | hit_wall

    first_hit_idx = np.argmax(hit, axis=2)
    no_hit = ~hit.any(axis=2)

    distancias = steps[first_hit_idx]
    distancias[no_hit] = max_range

    return distancias.astype(np.float32)


class MCLNode(Node):
    def __init__(self):
        super().__init__('mcl_node')

        # ── Cargar mapa ───────────────────────────────────────────────────────
        occ = np.load('/home/felipe/ros2_ws/src/mcl_robot/maps/slam_map.npy')
        occ = np.flipud(occ)
        self.grid = (occ == 0).astype(np.uint8)

        H, W = self.grid.shape
        self.res_x = RESOLUTION
        self.res_y = RESOLUTION
        self.res   = RESOLUTION

        # Precalcular celdas libres para distribución uniforme
        self.celdas_libres = np.argwhere(self.grid == 1)  # shape (M, 2): [fila, col]
        self.get_logger().info(
            f'Mapa cargado: {W}x{H} celdas | '
            f'{len(self.celdas_libres)} celdas libres'
        )

        # Origen del (0,0) mundo en píxeles — solo de referencia
        self.origen_col  = (0.0 - MAP_X_MIN) / self.res_x   # 80
        self.origen_fila = (MAP_Y_MAX - 0.0)  / self.res_y   # 120

        # ── Inicialización global: partículas uniformes en todo el mapa ───────
        self.particulas = self._sample_uniform(N)
        self.get_logger().info(f'Partículas inicializadas uniformemente (N={N})')

        self.max_range_px = None
        self.angles_cache = None

        self.pub_map  = self.create_publisher(OccupancyGrid, '/map',      1)
        self.pub_pose = self.create_publisher(PoseStamped,   '/mcl_pose', 10)
        self.create_timer(2.0, self.publish_map)

        self.pose_anterior = None
        self.sub_odom = self.create_subscription(Odometry,  '/odom', self.odom_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

    # ── Muestreo uniforme en celdas libres ────────────────────────────────────
    def _sample_uniform(self, n):
        idx     = np.random.choice(len(self.celdas_libres), n, replace=True)
        filas   = self.celdas_libres[idx, 0].astype(np.float32)
        cols    = self.celdas_libres[idx, 1].astype(np.float32)
        angulos = np.random.uniform(-math.pi, math.pi, n).astype(np.float32)
        return np.column_stack([filas, cols, angulos]).astype(np.float32)

    # ── Conversión píxel → mundo ──────────────────────────────────────────────
    def get_pose(self):
        # Recalcular pesos para la pose (usa el último scan)
        # Por ahora usa media simple pero solo de las partículas más concentradas
        
        # Encontrar el cluster más denso con la mediana
        fila_med = np.median(self.particulas[:, 0])
        col_med  = np.median(self.particulas[:, 1])
        
        # Solo promedia partículas cerca de la mediana (±20 celdas = ±1m)
        mask = (
            (np.abs(self.particulas[:, 0] - fila_med) < 20) &
            (np.abs(self.particulas[:, 1] - col_med)  < 20)
        )
        
        if mask.sum() > 10:
            mean_fila = self.particulas[mask, 0].mean()
            mean_col  = self.particulas[mask, 1].mean()
            mean_yaw  = self.particulas[mask, 2].mean()
        else:
            mean_fila = self.particulas[:, 0].mean()
            mean_col  = self.particulas[:, 1].mean()
            mean_yaw  = self.particulas[:, 2].mean()

        robot_x = MAP_X_MIN + mean_col  * self.res_x
        robot_y = MAP_Y_MAX - mean_fila * self.res_y
        return robot_x, robot_y, float(mean_yaw)

    # ── Publicar OccupancyGrid ────────────────────────────────────────────────
    def publish_map(self):
        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.resolution = self.res
        msg.info.width      = self.grid.shape[1]
        msg.info.height     = self.grid.shape[0]
        msg.info.origin.position.x = MAP_X_MIN
        msg.info.origin.position.y = MAP_Y_MIN
        msg.info.origin.orientation.w = 1.0
        grid_ros = np.flipud(self.grid)
        msg.data = np.where(grid_ros == 1, 0, 100).flatten().tolist()
        self.pub_map.publish(msg)

    # ── Motion model ──────────────────────────────────────────────────────────
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = 2.0 * np.arctan2(q.z, q.w)
        pose_actual = np.array([x, y, theta], dtype=np.float32)

        if self.pose_anterior is None:
            self.pose_anterior = pose_actual
            return

        dx     = pose_actual[0] - self.pose_anterior[0]
        dy     = pose_actual[1] - self.pose_anterior[1]
        dtheta = pose_actual[2] - self.pose_anterior[2]
        self.pose_anterior = pose_actual


        # Ignora si es puro giro sin avance — evita dispersión por safety
        if abs(dx) < 0.001 and abs(dy) < 0.001 and abs(dtheta) < 0.001:
            return

         # NUEVO: si solo gira sin moverse, reduce el ruido angular
        solo_giro = abs(dx) < 0.005 and abs(dy) < 0.005

        noise_ang = 0.01 if solo_giro else 0.02

        self.particulas[:, 0] -= (dy / self.res_y) + np.random.randn(N).astype(np.float32) * 0.05
        self.particulas[:, 1] += (dx / self.res_x) + np.random.randn(N).astype(np.float32) * 0.05
        self.particulas[:, 2] += dtheta            + np.random.randn(N).astype(np.float32) * noise_ang
        self.particulas[:, 0]  = np.clip(self.particulas[:, 0], 0, self.grid.shape[0] - 1)
        self.particulas[:, 1]  = np.clip(self.particulas[:, 1], 0, self.grid.shape[1] - 1)

    # ── Sensor model + resampleo ──────────────────────────────────────────────
    # ── Sensor model + resampleo ──────────────────────────────────────────────
    def scan_callback(self, msg):
        ranges_raw = np.array(msg.ranges, dtype=np.float32)
        ranges_raw = np.where(np.isfinite(ranges_raw), ranges_raw, msg.range_max)
        ranges = np.clip(ranges_raw, msg.range_min, msg.range_max)[::3]

        if self.angles_cache is None:
            self.angles_cache = np.arange(
                msg.angle_min, msg.angle_max, msg.angle_increment, dtype=np.float32
            )[::3]
            self.max_range_px = int(msg.range_max / self.res)
            self.get_logger().info(
                f'Scan: {len(self.angles_cache)} rayos, max_range={self.max_range_px} px'
            )

        n_particulas = len(self.particulas)

        sims = ray_casting_vectorized(
            self.particulas, self.grid, self.angles_cache, self.max_range_px, step=2
        )

        sims_metros = sims * self.res
        sigma = 0.5
        diff  = sims_metros - ranges

        # Peso mayor a rayos cortos (más informativos)
        ray_weights = 1.0 / (ranges + 0.5)
        ray_weights /= ray_weights.mean()

        pesos = np.exp(-0.5 * np.mean((diff ** 2) * ray_weights, axis=1) / sigma ** 2)
        peso_sum = pesos.sum()

        if peso_sum < 1e-10:
            self.get_logger().warn('Pesos degenerados — reiniciando')
            self.particulas = self._sample_uniform(N)
            return

        pesos /= peso_sum
        pesos  = pesos / pesos.sum()  # segunda normalización por seguridad

        # ── Convergencia adaptativa ───────────────────────────────────────────
        entropia     = -np.sum(pesos * np.log(pesos + 1e-10))
        entropia_max = np.log(n_particulas)
        convergencia = 1.0 - (entropia / entropia_max)

        if convergencia > 0.7:
            n_usar = 300
        elif convergencia > 0.4:
            n_usar = 600
        else:
            n_usar = N

        n_random = max(10, int(n_usar * 0.05))
        n_local  = n_usar - n_random

        # ── Resampleo: n_local locales + n_random uniformes ───────────────────
        indices          = np.random.choice(n_particulas, n_local, p=pesos)
        particulas_local = self.particulas[indices]

        particulas_local[:, 0] += np.random.randn(n_local).astype(np.float32) * 0.3
        particulas_local[:, 1] += np.random.randn(n_local).astype(np.float32) * 0.3
        particulas_local[:, 2] += np.random.randn(n_local).astype(np.float32) * 0.01
        particulas_local[:, 0]  = np.clip(particulas_local[:, 0], 0, self.grid.shape[0] - 1)
        particulas_local[:, 1]  = np.clip(particulas_local[:, 1], 0, self.grid.shape[1] - 1)

        particulas_rand = self._sample_uniform(n_random)

        self.particulas = np.vstack([particulas_local, particulas_rand]).astype(np.float32)

        # ── Publicar pose ─────────────────────────────────────────────────────
        rx, ry, ryaw = self.get_pose()

        pose_msg = PoseStamped()
        pose_msg.header.stamp    = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = rx
        pose_msg.pose.position.y = ry
        pose_msg.pose.orientation.z = float(np.sin(ryaw / 2))
        pose_msg.pose.orientation.w = float(np.cos(ryaw / 2))
        self.pub_pose.publish(pose_msg)

        self.get_logger().info(
            f'Pose: x={rx:.3f} y={ry:.3f} | '
            f'n={len(self.particulas)} conv={convergencia:.2f} | '
            f'fila={self.particulas[:,0].mean():.0f} col={self.particulas[:,1].mean():.0f}'
        )

        # ── Visualización ─────────────────────────────────────────────────────
        mapa_vis = cv2.cvtColor((self.grid * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
        pts = self.particulas[:, [1, 0]].astype(np.int32)
        pts_clip = pts[
            (pts[:, 0] >= 0) & (pts[:, 0] < self.grid.shape[1]) &
            (pts[:, 1] >= 0) & (pts[:, 1] < self.grid.shape[0])
        ]
        mapa_vis[pts_clip[:, 1], pts_clip[:, 0]] = [0, 0, 255]
        mapa_vis = cv2.resize(mapa_vis, (540, 540), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('MCL', mapa_vis)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = MCLNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
"""

import cv2
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster

N = 500  # aumentado de 100 → 500

MAP_X_MIN  = -4.0
MAP_X_MAX  =  5.0
MAP_Y_MIN  = -3.0
MAP_Y_MAX  =  6.0
RESOLUTION =  0.05

MAX_RANGE_PX_CAP = 150   # 7.5 m — evita tensores enormes
SIGMA_SENSOR     = 0.15  # antes 0.5 — más discriminativo
NOISE_POST_RESAMPLE_POS = 0.3
NOISE_POST_RESAMPLE_ANG = 0.03
NOISE_MOTION_POS = 0.1
NOISE_MOTION_ANG = 0.01
ADAPTIVE_RAND_FRAC = 0.05  # fracción máxima de partículas aleatorias (AMCL ligero)


def ray_casting_vectorized(particulas, grid, angles, max_range, step=3):
    H, W = grid.shape

    filas  = particulas[:, 0]
    cols   = particulas[:, 1]
    thetas = particulas[:, 2]

    abs_angles = thetas[:, None] + angles[None, :] + math.pi
    sin_a = np.sin(abs_angles)
    cos_a = np.cos(abs_angles)

    steps = np.arange(step, max_range + step, step, dtype=np.float32)

    f_idx = (filas[:, None, None] - sin_a[:, :, None] * steps[None, None, :]).astype(np.int32)
    c_idx = (cols[:, None, None]  + cos_a[:, :, None] * steps[None, None, :]).astype(np.int32)

    out_of_bounds = (f_idx < 0) | (f_idx >= H) | (c_idx < 0) | (c_idx >= W)
    f_safe = np.clip(f_idx, 0, H - 1)
    c_safe = np.clip(c_idx, 0, W - 1)

    hit_wall = (grid[f_safe, c_safe] == 0)
    hit = out_of_bounds | hit_wall

    first_hit_idx = np.argmax(hit, axis=2)
    no_hit = ~hit.any(axis=2)

    distancias = steps[first_hit_idx]
    distancias[no_hit] = max_range

    return distancias.astype(np.float32)


def systematic_resample(N, pesos):
    """Low-variance resampling — evita impoverishment del multinomial."""
    posiciones = (np.arange(N) + np.random.uniform()) / N
    indices = np.zeros(N, dtype=int)
    acum = np.cumsum(pesos)
    i, j = 0, 0
    while i < N:
        if posiciones[i] < acum[j]:
            indices[i] = j
            i += 1
        else:
            j = min(j + 1, N - 1)
    return indices


class MCLNode(Node):
    def __init__(self):
        super().__init__('mcl_node')

        # ── Cargar mapa ───────────────────────────────────────────────────────
        occ = np.load('/home/felipe/ros2_ws/src/mcl_robot/maps/slam_map.npy')
        occ = np.flipud(occ)
        self.grid = (occ == 0).astype(np.uint8)

        H, W = self.grid.shape
        self.res = RESOLUTION

        self.origen_col  = (0.0 - MAP_X_MIN) / self.res
        self.origen_fila = (MAP_Y_MAX - 0.0)  / self.res

        self.get_logger().info(f'Mapa: {W}x{H} celdas  |  origen pixel: fila={self.origen_fila:.1f} col={self.origen_col:.1f}')

        # ── Partículas iniciales centradas en (0,0) ───────────────────────────
        self.particulas = self._init_particles(N)

        self.max_range_px  = None
        self.angles_cache  = None
        self.n_rays        = None

        # ── Publishers / TF ───────────────────────────────────────────────────
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pub_map  = self.create_publisher(OccupancyGrid, '/map',      1)
        self.pub_pose = self.create_publisher(PoseStamped,   '/mcl_pose', 10)
        self.create_timer(2.0, self.publish_map)

        self.pose_anterior = None
        self.sub_odom = self.create_subscription(Odometry,  '/odom', self.odom_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _init_particles(self, n):
        H, W = self.grid.shape
        noise_fila = np.random.randn(n).astype(np.float32) * 10.0
        noise_col  = np.random.randn(n).astype(np.float32) * 10.0
        filas   = np.clip(self.origen_fila + noise_fila, 0, H - 1)
        cols    = np.clip(self.origen_col  + noise_col,  0, W - 1)
        angulos = np.random.randn(n).astype(np.float32) * 0.2
        return np.column_stack([filas, cols, angulos]).astype(np.float32)

    def _random_particles(self, n):
        """Partículas uniformes en zonas libres del mapa (para AMCL adaptativo)."""
        H, W = self.grid.shape
        free = np.argwhere(self.grid == 1)
        idx  = np.random.choice(len(free), n)
        filas = free[idx, 0].astype(np.float32)
        cols  = free[idx, 1].astype(np.float32)
        angs  = np.random.uniform(-math.pi, math.pi, n).astype(np.float32)
        return np.column_stack([filas, cols, angs])

    def _clip_particles(self):
        H, W = self.grid.shape
        self.particulas[:, 0] = np.clip(self.particulas[:, 0], 0, H - 1)
        self.particulas[:, 1] = np.clip(self.particulas[:, 1], 0, W - 1)
        self.particulas[:, 2] = (self.particulas[:, 2] + math.pi) % (2 * math.pi) - math.pi

    # ── Pose estimada (yaw circular correcto) ─────────────────────────────────
    def get_pose(self):
        mean_fila = self.particulas[:, 0].mean()
        mean_col  = self.particulas[:, 1].mean()
        mean_yaw  = np.arctan2(
            np.mean(np.sin(self.particulas[:, 2])),
            np.mean(np.cos(self.particulas[:, 2]))
        )
        robot_x = MAP_X_MIN + mean_col  * self.res
        robot_y = MAP_Y_MAX - mean_fila * self.res
        return robot_x, robot_y, float(mean_yaw), mean_fila, mean_col

    # ── OccupancyGrid ─────────────────────────────────────────────────────────
    def publish_map(self):
        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.resolution = self.res
        msg.info.width      = self.grid.shape[1]
        msg.info.height     = self.grid.shape[0]
        msg.info.origin.position.x = MAP_X_MIN
        msg.info.origin.position.y = MAP_Y_MIN
        msg.info.origin.orientation.w = 1.0
        grid_ros = np.flipud(self.grid)
        msg.data = np.where(grid_ros == 1, 0, 100).flatten().tolist()
        self.pub_map.publish(msg)

    # ── Motion model ──────────────────────────────────────────────────────────
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        theta = 2.0 * np.arctan2(q.z, q.w)
        pose_actual = np.array([x, y, theta], dtype=np.float32)

        if self.pose_anterior is None:
            self.pose_anterior = pose_actual
            return

        dx = pose_actual[0] - self.pose_anterior[0]
        dy = pose_actual[1] - self.pose_anterior[1]
        # normalizar delta de ángulo a [-π, π]
        dtheta = float((pose_actual[2] - self.pose_anterior[2] + math.pi) % (2 * math.pi) - math.pi)
        self.pose_anterior = pose_actual

        if abs(dx) < 0.001 and abs(dy) < 0.001 and abs(dtheta) < 0.001:
            return

        n = len(self.particulas)
        self.particulas[:, 0] -= (dy / self.res) + np.random.randn(n).astype(np.float32) * NOISE_MOTION_POS
        self.particulas[:, 1] += (dx / self.res) + np.random.randn(n).astype(np.float32) * NOISE_MOTION_POS
        self.particulas[:, 2] += dtheta          + np.random.randn(n).astype(np.float32) * NOISE_MOTION_ANG
        self._clip_particles()

    # ── Sensor model + resampleo ──────────────────────────────────────────────
    def scan_callback(self, msg):
        ranges_raw = np.array(msg.ranges, dtype=np.float32)
        ranges_raw = np.where(np.isfinite(ranges_raw), ranges_raw, msg.range_max)
        ranges = np.clip(ranges_raw, msg.range_min, msg.range_max)[::5]

        if self.angles_cache is None:
            # linspace garantiza consistencia con [::5]
            all_angles = np.linspace(msg.angle_min, msg.angle_max,
                                     len(msg.ranges), dtype=np.float32)
            self.angles_cache = all_angles[::5]
            self.n_rays       = len(self.angles_cache)
            raw_max_px        = int(msg.range_max / self.res)
            self.max_range_px = min(raw_max_px, MAX_RANGE_PX_CAP)
            self.get_logger().info(
                f'Scan: {self.n_rays} rayos, max_range={self.max_range_px} px '
                f'({self.max_range_px * self.res:.1f} m)'
            )

        sims = ray_casting_vectorized(
            self.particulas, self.grid, self.angles_cache, self.max_range_px, step=3
        )

        sims_metros = sims * self.res
        diff  = sims_metros - ranges
        pesos = np.exp(-0.5 * np.mean(diff ** 2, axis=1) / SIGMA_SENSOR ** 2)
        peso_sum = pesos.sum()

        if peso_sum < 1e-10:
            self.get_logger().warn('Pesos degenerados — reiniciando partículas')
            self.particulas = self._init_particles(N)
            return

        pesos /= peso_sum

        # ── Adaptive MCL: inyectar partículas aleatorias según incertidumbre ──
        n_rand = int(N * ADAPTIVE_RAND_FRAC * (1.0 - float(pesos.max()) * N))
        n_rand = max(0, min(n_rand, int(N * ADAPTIVE_RAND_FRAC)))
        n_keep = N - n_rand

        # Systematic resampling
        indices = systematic_resample(n_keep, pesos / pesos.sum())
        kept = self.particulas[indices].copy()

        if n_rand > 0:
            rand_p = self._random_particles(n_rand)
            self.particulas = np.vstack([kept, rand_p]).astype(np.float32)
        else:
            self.particulas = kept

        # Ruido post-resampleo
        n = len(self.particulas)
        self.particulas[:, 0] += np.random.randn(n).astype(np.float32) * NOISE_POST_RESAMPLE_POS
        self.particulas[:, 1] += np.random.randn(n).astype(np.float32) * NOISE_POST_RESAMPLE_POS
        self.particulas[:, 2] += np.random.randn(n).astype(np.float32) * NOISE_POST_RESAMPLE_ANG
        self._clip_particles()

        rx, ry, ryaw, mean_fila, mean_col = self.get_pose()
        now = self.get_clock().now().to_msg()

        # ── TF map → base_link ────────────────────────────────────────────────
        t = TransformStamped()
        t.header.stamp    = now
        t.header.frame_id = 'map'
        t.child_frame_id  = 'base_link'
        t.transform.translation.x = rx
        t.transform.translation.y = ry
        t.transform.rotation.z = float(np.sin(ryaw / 2))
        t.transform.rotation.w = float(np.cos(ryaw / 2))
        self.tf_broadcaster.sendTransform(t)

        # ── PoseStamped ───────────────────────────────────────────────────────
        pose_msg = PoseStamped()
        pose_msg.header.stamp    = now
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = rx
        pose_msg.pose.position.y = ry
        pose_msg.pose.orientation.z = t.transform.rotation.z
        pose_msg.pose.orientation.w = t.transform.rotation.w
        self.pub_pose.publish(pose_msg)

        self.get_logger().info(
            f'Pose: x={rx:.3f} y={ry:.3f} yaw={math.degrees(ryaw):.1f}°  |  '
            f'partículas: fila={mean_fila:.0f} col={mean_col:.0f}  |  '
            f'peso_max={pesos.max():.4f}'
        )

        # ── Visualización ─────────────────────────────────────────────────────
        H_map, W_map = self.grid.shape
        VIS_W, VIS_H = 540, 540
        sx = VIS_W / W_map   # factor de escala col → x_vis
        sy = VIS_H / H_map   # factor de escala fila → y_vis
        ARROW_LEN = 8        # longitud de flecha en px de visualización

        mapa_vis = cv2.cvtColor((self.grid * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
        mapa_vis = cv2.resize(mapa_vis, (VIS_W, VIS_H), interpolation=cv2.INTER_NEAREST)

        # Partículas: convertir a coordenadas de visualización y dibujar flecha
        # OpenCV: x = col, y = fila  (igual que aquí)
        for p in self.particulas:
            # p[0]=fila, p[1]=col, p[2]=ángulo en conv. mapa (x=este, y=norte)
            # en imagen: eje x → este (col), eje y → sur (fila) → sin se invierte
            vx = int(p[1] * sx)
            vy = int(p[0] * sy)
            ang = float(p[2])
            tx = int(vx + math.cos(ang) * ARROW_LEN)
            ty = int(vy - math.sin(ang) * ARROW_LEN)   # -sin porque y_img crece hacia abajo
            cv2.arrowedLine(mapa_vis, (vx, vy), (tx, ty), (0, 0, 220), 1,
                            line_type=cv2.LINE_AA, tipLength=0.35)

        # Pose estimada — naranja, más grande


        cv2.imshow('MCL', mapa_vis)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = MCLNode()
    rclpy.spin(node)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()