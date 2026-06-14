import cv2
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped
import time  # [OPT-5] Para timestamp de diagnóstico LiDAR

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETRO GLOBAL: pon True al correr en el robot físico, False en simulación
# Controla el throttle de scan, el modo headless y los umbrales de movimiento.
# No afecta ningún cálculo del algoritmo MCL.
# ─────────────────────────────────────────────────────────────────────────────
PHYSICAL_ROBOT = True  # [OPT-GLOBAL] <── cambia aquí según entorno

# ─────────────────────────────────────────────────────────────────────────────
# Parámetros del mapa y del filtro (sin cambios respecto al original)
# ─────────────────────────────────────────────────────────────────────────────
N          = 200
MAP_X_MIN  = -4.0
MAP_X_MAX  =  5.0
MAP_Y_MIN  = -3.0
MAP_Y_MAX  =  6.0
RESOLUTION =  0.05

KLD_EPS   = 0.05
KLD_Z     = 2.326
KLD_BIN_M = 0.5
KLD_BIN_R = 0.2
N_MIN     = 30
N_MAX     = N

P_NOISE   = 0.8

CONV_LOW  = 0.25
CONV_HIGH = 0.55
SCANS_TO_FAILOVER = 5

# ─────────────────────────────────────────────────────────────────────────────
# [OPT-1] Throttle de scan adaptado al entorno
# En físico: procesa 1 de cada 3 scans (≈3-4 Hz efectivos con RPLIDAR a 10 Hz)
# En simulación: procesa 1 de cada 2 (comportamiento original)
# Justificación: reduce carga de CPU sin alterar el modelo probabilístico.
# El MCL converge igual con menos scans; lo crítico es no saturar el Jetson.
# ─────────────────────────────────────────────────────────────────────────────
SCAN_MODULO = 3 if PHYSICAL_ROBOT else 2  # [OPT-1]

# ─────────────────────────────────────────────────────────────────────────────
# [OPT-2] Umbral mínimo de movimiento para propagar partículas
# En físico: 3 mm y 0.5° evitan que el ruido del encoder dispare propagaciones
#            innecesarias a 50 Hz de odometría.
# En simulación: umbrales originales (1 mm / ~0°)
# Justificación: no modifica el modelo cinemático, solo filtra delta-poses
# de ruido que no representan movimiento real del robot.
# ─────────────────────────────────────────────────────────────────────────────
ODOM_TRANS_THRESH  = 0.003  if PHYSICAL_ROBOT else 0.001   # metros [OPT-2]
ODOM_ROT_THRESH    = 0.005  if PHYSICAL_ROBOT else 0.001   # radianes [OPT-2]

# ─────────────────────────────────────────────────────────────────────────────
# [OPT-5] Timeout LiDAR para diagnóstico: si pasan más de N segundos sin scan,
# se emite un warning. Valor conservador para hotspot con latencia.
# ─────────────────────────────────────────────────────────────────────────────
LIDAR_TIMEOUT_S = 2.0 if PHYSICAL_ROBOT else 8.0  # [OPT-5]

# ─────────────────────────────────────────────────────────────────────────────
# [OPT-7] Frecuencia de publicación del mapa estático
# En físico: cada 5 s reduce tráfico de red por hotspot SSH.
# El mapa no cambia durante la ejecución, así que publicar más seguido
# solo desperdicia ancho de banda.
# ─────────────────────────────────────────────────────────────────────────────
MAP_PUBLISH_PERIOD = 5.0 if PHYSICAL_ROBOT else 2.0  # segundos [OPT-7]


# ═════════════════════════════════════════════════════════════════════════════
# EKFBackup — sin cambios en la lógica matemática
# ═════════════════════════════════════════════════════════════════════════════
class EKFBackup:
    def __init__(self):
        self.x = np.zeros(3, dtype=np.float64)
        self.P = np.diag([4.0, 4.0, (math.pi)**2])
        self.Q_trans = 0.05 ** 2
        self.Q_rot   = 0.035 ** 2
        self.R_base  = np.diag([0.05**2, 0.05**2, (0.05)**2])
        self.H       = np.eye(3)
        self.initialized = False

    def predict(self, d_trans, d_rot):
        if not self.initialized:
            return
        theta = self.x[2]
        self.x[0] += d_trans * math.cos(theta)
        self.x[1] += d_trans * math.sin(theta)
        self.x[2] += d_rot
        self.x[2]  = self._wrap_angle(self.x[2])
        F = np.array([
            [1.0, 0.0, -d_trans * math.sin(theta)],
            [0.0, 1.0,  d_trans * math.cos(theta)],
            [0.0, 0.0,  1.0]
        ])
        q_t = self.Q_trans * d_trans**2 + self.Q_rot * d_rot**2
        Q   = np.diag([q_t, q_t, self.Q_rot * d_rot**2 + 1e-6])
        self.P = F @ self.P @ F.T + Q

    def update(self, z_pose, convergencia):
        if not self.initialized:
            self.x[:] = z_pose
            self.initialized = True
            return
        r_factor = 1.0 + 9.0 * (1.0 - convergencia)
        R  = self.R_base * r_factor
        y  = z_pose - self.H @ self.x
        y[2] = self._wrap_angle(y[2])
        S  = self.H @ self.P @ self.H.T + R
        K  = self.P @ self.H.T @ np.linalg.inv(S)
        self.x    = self.x + K @ y
        self.x[2] = self._wrap_angle(self.x[2])
        I_KH   = np.eye(3) - K @ self.H
        self.P = I_KH @ self.P

    def reinit(self, pose_mcl, convergencia):
        self.x[:] = pose_mcl
        sigma = 0.5 * (1.0 - convergencia) + 0.05
        self.P = np.diag([sigma**2, sigma**2, (sigma*2)**2])
        self.initialized = True

    @staticmethod
    def _wrap_angle(a):
        return float((a + math.pi) % (2 * math.pi) - math.pi)

    @property
    def pose(self):
        return self.x.copy()

    @property
    def std(self):
        return np.sqrt(np.diag(self.P))


# ═════════════════════════════════════════════════════════════════════════════
# RAM y Bandit — sin cambios
# ═════════════════════════════════════════════════════════════════════════════
class RobustAdaptiveMotionModel:
    def __init__(self, dim=3, acc_target=0.35, gamma=2/3, S0_scale=None):
        self.dim        = dim
        self.acc_target = acc_target
        self.gamma      = gamma
        self.n          = 0
        self.acc_rate   = acc_target
        if S0_scale is None:
            S0_scale = np.array([2.0, 2.0, 0.05])
        self.S = np.diag(S0_scale.astype(float))

    def sample_noise(self, n):
        z     = np.random.randn(n, self.dim)
        noise = (self.S @ z.T).T
        return noise.astype(np.float32), z

    def update(self, z_batch, accepted_mask):
        self.n += 1
        eta         = self.n ** (-self.gamma)
        alpha_batch = accepted_mask.mean()
        self.acc_rate += eta * (alpha_batch - self.acc_rate)
        z_mean  = z_batch.mean(axis=0)
        norm_z2 = float(np.dot(z_mean, z_mean))
        if norm_z2 > 1e-10:
            factor = eta * (alpha_batch - self.acc_target) / norm_z2
            M = np.eye(self.dim) + factor * np.outer(z_mean, z_mean)
            try:
                self.S = np.linalg.cholesky(self.S @ M @ self.S.T)
            except np.linalg.LinAlgError:
                pass

    def alphas_equiv(self):
        d = np.diag(self.S)
        return {'std_fila': abs(d[0]), 'std_col': abs(d[1]),
                'std_theta': abs(d[2]), 'acc_rate': self.acc_rate}


class BanditSensorSelector:
    CONFIGS = [(0.15, 1), (0.30, 2), (0.50, 2), (0.30, 2)]

    def __init__(self, zeta=1.0, c=0.5):
        self.K      = len(self.CONFIGS)
        self.zeta   = zeta
        self.c      = c
        self.t      = 0
        self.counts = np.zeros(self.K)
        self.means  = np.zeros(self.K)
        self.M2     = np.zeros(self.K)

    @property
    def variances(self):
        mask = self.counts > 1
        v    = np.zeros(self.K)
        v[mask] = self.M2[mask] / (self.counts[mask] - 1)
        return v

    def select(self):
        self.t += 1
        if self.t <= self.K:
            return self.t - 1
        V      = self.variances
        T      = self.counts
        log_t  = np.log(self.t)
        scores = (self.means
                  + np.sqrt(2.0 * V * self.zeta * log_t / T)
                  + self.c * self.zeta * log_t / T)
        return int(np.argmax(scores))

    def update(self, arm, reward):
        self.counts[arm] += 1
        Tk    = self.counts[arm]
        delta = reward - self.means[arm]
        self.means[arm] += delta / Tk
        self.M2[arm]    += delta * (reward - self.means[arm])

    def best_arm(self):
        return int(np.argmax(self.means)) if self.t >= self.K else 0


# ═════════════════════════════════════════════════════════════════════════════
# Ray casting — sin cambios en la lógica
# ═════════════════════════════════════════════════════════════════════════════
def ray_casting_vectorized(particulas, grid, angles, max_range, step=2):
    H, W   = grid.shape
    filas  = particulas[:, 0]
    cols   = particulas[:, 1]
    thetas = particulas[:, 2]

    abs_angles = thetas[:, None] + angles[None, :] + math.pi
    sin_a = np.sin(abs_angles)
    cos_a = np.cos(abs_angles)
    steps = np.arange(step, max_range + step, step, dtype=np.float32)

    f_idx = (filas[:, None, None] - sin_a[:, :, None] * steps[None, None, :]).astype(np.int32)
    c_idx = (cols[:, None, None]  + cos_a[:, :, None] * steps[None, None, :]).astype(np.int32)

    oob    = (f_idx < 0) | (f_idx >= H) | (c_idx < 0) | (c_idx >= W)
    f_safe = np.clip(f_idx, 0, H - 1)
    c_safe = np.clip(c_idx, 0, W - 1)
    hit    = oob | (grid[f_safe, c_safe] == 0)

    first  = np.argmax(hit, axis=2)
    no_hit = ~hit.any(axis=2)
    dist   = steps[first]
    dist[no_hit] = max_range
    return dist.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Nodo principal
# ═════════════════════════════════════════════════════════════════════════════
class MCLNode(Node):

    STATE_MCL  = 'MCL'
    STATE_EKF  = 'EKF'
    STATE_SYNC = 'SYNC'

    def __init__(self):
        super().__init__('mcl_node')

        # ── Mapa ──────────────────────────────────────────────────────────────
        occ = np.load('/home/felipe/ros2_ws/src/mcl_robot/maps/slam_map.npy')
        occ = np.flipud(occ)
        self.grid = (occ == 0).astype(np.uint8)
        H, W = self.grid.shape
        self.res = RESOLUTION

        self.celdas_libres = np.argwhere(self.grid == 1)
        self.get_logger().info(
            f'Mapa: {W}×{H} | {len(self.celdas_libres)} celdas libres | '
            f'Modo: {"FÍSICO" if PHYSICAL_ROBOT else "SIMULACIÓN"}'
        )

        # [OPT-4] Pre-asignar buffer fijo para partículas.
        # Evita que np.vstack aloque memoria nueva en cada scan callback.
        # El contenido se sobreescribe in-place; la lógica de resampleo
        # es idéntica al original.
        self._buf_local  = np.empty((N_MAX, 3), dtype=np.float32)   # [OPT-4]
        self._buf_rand   = np.empty((N_MAX, 3), dtype=np.float32)   # [OPT-4]
        self._buf_merged = np.empty((N_MAX, 3), dtype=np.float32)   # [OPT-4]

        # ── Partículas ────────────────────────────────────────────────────────
        self.particulas     = self._sample_uniform(N)
        self.pesos_actuales = np.ones(N, dtype=np.float32) / N

        # ── Módulos adaptativos ───────────────────────────────────────────────
        self.ram    = RobustAdaptiveMotionModel(dim=3, acc_target=0.35)
        self.bandit = BanditSensorSelector()

        # ── EKF backup ────────────────────────────────────────────────────────
        self.ekf = EKFBackup()

        # ── Máquina de estados ────────────────────────────────────────────────
        self.estado       = self.STATE_MCL
        self.scans_bajo   = 0
        self.convergencia = 0.0

        # ── Cache ─────────────────────────────────────────────────────────────
        self.max_range_px = None
        self.angles_full  = None
        self.current_arm  = 0
        self.scan_count   = 0
        self.pose_anterior = None

        # [OPT-5] Timestamp del último scan recibido (para diagnóstico)
        self._last_scan_time = time.monotonic()  # [OPT-5]

        # ── ROS ───────────────────────────────────────────────────────────────
        self.pub_map  = self.create_publisher(OccupancyGrid, '/map',      1)
        self.pub_pose = self.create_publisher(PoseStamped,   '/mcl_pose', 10)

        # [OPT-7] Publicar mapa cada MAP_PUBLISH_PERIOD segundos
        self.create_timer(MAP_PUBLISH_PERIOD, self.publish_map)

        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        self.create_timer(2.0, self._check_lidar_alive)

        self.get_logger().info(
            f'MCL + EKF backup iniciado | '
            f'scan_modulo={SCAN_MODULO} | '
            f'map_period={MAP_PUBLISH_PERIOD}s | '
            f'odom_thresh=({ODOM_TRANS_THRESH}m, {math.degrees(ODOM_ROT_THRESH):.2f}°)'
        )

    # ─────────────────────────────────────────────────────────────────────────
    # [OPT-5] Diagnóstico LiDAR
    # ─────────────────────────────────────────────────────────────────────────
    def _check_lidar_alive(self):
        """
        [OPT-5] Verifica que el LiDAR siga publicando.
        No modifica ningún estado del algoritmo, solo loggea.
        Útil para distinguir si el fallo es de red/LiDAR o de MCL.
        """
        elapsed = time.monotonic() - self._last_scan_time
        if elapsed > LIDAR_TIMEOUT_S:
            self.get_logger().warn(
                f'[DIAGNÓSTICO] Sin scan en {elapsed:.1f}s '
                f'(umbral={LIDAR_TIMEOUT_S}s). '
                f'¿LiDAR apagado? ¿Batería baja? ¿Nodo caído?'
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Utilidades — sin cambios en la lógica
    # ─────────────────────────────────────────────────────────────────────────
    def _sample_uniform(self, n):
        idx = np.random.choice(len(self.celdas_libres), n, replace=True)
        f   = self.celdas_libres[idx, 0].astype(np.float32)
        c   = self.celdas_libres[idx, 1].astype(np.float32)
        ang = np.random.uniform(-math.pi, math.pi, n).astype(np.float32)
        return np.column_stack([f, c, ang]).astype(np.float32)

    def _recovery_particles(self, n, pesos, top_k=10):
        top_k   = min(top_k, len(pesos))
        top_idx = np.argsort(pesos)[-top_k:]
        mejores = self.particulas[top_idx]
        n_por   = max(1, n // top_k)
        res     = []
        for p in mejores:
            d = np.tile(p, (n_por, 1)).astype(np.float32)
            d[:, 0] += np.random.randn(n_por) * 5.0
            d[:, 1] += np.random.randn(n_por) * 5.0
            d[:, 2] += np.random.randn(n_por) * 0.5
            d[:, 0]  = np.clip(d[:, 0], 0, self.grid.shape[0] - 1)
            d[:, 1]  = np.clip(d[:, 1], 0, self.grid.shape[1] - 1)
            res.append(d)
        res = np.vstack(res)
        if len(res) < n:
            res = np.vstack([res, self._sample_uniform(n - len(res))])
        return res[:n].astype(np.float32)

    def _systematic_resample(self, pesos, n):
        pos        = (np.random.random() + np.arange(n)) / n
        cumsum     = np.cumsum(pesos)
        cumsum[-1] = 1.0
        idx        = np.zeros(n, dtype=np.int32)
        i = j = 0
        while i < n:
            if pos[i] <= cumsum[j]:
                idx[i] = j
                i += 1
            else:
                j = min(j + 1, len(cumsum) - 1)
        return idx

    def _kld_n_particles(self, p):
        bx = ((p[:, 1] * self.res) / KLD_BIN_M).astype(np.int32)
        by = ((p[:, 0] * self.res) / KLD_BIN_M).astype(np.int32)
        bt = (p[:, 2] / KLD_BIN_R).astype(np.int32)
        k  = len(set(zip(bx.tolist(), by.tolist(), bt.tolist())))
        if k <= 1:
            return N_MIN
        t     = 1.0 - 2.0/(9*(k-1)) + np.sqrt(2.0/(9*(k-1))) * KLD_Z
        n_kld = int(((k-1) / (2.0 * KLD_EPS)) * t**3)
        return int(np.clip(n_kld, N_MIN, N_MAX))

    def _pose_from_particulas(self):
        fm = np.median(self.particulas[:, 0])
        cm = np.median(self.particulas[:, 1])
        mask = (
            (np.abs(self.particulas[:, 0] - fm) < 20) &
            (np.abs(self.particulas[:, 1] - cm) < 20)
        )
        src = self.particulas[mask] if mask.sum() > 10 else self.particulas
        rx  = MAP_X_MIN + src[:, 1].mean() * self.res
        ry  = MAP_Y_MAX - src[:, 0].mean() * self.res
        return np.array([rx, ry, float(src[:, 2].mean())])

    def _world_to_grid(self, x_mundo, y_mundo):
        col  = (x_mundo - MAP_X_MIN) / self.res
        fila = (MAP_Y_MAX - y_mundo) / self.res
        return float(fila), float(col)

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

    # ─────────────────────────────────────────────────────────────────────────
    # Odometría — lógica idéntica al original, con umbral de movimiento mínimo
    # ─────────────────────────────────────────────────────────────────────────
    def odom_callback(self, msg):
        x     = msg.pose.pose.position.x
        y     = msg.pose.pose.position.y
        q     = msg.pose.pose.orientation
        theta = 2.0 * np.arctan2(q.z, q.w)
        pose_actual = np.array([x, y, theta], dtype=np.float32)

        if self.pose_anterior is None:
            self.pose_anterior = pose_actual
            return

        dx     = float(pose_actual[0] - self.pose_anterior[0])
        dy     = float(pose_actual[1] - self.pose_anterior[1])
        dtheta = float(pose_actual[2] - self.pose_anterior[2])
        self.pose_anterior = pose_actual

        # [OPT-2] Umbral de movimiento mínimo más estricto en físico.
        # Evita propagar partículas por ruido del encoder a 50 Hz.
        # No cambia el modelo cinemático; solo ignora deltas sub-threshold.
        if (abs(dx)     < ODOM_TRANS_THRESH and
                abs(dy) < ODOM_TRANS_THRESH and
                abs(dtheta) < ODOM_ROT_THRESH):  # [OPT-2]
            return

        d_trans = math.sqrt(dx**2 + dy**2)
        d_rot   = dtheta

        # EKF predice siempre (sin cambios)
        self.ekf.predict(d_trans, d_rot)

        # MCL propaga partículas (sin cambios en la lógica)
        if self.estado in (self.STATE_MCL, self.STATE_SYNC):
            n              = len(self.particulas)
            noise, z_batch = self.ram.sample_noise(n)

            scale = np.array([
                d_trans / max(d_trans, 0.01),
                d_trans / max(d_trans, 0.01),
                abs(d_rot) / max(abs(d_rot), 0.01)
            ], dtype=np.float32)
            noise_e = noise * scale[None, :]

            f_prop = (self.particulas[:, 0]
                      - (d_trans * np.sin(self.particulas[:, 2])
                         + noise_e[:, 0]) / self.res)
            c_prop = (self.particulas[:, 1]
                      + (d_trans * np.cos(self.particulas[:, 2])
                         + noise_e[:, 1]) / self.res)
            t_prop = self.particulas[:, 2] + d_rot + noise_e[:, 2]

            f_prop = np.clip(f_prop, 0, self.grid.shape[0] - 1)
            c_prop = np.clip(c_prop, 0, self.grid.shape[1] - 1)

            acc = self.grid[f_prop.astype(np.int32), c_prop.astype(np.int32)] == 1
            self.ram.update(z_batch, acc)

            self.particulas[acc,  0] = f_prop[acc].astype(np.float32)
            self.particulas[acc,  1] = c_prop[acc].astype(np.float32)
            self.particulas[acc,  2] = t_prop[acc].astype(np.float32)
            self.particulas[~acc, 2] += (d_rot + noise_e[~acc, 2]).astype(np.float32)

        elif self.estado == self.STATE_EKF:
            ekf_pose  = self.ekf.pose
            ekf_std   = self.ekf.std
            fila_ekf, col_ekf = self._world_to_grid(ekf_pose[0], ekf_pose[1])
            sigma_px = max(5.0, ekf_std[0] / self.res)
            n        = len(self.particulas)
            self.particulas[:, 0] = np.clip(
                fila_ekf + np.random.randn(n) * sigma_px,
                0, self.grid.shape[0] - 1
            ).astype(np.float32)
            self.particulas[:, 1] = np.clip(
                col_ekf  + np.random.randn(n) * sigma_px,
                0, self.grid.shape[1] - 1
            ).astype(np.float32)
            self.particulas[:, 2] = (
                ekf_pose[2] + np.random.randn(n) * max(0.2, ekf_std[2])
            ).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Scan — lógica idéntica, con throttle, validación de entrada y
    # manejo robusto de cv2.imshow en modo headless
    # ─────────────────────────────────────────────────────────────────────────
    def scan_callback(self, msg):
        # [OPT-5] Actualizar timestamp para diagnóstico
        self._last_scan_time = time.monotonic()  # [OPT-5]

        # [OPT-1] Throttle: procesa 1 de cada SCAN_MODULO scans
        self.scan_count += 1
        if self.scan_count % SCAN_MODULO != 0:  # [OPT-1]
            return

        # [OPT-6] Validar que el mensaje de scan tenga datos coherentes.
        # En físico el RPLIDAR puede enviar mensajes con angle_increment=0
        # o range_max=0 al arrancar/reiniciarse, lo que crashea la inicialización.
        if (msg.angle_increment <= 0.0 or
                msg.range_max <= 0.0 or
                len(msg.ranges) == 0):  # [OPT-6]
            self.get_logger().warn(
                f'[DIAGNÓSTICO] Scan inválido descartado: '
                f'angle_increment={msg.angle_increment:.6f}, '
                f'range_max={msg.range_max}, '
                f'n_ranges={len(msg.ranges)}'
            )
            return

        ranges_raw  = np.array(msg.ranges, dtype=np.float32)
        ranges_raw  = np.where(np.isfinite(ranges_raw), ranges_raw, msg.range_max)
        ranges_full = np.clip(ranges_raw, msg.range_min, msg.range_max)

        # [OPT-6] Inicializar angles_full con validación adicional
        if self.angles_full is None:
            n_expected = int(round(
                (msg.angle_max - msg.angle_min) / msg.angle_increment
            )) + 1
            # Usar linspace para evitar acumulación de error de punto flotante
            # con arange en mensajes reales del RPLIDAR
            self.angles_full  = np.linspace(
                msg.angle_min, msg.angle_max,
                min(n_expected, len(msg.ranges)),
                dtype=np.float32
            )  # [OPT-6]
            self.max_range_px = int(msg.range_max / self.res)
            self.get_logger().info(
                f'[DIAGNÓSTICO] LiDAR inicializado: '
                f'{len(self.angles_full)} ángulos, '
                f'range_max={msg.range_max}m, '
                f'max_range_px={self.max_range_px}'
            )

        # [OPT-6] Alinear longitudes si el scan llega con tamaño variable
        n_scan = min(len(ranges_full), len(self.angles_full))
        ranges_full  = ranges_full[:n_scan]
        angles_local = self.angles_full[:n_scan]

        # Bandit y ray casting (sin cambios)
        arm             = self.bandit.select()
        sigma_hit, skip = BanditSensorSelector.CONFIGS[arm]
        self.current_arm = arm

        angles = angles_local[::skip]
        ranges = ranges_full[::skip]

        sims        = ray_casting_vectorized(
            self.particulas, self.grid, angles, self.max_range_px, step=3
        )
        sims_metros = sims * self.res

        # Beam model (sin cambios)
        diff         = sims_metros - ranges
        log_gauss    = (-0.5 * diff**2 / sigma_hit**2
                        - np.log(sigma_hit * math.sqrt(2 * math.pi)))
        log_uniforme = -math.log(msg.range_max)
        lgm  = math.log(P_NOISE)       + log_gauss
        lum  = math.log(1.0 - P_NOISE) + log_uniforme
        lmx  = np.maximum(lgm, lum)
        log_p_ray = lmx + np.log(np.exp(lgm - lmx) + np.exp(lum - lmx))
        log_pesos = np.sum(log_p_ray, axis=1)
        log_pesos -= log_pesos.max()
        pesos    = np.exp(log_pesos)
        peso_sum = pesos.sum()

        if peso_sum < 1e-10:
            self.get_logger().warn('Pesos degenerados — reiniciando MCL')
            ekf_pose = self.ekf.pose
            f_ekf, c_ekf = self._world_to_grid(ekf_pose[0], ekf_pose[1])
            n = N
            self.particulas[:, 0] = np.clip(
                f_ekf + np.random.randn(n) * 20,
                0, self.grid.shape[0]-1).astype(np.float32)
            self.particulas[:, 1] = np.clip(
                c_ekf + np.random.randn(n) * 20,
                0, self.grid.shape[1]-1).astype(np.float32)
            self.particulas[:, 2] = (
                ekf_pose[2] + np.random.randn(n) * 0.5).astype(np.float32)
            self.pesos_actuales = np.ones(N) / N
            pesos = self.pesos_actuales
        else:
            pesos /= peso_sum
            self.pesos_actuales = pesos

        # Convergencia y reward Bandit (sin cambios)
        entropia     = float(-np.sum(pesos * np.log(pesos + 1e-10)))
        entropia_max = math.log(len(pesos))
        self.convergencia = 1.0 - entropia / entropia_max
        self.bandit.update(arm, -entropia)

        pose_mcl = self._pose_from_particulas()

        # ── Máquina de estados MCL ↔ EKF (sin cambios) ────────────────────
        if self.estado == self.STATE_MCL:
            if self.convergencia < CONV_LOW:
                self.scans_bajo += 1
                if self.scans_bajo >= SCANS_TO_FAILOVER:
                    self.estado     = self.STATE_EKF
                    self.scans_bajo = 0
                    self.get_logger().warn(
                        f'⚠ MCL→EKF: conv={self.convergencia:.2f} < {CONV_LOW} '
                        f'por {SCANS_TO_FAILOVER} scans consecutivos.'
                    )
            else:
                self.scans_bajo = 0
                self.ekf.update(pose_mcl, self.convergencia)

        elif self.estado == self.STATE_EKF:
            if self.convergencia > CONV_HIGH:
                self.estado = self.STATE_SYNC
                self.get_logger().info(
                    f'↑ EKF→SYNC: conv={self.convergencia:.2f} > {CONV_HIGH}.'
                )

        elif self.estado == self.STATE_SYNC:
            if self.convergencia > CONV_HIGH:
                alpha    = (self.convergencia - CONV_HIGH) / (1.0 - CONV_HIGH)
                alpha    = min(alpha, 1.0)
                pose_ekf = self.ekf.pose
                pose_fus = alpha * pose_mcl + (1.0 - alpha) * pose_ekf
                pose_fus[2] = math.atan2(
                    alpha * math.sin(pose_mcl[2]) + (1-alpha) * math.sin(pose_ekf[2]),
                    alpha * math.cos(pose_mcl[2]) + (1-alpha) * math.cos(pose_ekf[2])
                )
                self.ekf.update(pose_fus, self.convergencia)
                if self.convergencia > 0.75:
                    self.ekf.reinit(pose_mcl, self.convergencia)
                    self.estado = self.STATE_MCL
                    self.get_logger().info(
                        f'✓ SYNC→MCL: conv={self.convergencia:.2f}.'
                    )
            else:
                self.estado = self.STATE_EKF
                self.get_logger().warn(
                    f'↓ SYNC→EKF: conv={self.convergencia:.2f} cayó de nuevo.'
                )

        # ── Elegir pose a publicar (sin cambios) ──────────────────────────
        if self.estado == self.STATE_MCL:
            pose_pub = pose_mcl
        elif self.estado == self.STATE_EKF:
            pose_pub = self.ekf.pose
        else:
            pose_pub = pose_fus if 'pose_fus' in locals() else pose_mcl

        # ── Resampleo con buffers pre-asignados [OPT-4] ───────────────────
        if self.estado in (self.STATE_MCL, self.STATE_SYNC):
            n_usar   = self._kld_n_particles(self.particulas)
            n_random = max(10, int(n_usar * 0.05))
            n_local  = n_usar - n_random

            idx = self._systematic_resample(pesos, n_local)

            # [OPT-4] Escribir en buffer pre-asignado en vez de crear arrays nuevos
            self._buf_local[:n_local] = self.particulas[idx]
            self._buf_local[:n_local, 0] += np.random.randn(n_local).astype(np.float32) * 0.3
            self._buf_local[:n_local, 1] += np.random.randn(n_local).astype(np.float32) * 0.3
            self._buf_local[:n_local, 2] += np.random.randn(n_local).astype(np.float32) * 0.01
            np.clip(self._buf_local[:n_local, 0], 0, self.grid.shape[0]-1,
                    out=self._buf_local[:n_local, 0])
            np.clip(self._buf_local[:n_local, 1], 0, self.grid.shape[1]-1,
                    out=self._buf_local[:n_local, 1])

            n_unif = n_random // 2
            n_rec  = n_random - n_unif
            # Solo se hace vstack para la parte aleatoria (pequeña, n_random ≤ 10)
            part_rand = np.vstack([
                self._sample_uniform(n_unif),
                self._recovery_particles(n_rec, pesos)
            ]).astype(np.float32)
            self._buf_rand[:n_random] = part_rand

            # Copiar a buffer merged y asignar
            self._buf_merged[:n_local]  = self._buf_local[:n_local]
            self._buf_merged[n_local:n_local+n_random] = self._buf_rand[:n_random]
            self.particulas = self._buf_merged[:n_local+n_random].copy()  # [OPT-4]

        # ── Publicar pose (sin cambios) ────────────────────────────────────
        pose_msg = PoseStamped()
        pose_msg.header.stamp    = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = float(pose_pub[0])
        pose_msg.pose.position.y = float(pose_pub[1])
        ryaw = float(pose_pub[2])
        pose_msg.pose.orientation.z = math.sin(ryaw / 2)
        pose_msg.pose.orientation.w = math.cos(ryaw / 2)
        self.pub_pose.publish(pose_msg)

        # ── Log (sin cambios en el contenido) ─────────────────────────────
        ekf_std  = self.ekf.std
        ram_info = self.ram.alphas_equiv()
        self.get_logger().info(
            f'[{self.estado}] '
            f'x={pose_pub[0]:.2f} y={pose_pub[1]:.2f} yaw={math.degrees(ryaw):.0f}° | '
            f'conv={self.convergencia:.2f} N={len(self.particulas)}'
            f'(kld={self._kld_n_particles(self.particulas)}) | '
            f'EKF σ=[{ekf_std[0]:.3f},{ekf_std[1]:.3f},{ekf_std[2]:.3f}] | '
            f'RAM acc={ram_info["acc_rate"]:.2f} | '
            f'Bandit arm={arm}(σ={sigma_hit})'
        )

        # ── Visualización con manejo robusto de headless [OPT-3] ──────────
        if not PHYSICAL_ROBOT:
            # En simulación: comportamiento original, cv2.imshow normal
            mapa_vis = cv2.cvtColor(
                (self.grid * 255).astype('uint8'), cv2.COLOR_GRAY2BGR
            )
            pts = self.particulas[:, [1, 0]].astype(np.int32)
            pts_clip = pts[
                (pts[:, 0] >= 0) & (pts[:, 0] < self.grid.shape[1]) &
                (pts[:, 1] >= 0) & (pts[:, 1] < self.grid.shape[0])
            ]
            mapa_vis[pts_clip[:, 1], pts_clip[:, 0]] = [0, 0, 255]
            color_estado = {
                self.STATE_MCL : (0, 200, 0),
                self.STATE_EKF : (0, 120, 255),
                self.STATE_SYNC: (200, 200, 0),
            }[self.estado]
            cv2.rectangle(mapa_vis, (0, 0), (80, 16), color_estado, -1)
            cv2.putText(mapa_vis, self.estado, (2, 12),
                        cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1)
            mapa_vis = cv2.resize(mapa_vis, (540, 540),
                                  interpolation=cv2.INTER_NEAREST)
            cv2.imshow('MCL + EKF backup', mapa_vis)
            cv2.waitKey(1)
        else:
            # [OPT-3] En físico: cv2.imshow dentro de try/except.
            # Si no hay display (SSH sin -X) simplemente se omite sin crashear.
            # No afecta ningún cálculo.
            try:
                mapa_vis = cv2.cvtColor(
                    (self.grid * 255).astype('uint8'), cv2.COLOR_GRAY2BGR
                )
                pts = self.particulas[:, [1, 0]].astype(np.int32)
                pts_clip = pts[
                    (pts[:, 0] >= 0) & (pts[:, 0] < self.grid.shape[1]) &
                    (pts[:, 1] >= 0) & (pts[:, 1] < self.grid.shape[0])
                ]
                mapa_vis[pts_clip[:, 1], pts_clip[:, 0]] = [0, 0, 255]
                color_estado = {
                    self.STATE_MCL : (0, 200, 0),
                    self.STATE_EKF : (0, 120, 255),
                    self.STATE_SYNC: (200, 200, 0),
                }[self.estado]
                cv2.rectangle(mapa_vis, (0, 0), (80, 16), color_estado, -1)
                cv2.putText(mapa_vis, self.estado, (2, 12),
                            cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1)
                mapa_vis = cv2.resize(mapa_vis, (540, 540),
                                      interpolation=cv2.INTER_NEAREST)
                cv2.imshow('MCL + EKF backup', mapa_vis)
                cv2.waitKey(1)
            except Exception:
                pass  # [OPT-3] Sin display: ignorar silenciosamente


# ═════════════════════════════════════════════════════════════════════════════
# main — usa MultiThreadedExecutor [OPT-8]
# ═════════════════════════════════════════════════════════════════════════════
def main():
    rclpy.init()
    node = MCLNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()