import cv2
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped

N          = 200 #500 si se quiere mejor
MAP_X_MIN  = -4.0
MAP_X_MAX  =  5.0
MAP_Y_MIN  = -3.0
MAP_Y_MAX  =  6.0
RESOLUTION =  0.05

KLD_EPS   = 0.05
KLD_Z     = 2.326
KLD_BIN_M = 0.5
KLD_BIN_R = 0.2
N_MIN     = 30 #50 si se quiere mejor
N_MAX     = N

P_NOISE   = 0.8

# ── Umbrales de confianza MCL → EKF ──────────────────────────────────────────
# Cuando convergencia < CONV_LOW  → EKF toma el control
# Cuando convergencia > CONV_HIGH → MCL recupera el control
CONV_LOW  = 0.25
CONV_HIGH = 0.55
# Scans consecutivos en bajo antes de ceder el control al EKF
SCANS_TO_FAILOVER = 5


# ═════════════════════════════════════════════════════════════════════════════
# Extended Kalman Filter — backup de odometría
#
# Estado: x = [x_mundo, y_mundo, theta]^T
#
# Motion model (predicción):
#   x_k = f(x_{k-1}, u) + w,   w ~ N(0, Q)
#   f([x,y,θ], [d_trans, d_rot]) = [x + d_trans·cos(θ),
#                                    y + d_trans·sin(θ),
#                                    θ + d_rot          ]
#
# Observation model (actualización con pose MCL):
#   z_k = H x_k + v,   v ~ N(0, R)
#   H = I₃  (observamos directamente el estado)
#
# Cuando MCL está confiable → actualiza EKF con pose MCL (corrección).
# Cuando MCL falla          → EKF solo predice con odometría (sin corrección).
# Así el EKF nunca se "pierde" aunque el LiDAR no vea el mapa correcto.
# ═════════════════════════════════════════════════════════════════════════════
class EKFBackup:
    def __init__(self):
        # Estado inicial [x, y, theta] en coordenadas mundo
        self.x = np.zeros(3, dtype=np.float64)

        # Covarianza inicial — incertidumbre alta al inicio
        self.P = np.diag([4.0, 4.0, (math.pi)**2])

        # ── Ruido de proceso Q (cuánto confiamos en odometría) ─────────────
        # Valores de la tesis de Izmir (adaptados a PuzzleBot):
        # - ruido translacional:  std ≈ 5 cm / m recorrido → 0.05²
        # - ruido rotacional:     std ≈ 2° / rad girado   → 0.035²
        self.Q_trans = 0.05 ** 2   # var por metro recorrido
        self.Q_rot   = 0.035 ** 2  # var por radian girado

        # ── Ruido de observación R (cuánto confiamos en la pose MCL) ───────
        # Cuando MCL está convergido su std es ~5 cm → R=0.05²
        # Cuando está en transición usamos R más grande
        self.R_base = np.diag([0.05**2, 0.05**2, (0.05)**2])

        # Jacobiano de observación H = I (observamos el estado directamente)
        self.H = np.eye(3)

        self.initialized = False

    # ─────────────────────────────────────────────────────────────────────────
    # Paso de PREDICCIÓN — siempre se ejecuta con odometría
    # ─────────────────────────────────────────────────────────────────────────
    def predict(self, d_trans, d_rot):
        """
        Propaga el estado con el modelo cinemático del robot diferencial.
        Calcula el Jacobiano F = ∂f/∂x para propagar la covarianza.
        """
        if not self.initialized:
            return

        theta = self.x[2]

        # f(x, u): modelo no-lineal
        self.x[0] += d_trans * math.cos(theta)
        self.x[1] += d_trans * math.sin(theta)
        self.x[2] += d_rot
        self.x[2]  = self._wrap_angle(self.x[2])

        # Jacobiano F = ∂f/∂x (linealización alrededor del estado actual)
        # f = [x + d·cos(θ), y + d·sin(θ), θ + dθ]
        # ∂f/∂x = [[1, 0, -d·sin(θ)],
        #           [0, 1,  d·cos(θ)],
        #           [0, 0,  1       ]]
        F = np.array([
            [1.0, 0.0, -d_trans * math.sin(theta)],
            [0.0, 1.0,  d_trans * math.cos(theta)],
            [0.0, 0.0,  1.0                       ]
        ])

        # Ruido de proceso Q proporcional al movimiento
        # (misma lógica que los alphas del motion model AMCL)
        q_t = self.Q_trans * d_trans**2 + self.Q_rot * d_rot**2
        Q = np.diag([q_t, q_t, self.Q_rot * d_rot**2 + 1e-6])

        # P = F P F^T + Q
        self.P = F @ self.P @ F.T + Q

    # ─────────────────────────────────────────────────────────────────────────
    # Paso de ACTUALIZACIÓN — solo cuando MCL es confiable
    # ─────────────────────────────────────────────────────────────────────────
    def update(self, z_pose, convergencia):
        """
        Corrige el estado con la pose estimada por MCL.

        z_pose      : [x, y, theta] de MCL
        convergencia: valor [0,1] — escala el ruido de observación R.
                      Cuando MCL está muy convergido → R pequeño (confiamos más).
                      Cuando está en transición → R grande (confiamos menos).
        """
        if not self.initialized:
            # Primera observación: inicializar con la pose MCL
            self.x[:] = z_pose
            self.initialized = True
            return

        # R adaptativo: más grande cuando MCL tiene menor convergencia
        # convergencia ∈ [CONV_HIGH, 1.0] → factor ∈ [1, 10]
        r_factor = 1.0 + 9.0 * (1.0 - convergencia)
        R = self.R_base * r_factor

        # Innovación: diferencia entre observación y predicción
        y  = z_pose - self.H @ self.x
        y[2] = self._wrap_angle(y[2])  # normalizar ángulo

        # Ganancia de Kalman: K = P H^T (H P H^T + R)^{-1}
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Actualizar estado y covarianza
        self.x = self.x + K @ y
        self.x[2] = self._wrap_angle(self.x[2])

        # P = (I - KH) P  — forma estándar
        I_KH   = np.eye(3) - K @ self.H
        self.P = I_KH @ self.P

    # ─────────────────────────────────────────────────────────────────────────
    def reinit(self, pose_mcl, convergencia):
        """
        Reinicializa el EKF con la pose actual de MCL.
        Se llama cuando MCL recupera confianza tras un fallo.
        """
        self.x[:] = pose_mcl
        # Covarianza proporcional a la incertidumbre residual del MCL
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
        """Desviación estándar marginal de [x, y, theta]."""
        return np.sqrt(np.diag(self.P))


# ═════════════════════════════════════════════════════════════════════════════
# RAM y Bandit (sin cambios)
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
        eta          = self.n ** (-self.gamma)
        alpha_batch  = accepted_mask.mean()
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
    CONFIGS = [(0.15, 1), (0.30, 2), (0.50, 2), (0.30, 2)] #[(0.15, 1), (0.30, 1), (0.50, 1), (0.30, 2)] si se quiere mejor

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
        V     = self.variances
        T     = self.counts
        log_t = np.log(self.t)
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
# Ray casting
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

    oob     = (f_idx < 0) | (f_idx >= H) | (c_idx < 0) | (c_idx >= W)
    f_safe  = np.clip(f_idx, 0, H - 1)
    c_safe  = np.clip(c_idx, 0, W - 1)
    hit     = oob | (grid[f_safe, c_safe] == 0)

    first   = np.argmax(hit, axis=2)
    no_hit  = ~hit.any(axis=2)
    dist    = steps[first]
    dist[no_hit] = max_range
    return dist.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Nodo principal
# ═════════════════════════════════════════════════════════════════════════════
class MCLNode(Node):

    # Estados del sistema de control
    STATE_MCL  = 'MCL'    # MCL es la fuente principal
    STATE_EKF  = 'EKF'    # EKF es backup, MCL perdido
    STATE_SYNC = 'SYNC'   # MCL recuperándose, fusionando con EKF

    def __init__(self):
        super().__init__('mcl_node')

        # ── Mapa ──────────────────────────────────────────────────────────────
        occ = np.load('/home/felipe/ros2_ws/src/mcl_robot/maps/slam_map.npy')
        occ = np.flipud(occ)
        self.grid = (occ == 0).astype(np.uint8)
        H, W = self.grid.shape
        self.res = RESOLUTION

        self.celdas_libres = np.argwhere(self.grid == 1)
        self.get_logger().info(f'Mapa: {W}×{H} | {len(self.celdas_libres)} celdas libres')

        # ── Partículas ────────────────────────────────────────────────────────
        self.particulas     = self._sample_uniform(N)
        self.pesos_actuales = np.ones(N, dtype=np.float32) / N

        # ── Módulos adaptativos ───────────────────────────────────────────────
        self.ram    = RobustAdaptiveMotionModel(dim=3, acc_target=0.35)
        self.bandit = BanditSensorSelector()

        # ── EKF backup ────────────────────────────────────────────────────────
        self.ekf = EKFBackup()

        # ── Máquina de estados MCL ↔ EKF ─────────────────────────────────────
        self.estado          = self.STATE_MCL
        self.scans_bajo      = 0    # contador de scans con baja convergencia
        self.convergencia    = 0.0  # último valor calculado

        # ── Cache ─────────────────────────────────────────────────────────────
        self.max_range_px = None
        self.angles_full  = None
        self.current_arm  = 0
        self.scan_count   = 0 #se quita si se quiere mejro

        # ── ROS ───────────────────────────────────────────────────────────────
        self.pub_map  = self.create_publisher(OccupancyGrid, '/map',      1)
        self.pub_pose = self.create_publisher(PoseStamped,   '/mcl_pose', 10)
        self.create_timer(2.0, self.publish_map)

        self.pose_anterior = None
        self.sub_odom = self.create_subscription(Odometry,  '/odom', self.odom_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.get_logger().info('MCL + EKF backup iniciado')

    # ─────────────────────────────────────────────────────────────────────────
    # Utilidades
    # ─────────────────────────────────────────────────────────────────────────
    def _sample_uniform(self, n):
        idx  = np.random.choice(len(self.celdas_libres), n, replace=True)
        f    = self.celdas_libres[idx, 0].astype(np.float32)
        c    = self.celdas_libres[idx, 1].astype(np.float32)
        ang  = np.random.uniform(-math.pi, math.pi, n).astype(np.float32)
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
                idx[i] = j; i += 1
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
        """Convierte pose mundo → (fila, col) en píxeles."""
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
    # Odometría → predice MCL (RAM) y EKF simultáneamente
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

        if abs(dx) < 0.001 and abs(dy) < 0.001 and abs(dtheta) < 0.001:
            return

        d_trans = math.sqrt(dx**2 + dy**2)
        d_rot   = dtheta

        # ── EKF predice SIEMPRE (con o sin LiDAR) ─────────────────────────
        # Esta es la clave: el EKF nunca se detiene, siempre integra odometría.
        # Aunque el mapa tenga obstáculos no mapeados, la odometría sigue
        # siendo válida para estimar la trayectoria relativa.
        self.ekf.predict(d_trans, d_rot)

        # ── MCL propaga partículas solo si está activo ─────────────────────
        if self.estado in (self.STATE_MCL, self.STATE_SYNC):
            n             = len(self.particulas)
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
            # En modo EKF: reinicializar partículas alrededor de la pose EKF
            # para que cuando MCL recupere, parta de un punto razonable.
            # Esto evita que las partículas vaguen arbitrariamente mientras
            # el EKF está al mando.
            ekf_pose  = self.ekf.pose
            ekf_std   = self.ekf.std
            fila_ekf, col_ekf = self._world_to_grid(ekf_pose[0], ekf_pose[1])

            # Dispersar partículas alrededor de la estimación EKF
            # con radio proporcional a la incertidumbre del EKF
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
    # Scan → sensor model + máquina de estados MCL ↔ EKF
    # ─────────────────────────────────────────────────────────────────────────
    def scan_callback(self, msg):
        #esto se quita si se quiere mejorar
        self.scan_count += 1          
        if self.scan_count % 2 != 0: 
            return
        ranges_raw  = np.array(msg.ranges, dtype=np.float32)
        ranges_raw  = np.where(np.isfinite(ranges_raw), ranges_raw, msg.range_max)
        ranges_full = np.clip(ranges_raw, msg.range_min, msg.range_max)

        if self.angles_full is None:
            self.angles_full  = np.arange(
                msg.angle_min, msg.angle_max, msg.angle_increment, dtype=np.float32
            )
            self.max_range_px = int(msg.range_max / self.res)

        # Bandit selecciona configuración del sensor model
        arm             = self.bandit.select()
        sigma_hit, skip = BanditSensorSelector.CONFIGS[arm]
        self.current_arm = arm

        angles = self.angles_full[::skip]
        ranges = ranges_full[::skip]

        # Ray casting sobre las partículas actuales
        sims        = ray_casting_vectorized(
            self.particulas, self.grid, angles, self.max_range_px, step=3 #step 2 si se quiere mejor
        )
        sims_metros = sims * self.res

        # Beam model log-sum-exp
        diff          = sims_metros - ranges
        log_gauss     = (-0.5 * diff**2 / sigma_hit**2
                         - np.log(sigma_hit * math.sqrt(2 * math.pi)))
        log_uniforme  = -math.log(msg.range_max)
        lgm = math.log(P_NOISE)       + log_gauss
        lum = math.log(1.0 - P_NOISE) + log_uniforme
        lmx = np.maximum(lgm, lum)
        log_p_ray  = lmx + np.log(np.exp(lgm - lmx) + np.exp(lum - lmx))
        log_pesos  = np.sum(log_p_ray, axis=1)
        log_pesos -= log_pesos.max()
        pesos       = np.exp(log_pesos)
        peso_sum    = pesos.sum()

        if peso_sum < 1e-10:
            self.get_logger().warn('Pesos degenerados — reiniciando MCL')
            ekf_pose = self.ekf.pose
            f_ekf, c_ekf = self._world_to_grid(ekf_pose[0], ekf_pose[1])
            # Reiniciar alrededor del EKF, no uniformemente
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

        # Entropia → convergencia
        entropia     = float(-np.sum(pesos * np.log(pesos + 1e-10)))
        entropia_max = math.log(len(pesos))
        self.convergencia = 1.0 - entropia / entropia_max

        # Reward para bandit
        self.bandit.update(arm, -entropia)

        # ── Pose MCL actual ────────────────────────────────────────────────
        pose_mcl = self._pose_from_particulas()

        # ══════════════════════════════════════════════════════════════════
        # MÁQUINA DE ESTADOS: MCL ↔ EKF
        # ══════════════════════════════════════════════════════════════════
        estado_anterior = self.estado

        if self.estado == self.STATE_MCL:
            # ── Estado normal: MCL activo ──────────────────────────────────
            if self.convergencia < CONV_LOW:
                self.scans_bajo += 1
                if self.scans_bajo >= SCANS_TO_FAILOVER:
                    # MCL perdió confianza → ceder control al EKF
                    self.estado     = self.STATE_EKF
                    self.scans_bajo = 0
                    self.get_logger().warn(
                        f'⚠ MCL→EKF: conv={self.convergencia:.2f} < {CONV_LOW} '
                        f'por {SCANS_TO_FAILOVER} scans consecutivos. '
                        f'EKF toma control. Posible obstáculo no mapeado.'
                    )
            else:
                self.scans_bajo = 0
                # EKF se actualiza con pose MCL (corrección)
                self.ekf.update(pose_mcl, self.convergencia)

        elif self.estado == self.STATE_EKF:
            # ── Backup EKF activo ─────────────────────────────────────────
            # No actualizamos EKF con MCL (MCL no es confiable).
            # EKF solo predice con odometría (ya hecho en odom_callback).
            if self.convergencia > CONV_HIGH:
                # MCL empieza a recuperarse → entrar en modo sincronización
                self.estado = self.STATE_SYNC
                self.get_logger().info(
                    f'↑ EKF→SYNC: conv={self.convergencia:.2f} > {CONV_HIGH}. '
                    f'MCL recuperándose, fusionando con EKF.'
                )

        elif self.estado == self.STATE_SYNC:
            # ── Sincronización: MCL se recuperó, fusionar con EKF ─────────
            if self.convergencia > CONV_HIGH:
                # Fusión ponderada: mezclar pose MCL con EKF según convergencia
                # Cuanto más alta la convergencia, más peso al MCL
                alpha    = (self.convergencia - CONV_HIGH) / (1.0 - CONV_HIGH)
                alpha    = min(alpha, 1.0)
                pose_ekf = self.ekf.pose
                pose_fus = alpha * pose_mcl + (1.0 - alpha) * pose_ekf
                # Ángulo: media circular ponderada
                pose_fus[2] = math.atan2(
                    alpha * math.sin(pose_mcl[2]) + (1-alpha) * math.sin(pose_ekf[2]),
                    alpha * math.cos(pose_mcl[2]) + (1-alpha) * math.cos(pose_ekf[2])
                )

                # Actualizar EKF con pose fusionada
                self.ekf.update(pose_fus, self.convergencia)

                if self.convergencia > 0.75:
                    # MCL completamente recuperado → volver a estado normal
                    self.ekf.reinit(pose_mcl, self.convergencia)
                    self.estado = self.STATE_MCL
                    self.get_logger().info(
                        f'✓ SYNC→MCL: conv={self.convergencia:.2f}. '
                        f'MCL recuperado, EKF reiniciado con pose MCL.'
                    )
            else:
                # MCL volvió a caer durante sincronización → volver al EKF
                self.estado = self.STATE_EKF
                self.get_logger().warn(
                    f'↓ SYNC→EKF: conv={self.convergencia:.2f} cayó de nuevo.'
                )

        # ── Elegir pose a publicar según el estado ─────────────────────────
        if self.estado == self.STATE_MCL:
            pose_pub = pose_mcl
        elif self.estado == self.STATE_EKF:
            pose_pub = self.ekf.pose
        else:  # SYNC: pose fusionada ya calculada arriba
            pose_pub = pose_fus if 'pose_fus' in dir() else pose_mcl

        # ── Resampleo (solo cuando MCL está activo o sincronizando) ───────
        if self.estado in (self.STATE_MCL, self.STATE_SYNC):
            n_usar   = self._kld_n_particles(self.particulas)
            n_random = max(10, int(n_usar * 0.05))
            n_local  = n_usar - n_random

            idx              = self._systematic_resample(pesos, n_local)
            particulas_local = self.particulas[idx].copy()
            particulas_local[:, 0] += np.random.randn(n_local).astype(np.float32) * 0.3
            particulas_local[:, 1] += np.random.randn(n_local).astype(np.float32) * 0.3
            particulas_local[:, 2] += np.random.randn(n_local).astype(np.float32) * 0.01
            particulas_local[:, 0]  = np.clip(particulas_local[:, 0], 0, self.grid.shape[0]-1)
            particulas_local[:, 1]  = np.clip(particulas_local[:, 1], 0, self.grid.shape[1]-1)

            n_unif = n_random // 2
            n_rec  = n_random - n_unif
            particulas_rand = np.vstack([
                self._sample_uniform(n_unif),
                self._recovery_particles(n_rec, pesos)
            ]).astype(np.float32)

            self.particulas = np.vstack(
                [particulas_local, particulas_rand]
            ).astype(np.float32)

        # ── Publicar pose ─────────────────────────────────────────────────
        pose_msg = PoseStamped()
        pose_msg.header.stamp    = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = float(pose_pub[0])
        pose_msg.pose.position.y = float(pose_pub[1])
        ryaw = float(pose_pub[2])
        pose_msg.pose.orientation.z = math.sin(ryaw / 2)
        pose_msg.pose.orientation.w = math.cos(ryaw / 2)
        self.pub_pose.publish(pose_msg)

        # ── Log ───────────────────────────────────────────────────────────
        ekf_std  = self.ekf.std
        ram_info = self.ram.alphas_equiv()
        self.get_logger().info(
            f'[{self.estado}] '
            f'x={pose_pub[0]:.2f} y={pose_pub[1]:.2f} yaw={math.degrees(ryaw):.0f}° | '
            f'conv={self.convergencia:.2f} N={len(self.particulas)}(kld={self._kld_n_particles(self.particulas)}) | '
            f'EKF σ=[{ekf_std[0]:.3f},{ekf_std[1]:.3f},{ekf_std[2]:.3f}] | '
            f'RAM acc={ram_info["acc_rate"]:.2f} | '
            f'Bandit arm={arm}(σ={sigma_hit})'
        )

        # ── Visualización ─────────────────────────────────────────────────
        mapa_vis = cv2.cvtColor((self.grid * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)

        # Partículas MCL en rojo
        pts = self.particulas[:, [1, 0]].astype(np.int32)
        pts_clip = pts[
            (pts[:, 0] >= 0) & (pts[:, 0] < self.grid.shape[1]) &
            (pts[:, 1] >= 0) & (pts[:, 1] < self.grid.shape[0])
        ]
        mapa_vis[pts_clip[:, 1], pts_clip[:, 0]] = [0, 0, 255]

        # Estado en esquina
        color_estado = {
            self.STATE_MCL : (0, 200, 0),
            self.STATE_EKF : (0, 120, 255),
            self.STATE_SYNC: (200, 200, 0),
        }[self.estado]
        cv2.rectangle(mapa_vis, (0, 0), (80, 16), color_estado, -1)
        cv2.putText(mapa_vis, self.estado, (2, 12),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1)

        mapa_vis = cv2.resize(mapa_vis, (540, 540), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('MCL + EKF backup', mapa_vis)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = MCLNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
  