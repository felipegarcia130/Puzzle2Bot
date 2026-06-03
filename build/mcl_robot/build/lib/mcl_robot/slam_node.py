import math
import numpy as np
import threading
from pynput import keyboard
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped

# ── Parámetros del mapa ──────────────────────────────────────────────────────
MAP_X_MIN  = -4.0
MAP_X_MAX  =  5.0
MAP_Y_MIN  = -3.0
MAP_Y_MAX  =  6.0
RESOLUTION =  0.05

# ── Parámetros MCL ───────────────────────────────────────────────────────────
N     = 300
SIGMA = 0.3

# ── Log-odds ─────────────────────────────────────────────────────────────────
L_OCC  =  0.85
L_FREE = -0.40
L_MIN  = -5.0
L_MAX  =  5.0


def world_to_grid(x, y, W, H):
    col = int((x - MAP_X_MIN) / RESOLUTION)
    row = int((MAP_Y_MAX - y) / RESOLUTION)
    col = np.clip(col, 0, W - 1)
    row = np.clip(row, 0, H - 1)
    return col, row


def bresenham(c0, r0, c1, r1):
    cells = []
    dc = abs(c1 - c0)
    dr = abs(r1 - r0)
    sc = 1 if c1 > c0 else -1
    sr = 1 if r1 > r0 else -1
    err = dc - dr
    c, r = c0, r0
    while True:
        cells.append((c, r))
        if c == c1 and r == r1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c   += sc
        if e2 < dc:
            err += dc
            r   += sr
    return cells


def update_map_with_scan(log_odds, scan_ranges, scan_angles, rx, ry, ryaw, W, H):
    abs_angles = ryaw + scan_angles
    for i, r in enumerate(scan_ranges):
        if not np.isfinite(r) or r < 0.05:
            continue
        ex = rx + r * np.cos(abs_angles[i])
        ey = ry + r * np.sin(abs_angles[i])
        c0, r0 = world_to_grid(rx, ry, W, H)
        c1, r1 = world_to_grid(ex, ey, W, H)
        free_cells = bresenham(c0, r0, c1, r1)
        for fc, fr in free_cells[:-1]:
            log_odds[fr, fc] = np.clip(log_odds[fr, fc] + L_FREE, L_MIN, L_MAX)
        log_odds[r1, c1] = np.clip(log_odds[r1, c1] + L_OCC, L_MIN, L_MAX)


def score_particles(particulas, log_odds, scan_ranges, scan_angles, W, H):
    pesos = np.zeros(len(particulas))
    mask  = np.isfinite(scan_ranges) & (scan_ranges > 0.05)
    for i, p in enumerate(particulas):
        px, py, pyaw = p
        abs_a = pyaw + scan_angles
        ex = px + scan_ranges * np.cos(abs_a)
        ey = py + scan_ranges * np.sin(abs_a)
        cols = np.clip(((ex - MAP_X_MIN) / RESOLUTION).astype(int), 0, W - 1)
        rows = np.clip(((MAP_Y_MAX - ey) / RESOLUTION).astype(int), 0, H - 1)
        pesos[i] = log_odds[rows[mask], cols[mask]].sum()
    return pesos


class SLAMNode(Node):
    def __init__(self):
        super().__init__('slam_node')

        self.W = int((MAP_X_MAX - MAP_X_MIN) / RESOLUTION)
        self.H = int((MAP_Y_MAX - MAP_Y_MIN) / RESOLUTION)
        self.log_odds = np.zeros((self.H, self.W), dtype=np.float32)
        self.get_logger().info(f'Grid: {self.W}x{self.H} celdas')

        # Partículas centradas en (0,0)
        px = np.random.randn(N).astype(np.float32) * 0.1
        py = np.random.randn(N).astype(np.float32) * 0.1
        pt = np.random.randn(N).astype(np.float32) * 0.05
        self.particulas = np.column_stack([px, py, pt]).astype(np.float32)

        self.pose_anterior = None
        self.angles_cache  = None
        self.map_built     = False

        # Pose estimada actual (media ponderada)
        self.rx   = 0.0
        self.ry   = 0.0
        self.ryaw = 0.0

        self.pub_map  = self.create_publisher(OccupancyGrid, '/map',       1)
        self.pub_pose = self.create_publisher(PoseStamped,   '/slam_pose', 10)
        self.create_timer(2.0, self.publish_map)

        self.create_subscription(Odometry,  '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Escuchar tecla 's' en hilo separado (pynput no requiere root)
        def _on_press(key):
            try:
                if key.char == 's':
                    self.save_map_to_file()
            except AttributeError:
                pass  # teclas especiales no tienen .char

        def _key_listener():
            with keyboard.Listener(on_press=_on_press):
                threading.Event().wait()

        t = threading.Thread(target=_key_listener, daemon=True)
        t.start()
        self.get_logger().info("Presiona 's' para guardar el mapa en /tmp/")

    # ── Guardar mapa ──────────────────────────────────────────────────────────
    def save_map_to_file(self):
        """Guarda log-odds y mapa de ocupación en disco."""
        if not self.map_built:
            self.get_logger().warn('Mapa aún no construido, nada que guardar.')
            return
        np.save('/tmp/slam_log_odds.npy', self.log_odds)
        prob = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))
        occ  = np.full((self.H, self.W), -1, dtype=np.int8)
        occ[prob > 0.70] = 100
        occ[prob < 0.35] = 0
        np.save('/tmp/slam_map.npy', np.flipud(occ))
        self.get_logger().info('Mapa guardado → /tmp/slam_log_odds.npy  /tmp/slam_map.npy')

    # ── Motion model ─────────────────────────────────────────────────────────
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

        if abs(dx) < 0.001 and abs(dy) < 0.001 and abs(dtheta) < 0.001:
            return

        self.pose_anterior = pose_actual

        self.particulas[:, 0] += dx     + np.random.randn(N).astype(np.float32) * 0.02
        self.particulas[:, 1] += dy     + np.random.randn(N).astype(np.float32) * 0.02
        self.particulas[:, 2] += dtheta + np.random.randn(N).astype(np.float32) * 0.01

    # ── Sensor model + mapping ────────────────────────────────────────────────
    def scan_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges[ranges > msg.range_max] = np.nan
        ranges[ranges < msg.range_min] = np.nan

        if self.angles_cache is None:
            self.angles_cache = np.linspace(
                msg.angle_min, msg.angle_max,
                len(msg.ranges), dtype=np.float32
            )[::5]
            self.angles_cache += math.pi

        r = ranges[::5]
        a = self.angles_cache

        # 1. Actualizar mapa con pose estimada actual
        update_map_with_scan(
            self.log_odds, r, a,
            self.rx, self.ry, self.ryaw,
            self.W, self.H
        )
        self.map_built = True

        # 2. Calcular pesos
        pesos = score_particles(self.particulas, self.log_odds, r, a, self.W, self.H)

        pesos -= pesos.max()
        pesos  = np.exp(pesos / 5.0)
        peso_sum = pesos.sum()

        if peso_sum < 1e-10:
            self.get_logger().warn('Pesos degenerados — manteniendo pose anterior')
            return

        pesos /= peso_sum

        # 3. Resampleo
        indices = np.random.choice(N, N, p=pesos)
        self.particulas = self.particulas[indices].copy()

        self.particulas[:, 0] += np.random.randn(N).astype(np.float32) * 0.03
        self.particulas[:, 1] += np.random.randn(N).astype(np.float32) * 0.03
        self.particulas[:, 2] += np.random.randn(N).astype(np.float32) * 0.01

        # 4. Pose = media ponderada post-resampleo
        pesos2 = score_particles(self.particulas, self.log_odds, r, a, self.W, self.H)
        pesos2 -= pesos2.max()
        pesos2  = np.exp(pesos2 / 5.0)
        pesos2 /= pesos2.sum()

        self.rx   = float(np.dot(pesos2, self.particulas[:, 0]))
        self.ry   = float(np.dot(pesos2, self.particulas[:, 1]))
        self.ryaw = float(np.dot(pesos2, self.particulas[:, 2]))

        # 5. Publicar pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp    = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = self.rx
        pose_msg.pose.position.y = self.ry
        pose_msg.pose.orientation.z = float(np.sin(self.ryaw / 2))
        pose_msg.pose.orientation.w = float(np.cos(self.ryaw / 2))
        self.pub_pose.publish(pose_msg)

        self.get_logger().info(
            f'Pose: x={self.rx:.2f} y={self.ry:.2f} '
            f'yaw={math.degrees(self.ryaw):.1f}°'
        )

    # ── Publicar mapa ─────────────────────────────────────────────────────────
    def publish_map(self):
        if not self.map_built:
            return

        prob = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))
        occ  = np.full((self.H, self.W), -1, dtype=np.int8)
        occ[prob > 0.70]  = 100
        occ[prob < 0.35]  = 0

        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.resolution = RESOLUTION
        msg.info.width      = self.W
        msg.info.height     = self.H
        msg.info.origin.position.x = MAP_X_MIN
        msg.info.origin.position.y = MAP_Y_MIN
        msg.info.origin.orientation.w = 1.0
        occ_flipped = np.flipud(occ)
        msg.data = occ_flipped.flatten().tolist()
        self.pub_map.publish(msg)


def main():
    rclpy.init()
    node = SLAMNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()