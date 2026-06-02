import math
import heapq
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from scipy.ndimage import binary_dilation

# ── Parámetros ─────────────────────────────────────────────────────────────────
REPLAN_INTERVAL   = 6.0   # segundos entre replanificaciones periódicas
MIN_CLUSTER_SIZE  = 3     # celdas mínimas para considerar un cluster de frontera válido
GOAL_TOLERANCE    = 0.30  # metros para considerar un waypoint alcanzado
OBSTACLE_DIST     = 0.45  # metros — distancia mínima antes de evadir (aumentada)
MIN_FRONTIER_DIST = 10    # celdas — distancia mínima al robot para elegir frontera
INFLATION_CELLS   = 3     # celdas a inflar alrededor de obstáculos (~25 cm si res=0.05)
WAYPOINT_STEP     = 2     # tomar 1 de cada N waypoints del path (antes era 5)


def angle_to(x, y, gx, gy): return math.atan2(gy - y, gx - x)
def dist_to(x, y, gx, gy):  return math.hypot(gx - x, gy - y)
def angle_diff(a, b):
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


# ── Inflado de obstáculos ──────────────────────────────────────────────────────
def inflate_grid(grid, inflation_cells=INFLATION_CELLS):
    """
    Dilata los obstáculos N celdas en todas direcciones.
    Esto crea un margen de seguridad equivalente al radio del robot,
    evitando que A* genere rutas demasiado cercanas a las paredes.
    Las celdas desconocidas (-1) se preservan para la detección de fronteras.
    """
    obstacles = (grid > 50).astype(bool)
    inflated  = binary_dilation(obstacles, iterations=inflation_cells)
    result    = grid.copy()
    # Solo inflar sobre celdas libres (no sobreescribir desconocidas)
    result[(inflated) & (grid >= 0)] = 100
    return result


# ── A* ─────────────────────────────────────────────────────────────────────────
def astar(grid, width, height, start, goal):
    def h(a, b): return math.hypot(b[0] - a[0], b[1] - a[1])
    def neighbors(x, y):
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and grid[ny, nx] == 0:
                yield nx, ny, 1.414 if dx and dy else 1.0

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score   = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        for nx, ny, cost in neighbors(*current):
            nb = (nx, ny)
            tg = g_score[current] + cost
            if tg < g_score.get(nb, float('inf')):
                came_from[nb] = current
                g_score[nb]   = tg
                heapq.heappush(open_set, (tg + h(nb, goal), nb))
    return []


# ── Fronteras ──────────────────────────────────────────────────────────────────
def detect_frontiers(grid):
    """Devuelve (cols, rows) de celdas libres adyacentes a celdas desconocidas."""
    free    = (grid == 0).astype(np.uint8)
    unk     = (grid == -1).astype(np.uint8)
    unk_adj = (
        np.roll(unk,  1, axis=0) | np.roll(unk, -1, axis=0) |
        np.roll(unk,  1, axis=1) | np.roll(unk, -1, axis=1)
    )
    rows, cols = np.where(free & unk_adj)
    return cols, rows


def best_frontier(cols, rows, rx, ry):
    """
    Agrupa celdas de frontera en clusters de 10x10 celdas y elige el mejor
    según la relación tamaño/distancia (favorece clusters grandes y cercanos).
    """
    if len(cols) == 0:
        return None

    coarse_c = (cols // 10) * 10 + 5
    coarse_r = (rows // 10) * 10 + 5

    clusters = {}
    for c, r in zip(coarse_c.tolist(), coarse_r.tolist()):
        key = (c, r)
        clusters[key] = clusters.get(key, 0) + 1

    clusters = {k: v for k, v in clusters.items() if v >= MIN_CLUSTER_SIZE}
    if not clusters:
        return None

    clusters = {
        k: v for k, v in clusters.items()
        if math.hypot(k[0] - rx, k[1] - ry) >= MIN_FRONTIER_DIST
    }
    if not clusters:
        return None

    best = min(clusters, key=lambda k: -clusters[k] / (math.hypot(k[0]-rx, k[1]-ry) + 1e-3))
    return best


# ── Nodo principal ─────────────────────────────────────────────────────────────
class ExplorationNode(Node):
    def __init__(self):
        super().__init__('exploration_node')

        self.sub_map  = self.create_subscription(OccupancyGrid, '/map',       self.map_cb,  1)
        self.sub_pose = self.create_subscription(PoseStamped,   '/slam_pose', self.pose_cb, 10)
        self.sub_scan = self.create_subscription(LaserScan,     '/scan',      self.scan_cb, 10)
        self.pub      = self.create_publisher(Twist, '/cmd_vel', 10)

        self.grid       = None
        self.width      = 0
        self.height     = 0
        self.resolution = 0.05
        self.origin_x   = 0.0
        self.origin_y   = 0.0

        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0

        # Rangos LiDAR por sector
        self.min_front_range = float('inf')
        self.left_range      = float('inf')
        self.right_range     = float('inf')

        self.waypoints  = []
        self.current_wp = 0
        self.exploring  = False
        self.done       = False
        self.first_map  = True

        # Contador para no replanificar demasiado seguido tras evasión
        self._avoidance_ticks = 0

        self.create_timer(0.1,             self.control_loop)
        self.create_timer(REPLAN_INTERVAL, self.replan)

    # ── Callbacks ──────────────────────────────────────────────────────────────
    def scan_cb(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)
        n      = len(ranges)

        def safe_min(r):
            r = r[np.isfinite(r) & (r > 0.05)]
            return float(np.min(r)) if len(r) > 0 else float('inf')

        # Sectores: front ±22.5°, left ±45°, right ±45°
        # Asume 360 rayos; ajusta los índices si tu LiDAR es diferente
        front_half = n // 16          # ~22 rayos a cada lado del frente
        left_start = n // 4           # 90°
        right_end  = 3 * n // 4       # 270°

        self.min_front_range = safe_min(
            np.concatenate([ranges[:front_half], ranges[n - front_half:]])
        )
        self.left_range  = safe_min(ranges[front_half : left_start])
        self.right_range = safe_min(ranges[right_end  : n - front_half])

    def map_cb(self, msg):
        self.width      = msg.info.width
        self.height     = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x   = msg.info.origin.position.x
        self.origin_y   = msg.info.origin.position.y
        self.grid = np.array(msg.data, dtype=np.int8).reshape((self.height, self.width))

        if self.first_map and not self.done:
            self.first_map = False
            self.replan()

    def pose_cb(self, msg):
        self.robot_x   = msg.pose.position.x
        self.robot_y   = msg.pose.position.y
        q = msg.pose.orientation
        self.robot_yaw = 2.0 * math.atan2(q.z, q.w)

    # ── Coordenadas ────────────────────────────────────────────────────────────
    def world_to_grid(self, wx, wy):
        gx = int((wx - self.origin_x) / self.resolution)
        gy = self.height - 1 - int((wy - self.origin_y) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        gy_real = self.height - 1 - gy
        wx = gx      * self.resolution + self.origin_x + self.resolution / 2
        wy = gy_real * self.resolution + self.origin_y + self.resolution / 2
        return wx, wy

    def nearest_free(self, gx, gy, grid=None):
        """Encuentra la celda libre más cercana; usa el grid inflado si se pasa."""
        g = grid if grid is not None else self.grid
        for r in range(1, 30):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if g[ny, nx] == 0:
                            return nx, ny
        return gx, gy

    # ── Exploración ────────────────────────────────────────────────────────────
    def replan(self):
        if self.done or self.grid is None:
            return

        rx_g, ry_g = self.world_to_grid(self.robot_x, self.robot_y)

        # Detectar fronteras sobre el grid ORIGINAL (no inflado)
        cols, rows = detect_frontiers(self.grid)
        target     = best_frontier(cols, rows, rx_g, ry_g)

        if target is None:
            self.get_logger().info('Exploración completa — no quedan fronteras')
            self.pub.publish(Twist())
            self.done = True
            return

        # Inflar obstáculos para la planeación
        inflated = inflate_grid(self.grid, INFLATION_CELLS)

        start = self.nearest_free(rx_g, ry_g, inflated)
        goal  = self.nearest_free(*target, inflated)

        path = astar(inflated, self.width, self.height, start, goal)
        if not path:
            self.get_logger().warn('A* no encontró ruta a la frontera — reintentando')
            return

        # Paso reducido para no cortar esquinas
        self.waypoints  = [self.grid_to_world(gx, gy) for gx, gy in path[::WAYPOINT_STEP]]
        self.current_wp = 0
        self.exploring  = True

        wx, wy = self.grid_to_world(*goal)
        self.get_logger().info(
            f'Frontera → ({wx:.2f}, {wy:.2f}) | '
            f'{len(self.waypoints)} waypoints | '
            f'{len(cols)} celdas de frontera restantes'
        )

    # ── Control ────────────────────────────────────────────────────────────────
    def control_loop(self):
        if self.done or not self.exploring:
            return

        # Fin de ruta → buscar siguiente frontera
        if self.current_wp >= len(self.waypoints):
            self.exploring = False
            self.replan()
            return

        # ── Evasión reactiva ────────────────────────────────────────────────
        if self.min_front_range < OBSTACLE_DIST:
            cmd = Twist()
            # Gira hacia el lado más despejado
            cmd.angular.z = 0.3 if self.left_range >= self.right_range else -0.4
            self.pub.publish(cmd)
            self._avoidance_ticks += 1

            # Tras varios ticks girando, replanifica en lugar de seguir el wp actual
            if self._avoidance_ticks > 20:
                self._avoidance_ticks = 0
                self.exploring = False
                self.replan()
            return

        self._avoidance_ticks = 0

        wx, wy = self.waypoints[self.current_wp]

        if dist_to(self.robot_x, self.robot_y, wx, wy) < GOAL_TOLERANCE:
            self.current_wp += 1
            return

        err = angle_diff(angle_to(self.robot_x, self.robot_y, wx, wy), self.robot_yaw)
        cmd = Twist()
        if abs(err) > 0.5:
            # Solo girar, sin avanzar
            cmd.angular.z = max(-0.4, min(0.4, 0.8 * err)) 
        else:
            # Avanzar con corrección angular suave
            cmd.linear.x  = 0.10
            cmd.angular.z = max(-0.3, min(0.3, 0.6 * err))
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ExplorationNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()