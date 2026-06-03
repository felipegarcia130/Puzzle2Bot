import rclpy
import math
import heapq
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path

# ── Parámetros del mapa ───────────────────────────────────────────────────────
MAP_X_MIN  = -4.0
MAP_X_MAX  =  5.0
MAP_Y_MIN  = -3.0
MAP_Y_MAX  =  6.0
RESOLUTION =  0.05

# ── Velocidades ───────────────────────────────────────────────────────────────
LIN_VEL = 0.07
ANG_VEL = 0.07

# ── Parámetros A* ─────────────────────────────────────────────────────────────
INFLATE_RADIUS = 6
WAYPOINT_TOL   = 0.25
GOAL_TOL       = 0.35


def world_to_cell(x, y):
    col = int((x - MAP_X_MIN) / RESOLUTION)
    row = int((MAP_Y_MAX - y) / RESOLUTION)
    return row, col

def cell_to_world(row, col):
    x = MAP_X_MIN + col * RESOLUTION
    y = MAP_Y_MAX - row * RESOLUTION
    return x, y

def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def angle_diff(a, b):
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d

def euler_from_quaternion(q):
    x, y, z, w = q
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def inflate_grid(grid, radius):
    from scipy.ndimage import binary_dilation
    obstacle_mask = (grid == 0)
    struct = np.ones((2*radius+1, 2*radius+1), dtype=bool)
    inflated = binary_dilation(obstacle_mask, structure=struct)
    result = grid.copy()
    result[inflated] = 0
    return result

def astar(grid, start, goal):
    H, W = grid.shape

    def neighbors(r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),
                        (-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 1:
                cost = 1.414 if dr != 0 and dc != 0 else 1.0
                yield nr, nc, cost

    open_heap = []
    heapq.heappush(open_heap, (0, start))
    came_from = {start: None}
    g_score   = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        for nr, nc, cost in neighbors(*current):
            nb = (nr, nc)
            tentative_g = g_score[current] + cost
            if tentative_g < g_score.get(nb, float('inf')):
                g_score[nb] = tentative_g
                f = tentative_g + heuristic(nb, goal)
                heapq.heappush(open_heap, (f, nb))
                came_from[nb] = current
    return None

def smooth_path(path, skip=5):
    if len(path) <= 2:
        return path
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = min(i + skip, len(path) - 1)
        smoothed.append(path[j])
        i = j
    if smoothed[-1] != path[-1]:
        smoothed.append(path[-1])
    return smoothed


class AStarNode(Node):
    def __init__(self):
        super().__init__('astar_node')

        # ── Cargar mapa ───────────────────────────────────────────────────────
        occ = np.load('/home/felipe/ros2_ws/src/mcl_robot/maps/slam_map.npy')
        occ = np.flipud(occ)
        self.grid_static = (occ == 0).astype(np.uint8)
        self.H, self.W   = self.grid_static.shape
        self.grid_inflated = inflate_grid(self.grid_static, INFLATE_RADIUS)
        self.get_logger().info(f'Mapa cargado: {self.W}x{self.H}')

        # ── Estado ───────────────────────────────────────────────────────────
        self.x         = 0.0
        self.y         = 0.0
        self.yaw       = 0.0
        self.mcl_listo = False
        self.goal_world = None
        self.goal_cell  = None
        self.path         = []
        self.waypoint_idx = 0

        # ── Publishers ────────────────────────────────────────────────────────
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub_cmd      = self.create_publisher(Twist,       '/cmd_vel',          10)
        self.pub_path     = self.create_publisher(Path,        '/planned_path',     1)
        self.pub_waypoint = self.create_publisher(PoseStamped, '/current_waypoint', 10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.sub_scan = self.create_subscription(LaserScan,   '/scan',      self.scan_cb, qos)
        self.sub_pose = self.create_subscription(PoseStamped, '/mcl_pose',  self.pose_cb, 10)
        self.sub_goal = self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)

        self.get_logger().info('Esperando MCL — pica un goal con "2D Goal Pose" en RViz')

    # ── Goal desde RViz ───────────────────────────────────────────────────────
    def goal_cb(self, msg):
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        self.goal_world = (gx, gy)
        self.goal_cell  = world_to_cell(gx, gy)
        self.get_logger().info(f'Nuevo goal: ({gx:.2f}, {gy:.2f}) → celda {self.goal_cell}')
        if self.mcl_listo:
            self.plan()

    # ── Pose desde MCL ────────────────────────────────────────────────────────
    def pose_cb(self, msg):
        self.x   = msg.pose.position.x
        self.y   = msg.pose.position.y
        q        = msg.pose.orientation
        self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        if not self.mcl_listo:
            self.mcl_listo = True
            self.get_logger().info('MCL listo — pica un goal en RViz')

    # ── Planificación A* ──────────────────────────────────────────────────────
    def plan(self):
        if self.goal_cell is None:
            return

        sr, sc = world_to_cell(self.x, self.y)
        gr, gc = self.goal_cell
        sr = int(np.clip(sr, 0, self.H-1))
        sc = int(np.clip(sc, 0, self.W-1))
        gr = int(np.clip(gr, 0, self.H-1))
        gc = int(np.clip(gc, 0, self.W-1))

        # Si start bloqueado por inflado, buscar celda libre cercana
        if self.grid_inflated[sr, sc] == 0:
            found = False
            for d in range(1, 10):
                for dr in range(-d, d+1):
                    for dc in range(-d, d+1):
                        nr, nc = sr+dr, sc+dc
                        if 0 <= nr < self.H and 0 <= nc < self.W:
                            if self.grid_inflated[nr, nc] == 1:
                                sr, sc = nr, nc
                                found = True
                                break
                    if found: 
                        break
                if found: 
                    break

        path_cells = astar(self.grid_inflated, (sr, sc), (gr, gc))

        if path_cells is None:
            self.get_logger().warn('A*: sin camino al goal')
            return

        self.path         = smooth_path(path_cells, skip=5)
        self.waypoint_idx = 0
        self.get_logger().info(f'A*: {len(self.path)} waypoints')

        # Publicar path completo en RViz
        path_msg = Path()
        path_msg.header.stamp    = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        for (r, c) in path_cells:
            wx, wy = cell_to_world(r, c)
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.pub_path.publish(path_msg)

    # ── Control ───────────────────────────────────────────────────────────────
    def scan_cb(self, msg):
        if not self.mcl_listo or self.goal_world is None:
            return

        ranges = list(msg.ranges)
        cmd    = Twist()

        # Goal final alcanzado
        if math.hypot(self.x - self.goal_world[0],
                      self.y - self.goal_world[1]) < GOAL_TOL:
            self.get_logger().info('Goal alcanzado!')
            self.pub_cmd.publish(cmd)
            return

        if not self.path or self.waypoint_idx >= len(self.path):
            self.pub_cmd.publish(cmd)
            return

        # Waypoint actual
        wp_row, wp_col = self.path[self.waypoint_idx]
        wp_x, wp_y    = cell_to_world(wp_row, wp_col)

        wp_msg = PoseStamped()
        wp_msg.header.stamp    = self.get_clock().now().to_msg()
        wp_msg.header.frame_id = 'map'
        wp_msg.pose.position.x = wp_x
        wp_msg.pose.position.y = wp_y
        wp_msg.pose.orientation.w = 1.0
        self.pub_waypoint.publish(wp_msg)

        # Waypoint alcanzado — avanzar al siguiente
        dist_wp = math.hypot(self.x - wp_x, self.y - wp_y)
        if dist_wp < WAYPOINT_TOL:
            self.waypoint_idx += 1
            self.get_logger().info(f'WP {self.waypoint_idx}/{len(self.path)}')
            self.pub_cmd.publish(cmd)
            return

        # Control angular + lineal
        target_angle = math.atan2(wp_y - self.y, wp_x - self.x)
        err = angle_diff(target_angle, self.yaw)

        if abs(err) > math.radians(45):
            cmd.linear.x  = 0.0
            cmd.angular.z = max(-ANG_VEL, min(ANG_VEL, err))
        elif abs(err) > math.radians(10):
            cmd.linear.x  = LIN_VEL * 0.5
            cmd.angular.z = max(-ANG_VEL, min(ANG_VEL, err))
        else:
            cmd.linear.x  = LIN_VEL
            cmd.angular.z = 0.0

        # Safety LiDAR — frena si hay algo muy cerca al frente
        front = [r for r in (ranges[0:20] + ranges[340:360])
                 if math.isfinite(r) and not math.isnan(r)]
        min_front = min(front) if front else float('inf')
        if min_front < 0.25:
            cmd.linear.x  = 0.0
            cmd.angular.z = ANG_VEL
            self.get_logger().warn(f'[SAFETY] obstáculo a {min_front:.2f}m')

        self.pub_cmd.publish(cmd)


def main():
    rclpy.init()
    node = AStarNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()