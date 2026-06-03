import rclpy
import math
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped

GOAL = (1.1270, 0.7566)

LIN_VEL      = 0.05
ANG_VEL      = 0.09
ANG_VEL_WALL = 0.09

THRESHOLD       = 0.8   # distancia para entrar a FOLLOW_WALL
SAFETY_DIST     = 0.25  # distancia de emergencia — frena sin importar nada
SAFETY_DIST_LAT = 0.15  # distancia lateral de emergencia

def euler_from_quaternion(q):
    x, y, z, w = q
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return 0.0, 0.0, math.atan2(siny_cosp, cosy_cosp)

def angle_to(x, y, gx, gy):  return math.atan2(gy - y, gx - x)
def dist_to(x, y, gx, gy):   return math.hypot(gx - x, gy - y)
def angle_diff(a, b):
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


class Bug0(Node):
    def __init__(self):
        super().__init__('bug0')

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub_scan = self.create_subscription(LaserScan,   '/scan',     self.scan_cb, qos)
        self.sub_pose = self.create_subscription(PoseStamped, '/mcl_pose', self.pose_cb, 10)
        self.pub      = self.create_publisher(Twist, '/cmd_vel', 10)

        self.x         = 0.0
        self.y         = 0.0
        self.yaw       = 0.0
        self.state     = 'GO_TO_GOAL'
        self.goal_tol  = 0.3
        self.mcl_listo = False
        self.wall_side = 1   # +1 izquierda, -1 derecha

        # último scan guardado para el backup
        self.last_ranges = None

        self.get_logger().info(f'Bug0 listo — esperando MCL... goal={GOAL}')
        self.get_logger().info(f'lin={LIN_VEL} m/s  ang={ANG_VEL} rad/s  safety={SAFETY_DIST} m')

    # ── Pose desde MCL ────────────────────────────────────────────────────────
    def pose_cb(self, msg):
        if not self.mcl_listo:
            self.get_logger().info('MCL recibido — iniciando navegación')
            self.mcl_listo = True
        self.x   = msg.pose.position.x
        self.y   = msg.pose.position.y
        q        = msg.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    # ── Backup reactivo: filtra cualquier cmd antes de publicar ──────────────
    def safety_filter(self, cmd):
        """
        Revisa el último scan en tiempo real.
        Si hay algo demasiado cerca, sobreescribe el comando para evitar colisión.
        Devuelve (cmd_filtrado, emergencia:bool)
        """
        if self.last_ranges is None:
            return cmd, False

        ranges = self.last_ranges

        front_vals = [r for r in (ranges[0:20] + ranges[340:360])
                      if not math.isinf(r) and not math.isnan(r)]
        left_vals  = [r for r in ranges[20:90]
                      if not math.isinf(r) and not math.isnan(r)]
        right_vals = [r for r in ranges[270:340]
                      if not math.isinf(r) and not math.isnan(r)]

        min_front = min(front_vals) if front_vals else float('inf')
        min_left  = min(left_vals)  if left_vals  else float('inf')
        min_right = min(right_vals) if right_vals else float('inf')

        safe_cmd = Twist()

        # Emergencia frontal — para en seco y gira al lado más despejado
        if min_front < SAFETY_DIST:
            safe_cmd.linear.x  = 0.0
            safe_cmd.angular.z = ANG_VEL_WALL if min_left >= min_right else -ANG_VEL_WALL
            self.get_logger().warn(
                f'[SAFETY] Obstáculo frontal a {min_front:.2f} m — frenando'
            )
            return safe_cmd, True

        # Emergencia lateral — solo frena el avance, deja girar
        if min_left < SAFETY_DIST_LAT:
            safe_cmd.linear.x  = 0.0
            safe_cmd.angular.z = -ANG_VEL_WALL  # aleja de la izquierda
            self.get_logger().warn(
                f'[SAFETY] Obstáculo izquierdo a {min_left:.2f} m'
            )
            return safe_cmd, True

        if min_right < SAFETY_DIST_LAT:
            safe_cmd.linear.x  = 0.0
            safe_cmd.angular.z = ANG_VEL_WALL   # aleja de la derecha
            self.get_logger().warn(
                f'[SAFETY] Obstáculo derecho a {min_right:.2f} m'
            )
            return safe_cmd, True

        return cmd, False

    # ── Control principal ─────────────────────────────────────────────────────
    def scan_cb(self, msg):
        ranges = list(msg.ranges)
        self.last_ranges = ranges  # siempre actualiza el backup

        if not self.mcl_listo:
            return

        front_vals = [r for r in (ranges[0:30] + ranges[330:360])
                      if not math.isinf(r) and not math.isnan(r)]
        left_vals  = [r for r in ranges[30:90]
                      if not math.isinf(r) and not math.isnan(r)]
        right_vals = [r for r in ranges[270:330]
                      if not math.isinf(r) and not math.isnan(r)]

        min_frente = min(front_vals) if front_vals else float('inf')
        min_izq    = min(left_vals)  if left_vals  else float('inf')
        min_der    = min(right_vals) if right_vals else float('inf')

        cmd = Twist()

        # ── Goal alcanzado ────────────────────────────────────────────────────
        if dist_to(self.x, self.y, *GOAL) < self.goal_tol:
            self.get_logger().info('Goal alcanzado!')
            self.pub.publish(cmd)
            return

        # ── GO_TO_GOAL ────────────────────────────────────────────────────────
        if self.state == 'GO_TO_GOAL':
            if min_frente < THRESHOLD:
                self.wall_side = 1 if min_izq >= min_der else -1
                self.state = 'FOLLOW_WALL'
                self.get_logger().info(
                    f'Obstáculo — FOLLOW_WALL ({"izq" if self.wall_side == 1 else "der"})'
                )
            else:
                err = angle_diff(angle_to(self.x, self.y, *GOAL), self.yaw)
                cmd.linear.x  = LIN_VEL
                cmd.angular.z = max(-ANG_VEL, min(ANG_VEL, ANG_VEL * 2.0 * err))

        # ── FOLLOW_WALL ───────────────────────────────────────────────────────
        elif self.state == 'FOLLOW_WALL':
            if min_frente < THRESHOLD:
                cmd.linear.x  = 0.0
                cmd.angular.z = ANG_VEL_WALL * self.wall_side
            else:
                cmd.linear.x  = LIN_VEL
                cmd.angular.z = ANG_VEL_WALL * self.wall_side * 0.5
                err = angle_diff(angle_to(self.x, self.y, *GOAL), self.yaw)
                if abs(err) < 0.3:
                    self.state = 'GO_TO_GOAL'
                    self.get_logger().info('Camino libre — GO_TO_GOAL')

        # ── Filtro de seguridad reactivo (backup LiDAR) ───────────────────────
        cmd, emergencia = self.safety_filter(cmd)
        if emergencia:
            self.state = 'FOLLOW_WALL'  # fuerza bordeo tras emergencia

        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = Bug0()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


