import rclpy
import math
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped

GOAL = (1.1270, 0.7566)

LIN_VEL = 0.12
ANG_VEL = 0.10

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


class Bug1(Node):
    def __init__(self):
        super().__init__('bug1')

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub_scan = self.create_subscription(LaserScan,   '/scan',     self.scan_cb, qos)
        self.sub_pose = self.create_subscription(PoseStamped, '/mcl_pose', self.pose_cb, 10)
        self.pub      = self.create_publisher(Twist, '/cmd_vel', 10)

        self.x    = 0.0
        self.y    = 0.0
        self.yaw  = 0.0
        self.mcl_listo = False

        self.state     = 'GO_TO_GOAL'
        self.threshold = 0.3
        self.goal_tol  = 0.3

        self.hit_point = None
        self.min_dist  = float('inf')
        self.min_point = None
        self.toured    = False
        self.alejado   = False

        self.get_logger().info(f'Bug1 listo — esperando MCL... goal={GOAL}')

    # ── Pose desde MCL ────────────────────────────────────────────────────────
    def pose_cb(self, msg):
        if not self.mcl_listo:
            self.get_logger().info('MCL recibido — iniciando navegación')
            self.mcl_listo = True
        self.x   = msg.pose.position.x
        self.y   = msg.pose.position.y
        q        = msg.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    # ── Control principal ─────────────────────────────────────────────────────
    def scan_cb(self, msg):
        if not self.mcl_listo:
            return

        ranges = msg.ranges
        frente = [r for r in (list(ranges[0:30]) + list(ranges[330:360]))
                  if not math.isinf(r) and not math.isnan(r)]
        min_frente = min(frente) if frente else float('inf')

        cmd = Twist()

        # ── Goal alcanzado ────────────────────────────────────────────────────
        if dist_to(self.x, self.y, *GOAL) < self.goal_tol:
            self.get_logger().info('Goal alcanzado!')
            self.pub.publish(cmd)
            return

        # ── GO_TO_GOAL ────────────────────────────────────────────────────────
        if self.state == 'GO_TO_GOAL':
            if min_frente < self.threshold:
                self.hit_point = (self.x, self.y)
                self.min_dist  = dist_to(self.x, self.y, *GOAL)
                self.min_point = (self.x, self.y)
                self.toured    = False
                self.alejado   = False
                self.state     = 'FOLLOW_WALL'
                self.get_logger().info(f'Obstáculo — FOLLOW_WALL desde {self.hit_point}')
            else:
                err = angle_diff(angle_to(self.x, self.y, *GOAL), self.yaw)
                cmd.linear.x  = LIN_VEL
                cmd.angular.z = max(-ANG_VEL, min(ANG_VEL, 2.0 * err))

        # ── FOLLOW_WALL ───────────────────────────────────────────────────────
        elif self.state == 'FOLLOW_WALL':
            d = dist_to(self.x, self.y, *GOAL)
            if d < self.min_dist:
                self.min_dist  = d
                self.min_point = (self.x, self.y)

            if not self.alejado:
                if dist_to(self.x, self.y, *self.hit_point) > 0.5:
                    self.alejado = True
            else:
                if dist_to(self.x, self.y, *self.hit_point) < 0.3:
                    self.toured = True

            if self.toured:
                self.state = 'RETURN_TO_MIN'
                self.get_logger().info(f'Vuelta completa — RETURN_TO_MIN en {self.min_point}')
            else:
                if min_frente < self.threshold:
                    izquierda = [r for r in list(ranges[30:90])
                                 if not math.isinf(r) and not math.isnan(r)]
                    derecha   = [r for r in list(ranges[270:330])
                                 if not math.isinf(r) and not math.isnan(r)]
                    min_iz  = min(izquierda) if izquierda else float('inf')
                    min_der = min(derecha)   if derecha   else float('inf')
                    cmd.linear.x  = 0.0
                    cmd.angular.z = ANG_VEL if min_iz > min_der else -ANG_VEL
                else:
                    err = angle_diff(angle_to(self.x, self.y, *GOAL), self.yaw)
                    cmd.linear.x  = LIN_VEL
                    cmd.angular.z = ANG_VEL if err > 0 else -ANG_VEL

        # ── RETURN_TO_MIN ─────────────────────────────────────────────────────
        elif self.state == 'RETURN_TO_MIN':
            if dist_to(self.x, self.y, *self.min_point) < 0.3:
                self.state = 'GO_TO_GOAL'
                self.get_logger().info('En punto mínimo — GO_TO_GOAL')
            else:
                err = angle_diff(angle_to(self.x, self.y, *self.min_point), self.yaw)
                cmd.linear.x  = LIN_VEL
                cmd.angular.z = max(-ANG_VEL, min(ANG_VEL, 2.0 * err))

        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = Bug1()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()