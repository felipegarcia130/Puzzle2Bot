"""
odom_publisher.py
-----------------
Nodo ROS2 que lee velocidades medidas de encoders (rad/s) desde:
  /VelocityEncL  →  std_msgs/Float32
  /VelocityEncR  →  std_msgs/Float32

Y publica odometría integrada en:
  /odom          →  nav_msgs/Odometry  (+ TF odom → base_link)

Cinemática diferencial:
  v_izq = omega_izq * WHEEL_RADIUS
  v_der = omega_der * WHEEL_RADIUS
  v     = (v_der + v_izq) / 2
  w     = (v_der - v_izq) / WHEEL_BASE
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32
from geometry_msgs.msg import TransformStamped, Quaternion
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster

# ── Parámetros del robot ──────────────────────────────────────────────────────
WHEEL_RADIUS = 0.05    # [m]   radio de cada rueda
WHEEL_BASE   = 0.19    # [m]   distancia entre centros de rueda

# ── Topics ────────────────────────────────────────────────────────────────────
TOPIC_ENC_L = '/VelocityEncL'
TOPIC_ENC_R = '/VelocityEncR'
TOPIC_ODOM  = '/odom'


def euler_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class OdomPublisher(Node):
    def __init__(self):
        super().__init__('odom_publisher')

        # Pose integrada
        self.x     = 0.0
        self.y     = 0.0
        self.theta = 0.0

        # Últimas velocidades recibidas [rad/s]
        self.omega_l = 0.0
        self.omega_r = 0.0

        # Control de tiempo
        self.last_time = self.get_clock().now()

        # Publisher y TF
        self.pub_odom       = self.create_publisher(Odometry, TOPIC_ODOM, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # QoS BEST_EFFORT para coincidir con la hackerboard
        qos_enc = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Subscribers encoders
        self.create_subscription(Float32, TOPIC_ENC_L, self.cb_enc_l, qos_enc)
        self.create_subscription(Float32, TOPIC_ENC_R, self.cb_enc_r, qos_enc)

        # Timer de integración a 20 Hz
        self.create_timer(0.05, self.integrate_and_publish)

        self.get_logger().info(
            f'OdomPublisher listo | '
            f'wheelbase={WHEEL_BASE} m  radio={WHEEL_RADIUS} m'
        )

    # ── Callbacks encoders ────────────────────────────────────────────────────
    def cb_enc_l(self, msg: Float32):
        self.omega_l = msg.data   # [rad/s]

    def cb_enc_r(self, msg: Float32):
        self.omega_r = msg.data   # [rad/s]

    # ── Integración y publicación ─────────────────────────────────────────────
    def integrate_and_publish(self):
        now = self.get_clock().now()
        dt  = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        if dt <= 0.0 or dt > 1.0:
            return

        # rad/s → m/s
        v_l = self.omega_l * WHEEL_RADIUS
        v_r = self.omega_r * WHEEL_RADIUS

        # Cinemática diferencial
        v = (v_r + v_l) / 2.0
        w = (v_r - v_l) / WHEEL_BASE

        # Integración Euler
        self.x     += v * math.cos(self.theta) * dt
        self.y     += v * math.sin(self.theta) * dt
        self.theta += w * dt
        self.theta  = math.atan2(math.sin(self.theta), math.cos(self.theta))

        q     = euler_to_quaternion(self.theta)
        stamp = now.to_msg()

        # TF odom → base_link
        tf_msg = TransformStamped()
        tf_msg.header.stamp    = stamp
        tf_msg.header.frame_id = 'odom'
        tf_msg.child_frame_id  = 'base_link'
        tf_msg.transform.translation.x = self.x
        tf_msg.transform.translation.y = self.y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation      = q
        self.tf_broadcaster.sendTransform(tf_msg)

        # Odometry
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'

        odom.pose.pose.position.x  = self.x
        odom.pose.pose.position.y  = self.y
        odom.pose.pose.position.z  = 0.0
        odom.pose.pose.orientation = q

        odom.twist.twist.linear.x  = v
        odom.twist.twist.angular.z = w

        self.pub_odom.publish(odom)

        self.get_logger().debug(
            f'x={self.x:.3f} y={self.y:.3f} θ={math.degrees(self.theta):.1f}°  '
            f'ωL={self.omega_l:.3f} ωR={self.omega_r:.3f} rad/s'
        )


def main():
    rclpy.init()
    node = OdomPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()