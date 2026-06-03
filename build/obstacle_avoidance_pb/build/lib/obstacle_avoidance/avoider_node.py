import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class ObstacleAvoider(Node):
    def __init__(self):
        super().__init__('obstacle_avoider')
        self.sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.threshold = 0.3
        self.girando_escape = False

    def scan_cb(self, msg):
        ranges = msg.ranges

        izquierda = ranges[30:90]
        frente    = ranges[0:30] + ranges[330:360]
        derecha   = ranges[270:330]
        atras     = ranges[150:210]  # sector trasero

        min_frente    = min(frente)
        min_izquierda = min(izquierda)
        min_derecha   = min(derecha)
        min_atras     = min(atras)

        if self.girando_escape:
            if min_frente > self.threshold:
                self.girando_escape = False
            else:
                cmd = Twist()
                cmd.angular.z = 1.0
                self.pub.publish(cmd)
                return

        if min_frente < self.threshold:
            if min_izquierda < self.threshold and min_derecha < self.threshold:
                self.girando_escape = True
                cmd = Twist()
                cmd.linear.x = 0.0
                # si atrás está libre, retrocede en lugar de girar en sitio
                if min_atras > 0.2:
                    cmd.linear.x = -0.2
                    cmd.angular.z = 0.0
                else:
                    cmd.angular.z = 1.0
                self.pub.publish(cmd)
            elif min_izquierda > min_derecha:
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 1.0
                self.pub.publish(cmd)
            else:
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = -1.0
                self.pub.publish(cmd)
        else:
            cmd = Twist()
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
            self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoider()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()