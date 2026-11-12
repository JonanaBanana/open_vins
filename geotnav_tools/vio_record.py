"""Record a VIO odometry topic and PX4's odometry (the reference) to CSV until Ctrl+C.

Both are stamped with the receive time: PX4's own stamps are its boot time.
"""
import csv
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry


class Recorder(Node):
    def __init__(self, vio_topic, out_prefix):
        super().__init__('vio_recorder')
        self.vio = csv.writer(open(out_prefix + '_vio.csv', 'w'))
        self.ref = csv.writer(open(out_prefix + '_ref.csv', 'w'))
        for w in (self.vio, self.ref):
            w.writerow(['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'vx', 'vy', 'vz'])
        self.create_subscription(Odometry, vio_topic, self.on_vio, 50)
        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.on_ref,
                                 qos_profile_sensor_data)
        self.n = 0

    def on_vio(self, m):
        p, q, v = m.pose.pose.position, m.pose.pose.orientation, m.twist.twist.linear
        self.vio.writerow([self.get_clock().now().nanoseconds * 1e-9,
                           p.x, p.y, p.z, q.x, q.y, q.z, q.w, v.x, v.y, v.z])
        self.n += 1
        if self.n % 300 == 0:
            self.get_logger().info(f'{self.n} VIO poses, at {p.x:.1f} {p.y:.1f} {p.z:.1f}')

    def on_ref(self, m):
        # NED / FRD -> ENU / FLU; quaternion as w, x, y, z in PX4.
        p, q, v = m.position, m.q, m.velocity
        self.ref.writerow([self.get_clock().now().nanoseconds * 1e-9, p[1], p[0], -p[2], q[1], q[2], q[3], q[0],
                           v[1], v[0], -v[2]])


rclpy.init()
node = Recorder(sys.argv[1], sys.argv[2])
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
