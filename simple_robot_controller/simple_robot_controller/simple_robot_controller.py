
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Quaternion, PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Int32

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0.0, output_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits

        self.error = 0.0
        self.error_sum = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def update(self, feedback, dt):
        if self.prev_time is None:
            self.prev_time = dt

        error = self.setpoint - feedback
        self.error_sum += error * dt
        derivative = (error - self.prev_error) / dt
        
        # Calculate output
        output = self.kp * error + self.ki * self.error_sum + self.kd * derivative

        # Apply output limits
        if self.output_limits is not None:
            min_output, max_output = self.output_limits
            output = max(min_output, min(max_output, output))

        # Update state variables
        self.prev_error = error

        return -output
    
class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.timer = self.create_timer(0.1, self.calculate_velocities_callback)
        
        # Waypoints Queue
        self.wp_queue = []

        # Controller state
        self.controller_state = 0  # ready to receive

        # Tolerances
        self.goal_tolerance = 0.1 
        self.goal_angular_tolerance = 0.1

        # Initialize PID controllers
        self.linear_pid = PIDController(kp=0.1, ki=0.0, kd=0.0)
        self.angular_pid = PIDController(kp=0.6, ki=0.0, kd=0.0)
        self.orientation_pid = PIDController(kp=0.1, ki=0.0, kd=0.0)

        # Set input limits and output limits for PID controllers
        self.linear_pid.setpoint = 0.0
        self.linear_pid.output_limits = (-0.5, 0.5)
        self.angular_pid.setpoint = 0.0
        self.angular_pid.output_limits = (-2.0, 2.0)

        # Subscribers
        self.subscription_linear = self.create_subscription(Float32, '/pid_settings/linear', self.linear_callback, 10)
        self.subscription_angular = self.create_subscription(Float32, '/pid_settings/angular', self.angular_callback, 10)
        self.waypoints_subscription = self.create_subscription(PoseStamped, '/waypoints', self.wp_callback, 10)
        self.controllerState_subscription = self.create_subscription(Int32, '/ControllerState', self.cs_callback, 10)
        self.goal_pose_subscription = self.create_subscription(PoseStamped, '/goal_pose', self.goal_pose_callback, 10)
        self.odom_subscription = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)

        # Publisher
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Initialize variables
        self.goal_pose = None
        self.current_pose = None
        self.prev_time = self.get_clock().now().nanoseconds / 1e9
    
    def linear_callback(self, msg):
        # Update linear PID controller gain
        self.linear_pid.kp = msg.data

    def angular_callback(self, msg):
        # Update angular PID controller gain
        self.angular_pid.kp = msg.data
    
    def wp_callback(self, msg):
        print("waypoint received")
        if self.goal_pose is None:
            self.goal_pose = msg.pose
        else:
            self.wp_queue.append(msg.pose)

    def cs_callback(self, msg):
        self.controller_state = msg.data

    def goal_pose_callback(self, msg):
        #print("goal received")
        self.goal_pose = msg.pose        

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def calculate_velocities(self):
        if self.goal_pose is None or self.current_pose is None:
            #print("No pose")
            return 0.0, 0.0

        dt = self.get_clock().now().nanoseconds / 1e9 - self.prev_time
        self.prev_time = self.get_clock().now().nanoseconds / 1e9

        orientation_error = self.calculate_orientation_error(self.goal_pose.orientation, self.current_pose.orientation)
        linear_error = self.calculate_linear_error(self.goal_pose.position, self.current_pose.position)
        linear_vel = self.linear_pid.update(linear_error, dt)       
        angular_error = self.calculate_angular_error(self.goal_pose.orientation, self.current_pose.orientation)
        angular_vel = self.angular_pid.update(angular_error, dt)

        # Check if the goal has been reached
        if linear_error < self.goal_tolerance and abs(angular_error) < self.goal_angular_tolerance:
            print("Goal reached!")
            if self.wp_queue:
                self.goal_pose = self.wp_queue.pop(0)
            else:
                self.goal_pose = None

            linear_vel = 0.0
            angular_vel = 0.0

        return linear_vel, angular_vel

    def calculate_linear_error(self, goal_position, current_position):
        x_error = goal_position.x - current_position.x
        y_error = goal_position.y - current_position.y
        linear_error = math.sqrt(x_error ** 2 + y_error ** 2)
        return linear_error

    def quaternion_to_euler(self, quaternion):
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return roll, pitch, yaw

    
    def calculate_angular_error(self, goal_orientation, current_orientation):
        goal_position = self.goal_pose.position
        current_position = self.current_pose.position
        goal_angle = math.atan2(goal_position.y - current_position.y, goal_position.x - current_position.x)
        
        goal_angle = math.atan2(math.sin(goal_angle), math.cos(goal_angle))
        
        _, _, current_yaw = self.quaternion_to_euler(current_orientation)
        angular_error = goal_angle - current_yaw

        if angular_error > math.pi:
            angular_error -= 2 * math.pi
        elif angular_error < -math.pi:
            angular_error += 2 * math.pi

        return angular_error
 
    def calculate_velocities_callback(self):  
          
        linear_vel, angular_vel = self.calculate_velocities()
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_vel
        cmd_vel.angular.z = angular_vel
        #print(linear_vel)
        #print(angular_vel)
        self.velocity_publisher.publish(cmd_vel)

    def orientation_callback(self, msg):
        # Update orientation PID controller gain
        self.orientation_pid.kp = msg.data
    
    def calculate_orientation_error(self, goal_orientation, current_orientation):
        _, _, goal_yaw = self.quaternion_to_euler(goal_orientation)
        _, _, current_yaw = self.quaternion_to_euler(current_orientation)
        orientation_error = goal_yaw - current_yaw

        if orientation_error > math.pi:
            orientation_error -= 2 * math.pi
        elif orientation_error < -math.pi:
            orientation_error += 2 * math.pi

        return orientation_error

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


# class RobotController(Node):
#     def __init__(self):
#         super().__init__('robot_controller')
#         self.timer = self.create_timer(0.1, self.calculate_velocities_callback)
        
#         self.goal_tolerance = 0.1 
#         self.goal_angular_tolerance = 0.1
#         # Wheel parameters
#         self.wheel_diameter = 0.175  # meters
#         self.wheel_center_to_center = 0.3  # meters

#         # Initialize PID controllers
#         self.linear_pid = PIDController(kp=0.1, ki=0.0, kd=0.0)
#         self.angular_pid = PIDController(kp=0.1, ki=0.0, kd=0.0)

#         # Set input limits and output limits for PID controllers
#         self.linear_pid.setpoint = 0.0
#         self.linear_pid.output_limits = (-0.5, 0.5)
#         self.angular_pid.setpoint = 0.0
#         self.angular_pid.output_limits = (-1.0, 1.0)

#         #tuning subscribers
#         self.subscription_linear = self.create_subscription(Float32, '/pid_settings/linear', self.linear_callback, 10)
#         self.subscription_angular = self.create_subscription(Float32, '/pid_settings/angular', self.angular_callback, 10)

#         # Subscribers
#         self.goal_pose_subscription = self.create_subscription(PoseStamped, 'goal_pose', self.goal_pose_callback, 10)
#         #self.pose_subscription = self.create_subscription(PoseStamped, 'odom', self.pose_callback, 10)
#         self.odom_subscription = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)

#         # Publisher
#         self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

#         # Initialize variables
#         self.goal_pose = None
#         self.current_pose = None
#         self.prev_time = self.get_clock().now().nanoseconds / 1e9
    
#     def linear_callback(self, msg):
#         # Update linear PID controller gain
#         self.linear_pid.kp = msg.data

#     def angular_callback(self, msg):
#         # Update angular PID controller gain
#         self.angular_pid.kp = msg.data

#     def goal_pose_callback(self, msg):
#         self.goal_pose = msg.pose
        

#     def odom_callback(self, msg):
#         self.current_pose = msg.pose.pose
#         # Extract position and orientation from the odometry message
#         current_x = msg.pose.pose.position.x
#         current_y = msg.pose.pose.position.y
#         current_z = msg.pose.pose.position.z
#         current_orientation = msg.pose.pose.orientation

#     def calculate_velocities(self):
#         if self.goal_pose is None or self.current_pose is None:
#             return 0.0,0.0

#         dt = self.get_clock().now().nanoseconds / 1e9 - self.prev_time
#         self.prev_time = self.get_clock().now().nanoseconds / 1e9

#         linear_error = self.calculate_linear_error(self.goal_pose.position, self.current_pose.position)
#         linear_vel = self.linear_pid.update(linear_error, dt)
        

#         angular_error = self.calculate_angular_error(self.goal_pose.orientation, self.current_pose.orientation)
#         angular_vel = self.angular_pid.update(angular_error, dt)

#         print("velocities")
#         print(linear_vel)
#         print(angular_vel)

#         print("error")

#         print(linear_error)
#         print(angular_error)

#         # cmd_vel = Twist()
#         # cmd_vel.linear.x = linear_vel
#         # cmd_vel.angular.z = angular_vel
        
#             # Check if the goal has been reached
#         if linear_error < self.goal_tolerance and abs(angular_error) < self.goal_angular_tolerance:
#             print("Goal reached!")
#             self.goal_pose = None  # Clear the goal
#             linear_vel = 0.0
#             angular_vel = 0.0

#         return linear_vel, angular_vel

#     def calculate_linear_error(self, goal_position, current_position):
#         x_error = goal_position.x - current_position.x
#         y_error = goal_position.y - current_position.y
#         linear_error = math.sqrt(x_error ** 2 + y_error ** 2)
#         return linear_error

#     #def calculate_angular_error(self, goal_orientation, current_orientation):
#     def calculate_angular_error(self, goal_orientation, current_orientation):
#         def quaternion_to_euler(quaternion):
#             x = quaternion.x
#             y = quaternion.y
#             z = quaternion.z
#             w = quaternion.w

#             t0 = +2.0 * (w * x + y * z)
#             t1 = +1.0 - 2.0 * (x * x + y * y)
#             roll = math.atan2(t0, t1)

#             t2 = +2.0 * (w * y - z * x)
#             t2 = +1.0 if t2 > +1.0 else t2
#             t2 = -1.0 if t2 < -1.0 else t2
#             pitch = math.asin(t2)

#             t3 = +2.0 * (w * z + x * y)
#             t4 = +1.0 - 2.0 * (y * y + z * z)
#             yaw = math.atan2(t3, t4)

#             return roll, pitch, yaw

#             # Calculate the angle between the current orientation and the vector pointing towards the goal position
#         goal_position = self.goal_pose.position
#         current_position = self.current_pose.position
#         goal_angle = math.atan2(goal_position.y - current_position.y, goal_position.x - current_position.x)
        
#         # Convert the goal angle to the range [-pi, pi]
#         goal_angle = math.atan2(math.sin(goal_angle), math.cos(goal_angle))
        
#         # Extract the current yaw angle from the quaternion
#         _, _, current_yaw = quaternion_to_euler(current_orientation)

#         # Calculate the angular error as the difference between the goal angle and the current yaw angle
#         angular_error = goal_angle - current_yaw

#         # Normalize the angular error to the range [-pi, pi]
#         if angular_error > math.pi:
#             angular_error -= 2 * math.pi
#         elif angular_error < -math.pi:
#             angular_error += 2 * math.pi

#         return angular_error
 
#     def calculate_velocities_callback(self):
        
#         linear_vel, angular_vel = self.calculate_velocities()
#         cmd_vel = Twist()
#         cmd_vel.linear.x = linear_vel
#         cmd_vel.angular.z = angular_vel
#         self.velocity_publisher.publish(cmd_vel)

# def main(args=None):
#     rclpy.init(args=args)
#     node = RobotController()
    
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()



# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist, PoseStamped
# from nav_msgs.msg import Odometry

# import math
# from geometry_msgs.msg import Vector3

# import tf2_ros
# import tf2_geometry_msgs

# import geometry_msgs.msg

# #from tf.transformations import quaternion_to_euler
# def quaternion_to_euler(q):
#     x = q.x
#     y = q.y
#     z = q.z
#     w = q.w

#     t0 = +2.0 * (w * x + y * z)
#     t1 = +1.0 - 2.0 * (x * x + y * y)
#     roll = math.atan2(t0, t1)

#     t2 = +2.0 * (w * y - z * x)
#     t2 = +1.0 if t2 > +1.0 else t2
#     t2 = -1.0 if t2 < -1.0 else t2
#     pitch = math.asin(t2)

#     t3 = +2.0 * (w * z + x * y)
#     t4 = +1.0 - 2.0 * (y * y + z * z)
#     yaw = math.atan2(t3, t4)

#     return roll, pitch, yaw

# def euler_from_quaternion(quaternion, node):
#     buffer = tf2_ros.Buffer()
#     listener = tf2_ros.TransformListener(buffer, node)

#     quaternion_msg = geometry_msgs.msg.Quaternion()
#     quaternion_msg.x = quaternion.x
#     quaternion_msg.y = quaternion.y
#     quaternion_msg.z = quaternion.z
#     quaternion_msg.w = quaternion.w

#     euler_roll_pitch_yaw = tf2_ros.transform_to_euler_angles(quaternion_msg, 'sxyz')

#     return euler_roll_pitch_yaw

# class ControlNode(Node):
#     def __init__(self):
#         super().__init__('control_node')
#         self.cmd_vel_publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
#         self.odom_subscription_ = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
#         self.simple_goal_subscription_ = self.create_subscription(PoseStamped, 'goal_pose', self.goal_callback, 10)
#         self.current_pose_ = PoseStamped()

#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

#     def odom_callback(self, msg: Odometry):
#         self.current_pose_ = msg.pose.pose

#     def goal_callback(self, msg: PoseStamped):
#         velocity = self.calculate_velocity(self.current_pose_, msg.pose)
#         self.cmd_vel_publisher_.publish(velocity)

#     def calculate_velocity(self, current_pose, goal_pose):
#         kp_linear = 0.5  # Proportional gain for linear velocity
#         kp_angular = 1.0  # Proportional gain for angular velocity
#         max_vel = 0.2  # Maximum linear velocity
#         threshold = 0.1  # Threshold for considering the goal reached

#         dx = goal_pose.position.x - current_pose.position.x
#         dy = goal_pose.position.y - current_pose.position.y
#         distance = math.sqrt(dx * dx + dy * dy)

#         if distance < threshold:
#             return Twist()  # Stop if within threshold

#         linear_velocity = min(kp_linear * distance, max_vel)

#         #current_euler = euler_from_quaternion(current_pose.orientation)[2]
#         current_euler = euler_from_quaternion(current_pose.orientation, self)[2]

#         desired_orientation = math.atan2(dy, dx)
#         error_angle = math.atan2(math.sin(desired_orientation - current_euler), math.cos(desired_orientation - current_euler))

#         angular_velocity = kp_angular * error_angle

#         return Twist(linear=Vector3(x=linear_velocity, y=0.0, z=0.0),
#                     angular=Vector3(x=0.0, y=0.0, z=angular_velocity))


# def main(args=None):
#     rclpy.init(args=args)
#     control_node = ControlNode()
#     rclpy.spin(control_node)
#     control_node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()


# # import rclpy
# # from rclpy.node import Node
# # from geometry_msgs.msg import Twist, PoseStamped
# # from nav_msgs.msg import Odometry
# # from tf2_ros import TransformListener, Buffer, LookupException, ConnectivityException, ExtrapolationException
# # import math

# # class RobotState:
# #     MOVE_FORWARD = 0
# #     TURN_AROUND = 1
# #     REVERSE = 2
# #     STOP = 3
# #     RECOVERY = 4

# # class SimpleRobotController(Node):
# #     def __init__(self):
# #         super().__init__('simple_robot_controller')
        
# #         # Configurable parameters
# #         self.max_linear_velocity = 0.1  # Maximum linear velocity (m/s)
# #         self.max_angular_velocity = 0.5  # Maximum angular velocity (rad/s)
# #         self.lookahead_distance = 0.5  # Lookahead distance for pure pursuit (m)
# #         self.forward_distance_threshold = 0.1  # Distance threshold to stop moving forward (m)
# #         self.turn_around_timeout = 5.0  # Timeout for TURN_AROUND state (seconds)
# #         self.recovery_timeout_factor = 1.3  # Factor to estimate recovery timeout based on the estimated time to target
        
# #         # State variables
# #         self.estimated_time_to_target = 0.0
# #         self.turn_around_start_time = None
# #         self.target_angle = None
# #         self.start_time = None
# #         self.current_position = None
# #         self.target_position = None
# #         self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
# #         self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
# #         self.goal_subscription = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

# #         self.tf_buffer = Buffer()
# #         self.tf_listener = TransformListener(self.tf_buffer, self)

# #         # Flags
# #         self.reached_target = False
# #         self.move_backward = True

# #         # State machine
# #         self.state = RobotState.STOP

# #         # Derate variables
# #         self.derate_factor = 1.0  # Initial derate factor
# #         self.recovery_cycles_without_error = 0

# #         # Wait for TF tree to be populated
# #         while rclpy.ok():
# #             try:                
# #                 transform = self.tf_buffer.lookup_transform("odom","base_link", rclpy.time.Time())
# #                 break
# #             except (LookupException, ConnectivityException, ExtrapolationException):
# #                 rclpy.spin_once(self, timeout_sec=0.1)
# #         self.get_logger().info("TF tree populated")

# #     def odom_callback(self, odom):
# #         pose = PoseStamped()
# #         pose.header = odom.header
# #         pose.pose = odom.pose.pose
# #         try:                       
# #             self.current_position = pose            
# #         except Exception as e:
# #             self.get_logger().error(f"TF buffer transform failed: {str(e)}")

# #     def goal_callback(self, goal):
# #         self.target_position = (goal.pose.position.x, goal.pose.position.y)
# #         self.reached_target = False  # Reset the flag when a new goal is received
# #         self.get_logger().info("Goal Received")
        
# #     def determine_state(self):
# #         if self.current_position is not None and self.target_position is not None:
# #             current_x, current_y = self.current_position.pose.position.x, self.current_position.pose.position.y
# #             target_x, target_y = self.target_position
# #             angle_to_target = math.atan2(target_y - current_y, target_x - current_x)
# #             if abs(angle_to_target - self.current_position.pose.orientation.z) < math.pi / 2:
# #                 self.state = RobotState.MOVE_FORWARD
# #             else:
# #                 self.state = RobotState.TURN_AROUND
# #         else:
# #             # Default to moving forward if current position or target position is not available
# #             self.state = RobotState.MOVE_FORWARD
# #         # Estimate time to target, this will be used to calculate when to go into recovery mode
# #         if self.state == RobotState.STOP and not self.reached_target:
# #             distance = math.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)
# #             self.estimated_time_to_target = distance / self.max_linear_velocity
# #             self.start_time = self.get_clock().now()
# #             self.state = RobotState.MOVE_FORWARD

# #     def calculate_velocities(self):        
# #         self.determine_state()

# #         linear_velocity = 0.0
# #         angular_velocity = 0.0

# #         if self.current_position is not None and self.target_position is not None:
# #             current_x, current_y = self.current_position.pose.position.x, self.current_position.pose.position.y
# #             target_x, target_y = self.target_position

# #             # Calculate the distance to the target
# #             distance = math.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)

# #             if self.state == RobotState.MOVE_FORWARD:
# #                 if distance < self.forward_distance_threshold:
# #                     self.state = RobotState.STOP
# #                 else:
# #                     # Calculate the angle to the target
# #                     angle_to_target = math.atan2(target_y - current_y, target_x - current_x)
# #                     # Calculate the angle difference
# #                     angle_diff = self.current_position.pose.orientation.z - angle_to_target
# #                     if angle_diff > math.pi:
# #                         angle_diff -= 2 * math.pi
# #                     elif angle_diff < -math.pi:
# #                         angle_diff += 2 * math.pi
# #                     # Calculate the lookahead point
# #                     lookahead_x = current_x + self.lookahead_distance * math.cos(self.current_position.pose.orientation.z)
# #                     lookahead_y = current_y + self.lookahead_distance * math.sin(self.current_position.pose.orientation.z)
# #                     # Calculate the curvature
# #                     curvature = 2 * lookahead_y / (self.lookahead_distance ** 2)
# #                     # Calculate the desired angular velocity using the pure pursuit algorithm
# #                     angular_velocity = self.max_angular_velocity * curvature
# #                     # Clip the angular velocity within the maximum angular velocity
# #                     angular_velocity = min(max(angular_velocity, -self.max_angular_velocity), self.max_angular_velocity)
# #                     # Calculate the desired linear velocity using the linear velocity and angular velocity relation
# #                     linear_velocity = self.max_linear_velocity / (1 + curvature ** 2)
# #                     # Clip the linear velocity within the maximum linear velocity
# #                     linear_velocity = min(max(linear_velocity, 0), self.max_linear_velocity)
# #             elif self.state == RobotState.TURN_AROUND:
# #                 # Rotate in place until the angle difference is within a threshold
# #                 angle_to_target = math.atan2(target_y - current_y, target_x - current_x)
# #                 angle_diff = self.current_position.pose.orientation.z - angle_to_target
# #                 if angle_diff > math.pi:
# #                     angle_diff -= 2 * math.pi
# #                 elif angle_diff < -math.pi:
# #                     angle_diff += 2 * math.pi
# #                 if abs(angle_diff) < math.radians(60):
# #                     self.state = RobotState.MOVE_FORWARD
# #                 else:
# #                     angular_velocity = self.max_angular_velocity if angle_diff < 0 else -self.max_angular_velocity
# #             elif self.state == RobotState.REVERSE:
# #                 linear_velocity = -self.max_linear_velocity / 2  # Reverse at half of the maximum linear velocity
# #                 angular_velocity = self.max_angular_velocity / 2  # Gentle turn while reversing

# #         return linear_velocity, angular_velocity

# #     def run(self):
# #         while rclpy.ok():
# #             linear_vel, angular_vel = self.calculate_velocities()

# #             # Create Twist message and publish velocities
# #             twist_msg = Twist()
# #             twist_msg.linear.x = linear_vel
# #             twist_msg.angular.z = angular_vel
# #             self.velocity_publisher.publish(twist_msg)

# #             # Enter recovery state if needed
# #             if self.state == RobotState.STOP and self.start_time is not None:
# #                 elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9  # Convert to seconds
# #                 if elapsed_time > self.recovery_timeout_factor * self.estimated_time_to_target:
# #                     if self.recovery_cycles_without_error == 2:
# #                         self.state = RobotState.REVERSE
# #                     else:
# #                         self.state = RobotState.RECOVERY
# #                     self.recovery_cycles_without_error += 1
# #                     self.derate_factor *= 0.9  # Reduce derate factor for next cycle
# #                     self.get_logger().info("Entering recovery state")

# #             # Reset derate factor and recovery cycle counter if no error in the recovery cycle
# #             if self.state != RobotState.RECOVERY:
# #                 self.derate_factor = 1.0
# #                 self.recovery_cycles_without_error = 0

# #             # Add a timeout for TURN_AROUND state
# #             if self.state == RobotState.TURN_AROUND:
# #                 if self.turn_around_start_time is None:
# #                     self.turn_around_start_time = self.get_clock().now()
# #                 elif (self.get_clock().now() - self.turn_around_start_time).nanoseconds / 1e9 > self.turn_around_timeout:
# #                     if self.recovery_cycles_without_error == 2:
# #                         self.state = RobotState.REVERSE
# #                     else:
# #                         self.state = RobotState.RECOVERY
# #                     self.recovery_cycles_without_error += 1
# #                     self.derate_factor *= 0.9  # Reduce derate factor for next cycle
# #                     self.get_logger().info("Entering recovery state due to TURN_AROUND timeout")
# #                     self.turn_around_start_time = None

# #             rclpy.spin_once(self)

# # def main(args=None):
# #     rclpy.init(args=args)

# #     controller = SimpleRobotController()
# #     controller.run()

# #     controller.destroy_node()
# #     rclpy.shutdown()

# # if __name__ == '__main__':
# #     main()


# # import rclpy
# # from rclpy.node import Node
# # from geometry_msgs.msg import Twist, PoseStamped
# # from nav_msgs.msg import Odometry
# # from tf2_ros import TransformListener, Buffer, LookupException, ConnectivityException, ExtrapolationException
# # import math

# # class RobotState:
# #     MOVE_FORWARD = 0
# #     TURN_AROUND = 1
# #     REVERSE = 2
# #     STOP = 3
# #     RECOVERY = 4

# # class SimpleRobotController(Node):
# #     def __init__(self):
# #         super().__init__('simple_robot_controller')
        
# #         # Configurable parameters
# #         self.max_linear_velocity = 00.1  # Maximum linear velocity (m/s)
# #         self.max_angular_velocity = 0.10  # Maximum angular velocity (rad/s)
# #         self.forward_distance_threshold = 0.1  # Distance threshold to stop moving forward (m)
# #         self.turn_around_timeout = 5.0  # Timeout for TURN_AROUND state (seconds)
# #         self.recovery_timeout_factor = 1.3  # Factor to estimate recovery timeout based on the estimated time to target
        
# #         # State variables
# #         self.estimated_time_to_target = 0.0
# #         self.turn_around_start_time = None
# #         self.target_angle = None
# #         self.start_time = None
# #         self.current_position = None
# #         self.target_position = None
# #         self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
# #         self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
# #         self.goal_subscription = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

# #         self.tf_buffer = Buffer()
# #         self.tf_listener = TransformListener(self.tf_buffer, self)

# #         # Flags
# #         self.reached_target = False
# #         self.move_backward = True

# #         # State machine
# #         self.state = RobotState.STOP

# #         # Derate variables
# #         self.derate_factor = 1.0  # Initial derate factor
# #         self.recovery_cycles_without_error = 0

# #         # Wait for TF tree to be populated
# #         while rclpy.ok():
# #             try:                
# #                 transform = self.tf_buffer.lookup_transform("odom","base_link", rclpy.time.Time())
# #                 break
# #             except (LookupException, ConnectivityException, ExtrapolationException):
# #                 rclpy.spin_once(self, timeout_sec=0.1)
# #         self.get_logger().info("TF tree populated")

# #     def odom_callback(self, odom):
# #         pose = PoseStamped()
# #         pose.header = odom.header
# #         pose.pose = odom.pose.pose
# #         try:                       
# #             self.current_position = pose            
# #         except Exception as e:
# #             self.get_logger().error(f"TF buffer transform failed: {str(e)}")

# #     def goal_callback(self, goal):
# #         self.target_position = (goal.pose.position.x, goal.pose.position.y)
# #         self.reached_target = False  # Reset the flag when a new goal is received
# #         self.get_logger().info("Goal Received")
        
# #     def determine_state(self):
# #         if self.current_position is not None and self.target_position is not None:
# #             current_x, current_y = self.current_position.pose.position.x, self.current_position.pose.position.y
# #             target_x, target_y = self.target_position
# #             angle_to_target = math.atan2(target_y - current_y, target_x - current_x)
# #             if abs(angle_to_target - self.current_position.pose.orientation.z) < math.pi / 2:
# #                 self.state = RobotState.MOVE_FORWARD
# #             else:
# #                 self.state = RobotState.TURN_AROUND
# #         else:
# #             # Default to moving forward if current position or target position is not available
# #             self.state = RobotState.MOVE_FORWARD
# #         # Estimate time to target, this will be used to calculate when to go into recovery mode
# #         if self.state == RobotState.STOP and not self.reached_target:
# #             distance = math.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)
# #             self.estimated_time_to_target = distance / self.max_linear_velocity
# #             self.start_time = self.get_clock().now()
# #             self.state = RobotState.MOVE_FORWARD

# #     def calculate_velocities(self):        
# #         self.determine_state()

# #         linear_velocity = 0.0
# #         angular_velocity = 0.0

# #         if self.current_position is not None and self.target_position is not None:
# #             current_x, current_y = self.current_position.pose.position.x, self.current_position.pose.position.y
# #             target_x, target_y = self.target_position

# #             # Calculate the distance to the target
# #             distance = math.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)

# #             angle_to_target = math.atan2(target_y - current_y, target_x - current_x)
# #             angle_diff = self.current_position.pose.orientation.z - angle_to_target
# #             if angle_diff > math.pi:
# #                 angle_diff -= 2 * math.pi
# #             elif angle_diff < -math.pi:
# #                 angle_diff += 2 * math.pi
            
# #             if self.state == RobotState.MOVE_FORWARD:
# #                 if distance < self.forward_distance_threshold:
# #                     self.state = RobotState.STOP
# #                 else:
# #                     linear_velocity = min(self.max_linear_velocity * distance, self.max_linear_velocity)
# #                     angular_velocity = self.max_angular_velocity * angle_diff * self.derate_factor
# #             elif self.state == RobotState.TURN_AROUND:
# #                 if self.target_angle is None:
# #                     self.target_angle = angle_to_target + math.pi
# #                 if angle_diff > 0:
# #                     self.target_angle -= 2 * math.pi

# #                 # Determine the shortest turning direction
# #                 if angle_diff < 0:
# #                     angular_velocity = self.max_angular_velocity
# #                 else:
# #                     angular_velocity = -self.max_angular_velocity

# #                 if abs(angle_diff) < 15:  # Threshold angle
# #                     self.state = RobotState.MOVE_FORWARD
# #                 else:
# #                     # Gradually reduce angular velocity when turning around
# #                     angular_velocity_decay = 0.9  # Decay factor for angular velocity
# #                     angular_velocity *= angular_velocity_decay
# #             elif self.state == RobotState.REVERSE:
# #                 linear_velocity = -0.5 * self.max_linear_velocity
# #                 angular_velocity = 0.5 * self.max_angular_velocity  # Gentle turn while reversing

# #         return linear_velocity, angular_velocity

# #     def run(self):
# #         while rclpy.ok():
# #             linear_vel, angular_vel = self.calculate_velocities()

# #             # Create Twist message and publish velocities
# #             twist_msg = Twist()
# #             twist_msg.linear.x = linear_vel
# #             twist_msg.angular.z = angular_vel
# #             self.velocity_publisher.publish(twist_msg)

# #             # Enter recovery state if needed
# #             if self.state == RobotState.STOP and self.start_time is not None:
# #                 elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9  # Convert to seconds
# #                 if elapsed_time > self.recovery_timeout_factor * self.estimated_time_to_target:
# #                     if self.recovery_cycles_without_error == 2:
# #                         self.state = RobotState.REVERSE
# #                     else:
# #                         self.state = RobotState.RECOVERY
# #                     self.recovery_cycles_without_error += 1
# #                     self.derate_factor *= 0.9  # Reduce derate factor for next cycle
# #                     self.get_logger().info("Entering recovery state")

# #             # Reset derate factor and recovery cycle counter if no error in the recovery cycle
# #             if self.state != RobotState.RECOVERY:
# #                 self.derate_factor = 1.0
# #                 self.recovery_cycles_without_error = 0

# #             # Add a timeout for TURN_AROUND state
# #             if self.state == RobotState.TURN_AROUND:
# #                 if self.turn_around_start_time is None:
# #                     self.turn_around_start_time = self.get_clock().now()
# #                 elif (self.get_clock().now() - self.turn_around_start_time).nanoseconds / 1e9 > self.turn_around_timeout:
# #                     if self.recovery_cycles_without_error == 2:
# #                         self.state = RobotState.REVERSE
# #                     else:
# #                         self.state = RobotState.RECOVERY
# #                     self.recovery_cycles_without_error += 1
# #                     self.derate_factor *= 0.9  # Reduce derate factor for next cycle
# #                     self.get_logger().info("Entering recovery state due to TURN_AROUND timeout")
# #                     #self.get_logger().info(self.recovery_cycles_without_error)
# #                     self.turn_around_start_time = None

# #             rclpy.spin_once(self)

# # def main(args=None):
# #     rclpy.init(args=args)

# #     controller = SimpleRobotController()
# #     controller.run()

# #     controller.destroy_node()
# #     rclpy.shutdown()

# # if __name__ == '__main__':
# #     main()
