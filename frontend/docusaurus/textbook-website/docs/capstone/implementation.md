---
sidebar_position: 3
---

# Capstone Implementation: Building Your Autonomous Humanoid

## Implementation Overview

This section provides a detailed guide to implementing your autonomous humanoid robot project, covering the essential components and their integration. The implementation follows a modular approach to ensure scalability and maintainability.

## System Architecture

### High-Level Architecture

The humanoid robot system consists of these interconnected subsystems:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Perception    │    │    Planning     │    │     Control     │
│                 │    │                 │    │                 │
│ • Vision System │◄──►│ • Path Planning │◄──►│ • Walking Gait  │
│ • Object Detect │    │ • Manipulation  │    │ • Arm Control   │
│ • Localization  │    │ • Task Planning │    │ • Balance Ctrl  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Behavior Engine                            │
│  • State Management      • Decision Making                    │
│  • Human Interaction     • Safety Monitoring                  │
└─────────────────────────────────────────────────────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│   Simulation    │    │   Real Robot    │
│                 │    │                 │
│ • Isaac Sim     │    │ • Hardware Int. │
│ • Gazebo Models │    │ • ROS 2 Bridge  │
└─────────────────┘    └─────────────────┘
```

### Core Implementation Modules

#### 1. Perception Module

```cpp
// perception_module.h
#ifndef PERCEPTION_MODULE_H
#define PERCEPTION_MODULE_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

class PerceptionModule : public rclcpp::Node
{
public:
    PerceptionModule();
    
private:
    // Publishers and subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr object_pose_pub_;
    
    // Computer vision components
    cv::dnn::Net detection_net_;
    cv::Mat camera_matrix_;
    
    // Callbacks
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    
    // Processing functions
    cv::Mat rosImageToCVMat(const sensor_msgs::msg::Image::SharedPtr& msg);
    std::vector<cv::Rect> detectObjects(const cv::Mat& image);
    geometry_msgs::msg::PoseStamped localizeObject(
        const cv::Rect& bbox, 
        const cv::Mat& camera_matrix
    );
};

#endif // PERCEPTION_MODULE_H
```

```cpp
// perception_module.cpp
#include "perception_module.h"

PerceptionModule::PerceptionModule() : Node("perception_module")
{
    // Initialize subscribers and publishers
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "camera/image_raw", 10,
        std::bind(&PerceptionModule::imageCallback, this, std::placeholders::_1)
    );
    
    object_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "detected_object", 10
    );
    
    // Load detection model (e.g., YOLO)
    detection_net_ = cv::dnn::readNetFromDarknet(
        "config/yolo.cfg", 
        "weights/yolo.weights"
    );
    
    // Initialize camera matrix (from calibration)
    // In practice, this would be loaded from calibration data
    camera_matrix_ = cv::Mat::eye(3, 3, CV_64F);
    camera_matrix_.at<double>(0, 0) = 525.0;  // fx
    camera_matrix_.at<double>(1, 1) = 525.0;  // fy
    camera_matrix_.at<double>(0, 2) = 319.5;  // cx
    camera_matrix_.at<double>(1, 2) = 239.5;  // cy
}

void PerceptionModule::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    cv::Mat image = rosImageToCVMat(msg);
    
    // Detect objects in the image
    std::vector<cv::Rect> detections = detectObjects(image);
    
    if (!detections.empty()) {
        // Localize the first detected object
        geometry_msgs::msg::PoseStamped object_pose = 
            localizeObject(detections[0], camera_matrix_);
        
        // Publish the object pose
        object_pose_pub_->publish(object_pose);
    }
}

cv::Mat PerceptionModule::rosImageToCVMat(const sensor_msgs::msg::Image::SharedPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return cv::Mat();
    }
    
    return cv_ptr->image;
}

std::vector<cv::Rect> PerceptionModule::detectObjects(const cv::Mat& image)
{
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0/255.0, 
                          cv::Size(416, 416), cv::Scalar(0,0,0), true, false);
    
    detection_net_.setInput(blob);
    std::vector<cv::Mat> outputs;
    detection_net_.forward(outputs, detection_net_.getUnconnectedOutLayersNames());
    
    // Process outputs to get bounding boxes
    // (Implementation would involve processing YOLO output tensors)
    std::vector<cv::Rect> detections;
    // ... detection processing logic ...
    
    return detections;
}

geometry_msgs::msg::PoseStamped PerceptionModule::localizeObject(
    const cv::Rect& bbox, 
    const cv::Mat& camera_matrix)
{
    geometry_msgs::msg::PoseStamped pose;
    
    // Calculate 3D position from 2D bounding box using camera parameters
    double center_x = bbox.x + bbox.width / 2.0;
    double center_y = bbox.y + bbox.height / 2.0;
    
    // This is a simplified example - real implementation would use depth data
    // or stereo vision to compute accurate 3D position
    pose.pose.position.x = (center_x - camera_matrix.at<double>(0, 2)) / 
                           camera_matrix.at<double>(0, 0) * 1.0; // depth estimate
    pose.pose.position.y = (center_y - camera_matrix.at<double>(1, 2)) / 
                           camera_matrix.at<double>(1, 1) * 1.0; // depth estimate
    pose.pose.position.z = 1.0; // depth estimate
    
    pose.pose.orientation.w = 1.0; // Identity orientation
    
    return pose;
}
```

#### 2. Planning Module

```python
# planning_module.py
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import String
import ompl.base as ob
import ompl.control as oc
from ompl import tools as opt

class PlanningModule(Node):
    def __init__(self):
        super().__init__('planning_module')
        
        # Publishers and subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped, 'goal_pose', self.goal_callback, 10
        )
        self.path_pub = self.create_publisher(
            Path, 'robot_path', 10
        )
        
        # Initialize OMPL planning components
        self.setup_space_and_planner()
        
        # Current robot state
        self.current_pose = PoseStamped()
        
    def setup_space_and_planner(self):
        # Define the state space (x, y, theta for 2D navigation)
        space = ob.SE2StateSpace()
        
        # Set bounds for the state space
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(-10)
        bounds.setHigh(10)
        space.setBounds(bounds)
        
        # Create space information
        self.si = ob.SpaceInformation(space)
        
        # Set state validity checker
        self.si.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        
        # Create and configure the planner
        self.planner = oc.RRT(self.si)
        self.si.setPlanner(self.planner)
        
    def is_state_valid(self, state):
        """
        Check if a state is valid (not in collision)
        """
        # In a real implementation, this would check against a map
        # For this example, we'll assume all states are valid
        x = state.getX()
        y = state.getY()
        
        # Simple boundary check
        if x < -10 or x > 10 or y < -10 or y > 10:
            return False
            
        # Add obstacle collision checking here
        # ...
        
        return True
    
    def plan_path(self, start_pose, goal_pose):
        """
        Plan a path from start_pose to goal_pose
        """
        # Clear previous planner data
        self.si.clearPlannerData()
        
        # Define start and goal states
        start = ob.State(self.si.getStateSpace())
        start().setX(start_pose.pose.position.x)
        start().setY(start_pose.pose.position.y)
        # start().setYaw(start_pose.pose.orientation.z)  # Simplified
        
        goal = ob.State(self.si.getStateSpace())
        goal().setX(goal_pose.pose.position.x)
        goal().setY(goal_pose.pose.position.y)
        # goal().setYaw(goal_pose.pose.orientation.z)  # Simplified
        
        # Set start and goal states
        self.si.setStartState(start)
        self.si.setGoalState(goal)
        
        # Plan the path
        solution = self.si.solve(5.0)  # 5-second time limit
        
        if solution:
            # Extract the path
            path = self.si.getSolutionPath()
            
            # Convert to ROS Path message
            ros_path = self.convert_path_to_ros(path)
            return ros_path
        else:
            self.get_logger().warn('No path found to goal')
            return None
    
    def convert_path_to_ros(self, ompl_path):
        """
        Convert OMPL path to ROS Path message
        """
        ros_path = Path()
        ros_path.header.frame_id = "map"
        
        states = ompl_path.getStates()
        for i, state in enumerate(states):
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "map"
            
            pose.pose.position.x = state.getX()
            pose.pose.position.y = state.getY()
            pose.pose.position.z = 0.0  # Ground plane
            
            # Set orientation to face direction of movement
            if i < len(states) - 1:
                next_state = states[i+1]
                dx = next_state.getX() - state.getX()
                dy = next_state.getY() - state.getY()
                
                yaw = np.arctan2(dy, dx)
                pose.pose.orientation.z = np.sin(yaw/2)
                pose.pose.orientation.w = np.cos(yaw/2)
            else:
                # Last point - maintain previous orientation
                pose.pose.orientation.z = 0.0
                pose.pose.orientation.w = 1.0
                
            ros_path.poses.append(pose)
        
        return ros_path
    
    def goal_callback(self, msg):
        """
        Handle new goal requests
        """
        # Get current robot pose (in practice, from localization)
        current_pose = self.get_current_pose()  # Implementation needed
        
        # Plan path to goal
        path = self.plan_path(current_pose, msg)
        
        if path:
            # Publish the computed path
            self.path_pub.publish(path)
        else:
            self.get_logger().error('Failed to plan path to goal')

    def get_current_pose(self):
        """
        Get current robot pose (would interface with localization system)
        """
        # Placeholder - in practice, this would come from localization
        current = PoseStamped()
        current.pose.position.x = 0.0
        current.pose.position.y = 0.0
        current.pose.position.z = 0.0  # Ground plane
        current.pose.orientation.w = 1.0  # Identity orientation
        return current
```

#### 3. Control Module

```python
# control_module.py
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64
from humanoid_robot_msgs.msg import WalkingGoal, WalkingFeedback, WalkingResult
from rclpy.action import ActionServer
import math

class WalkingController:
    def __init__(self, node):
        self.node = node
        self.joint_pub = node.create_publisher(JointState, 'joint_commands', 10)
        self.imu_sub = node.create_subscription(Imu, 'imu/data', self.imu_callback, 10)
        
        # Walking parameters
        self.step_height = 0.05  # meters
        self.step_length = 0.3   # meters
        self.step_time = 1.0     # seconds per step
        self.stride_frequency = 0.5  # steps per second
        
        # Robot state
        self.current_pose = Pose()
        self.current_twist = Twist()
        self.imu_data = Imu()
        self.balance_feedback = {'roll': 0.0, 'pitch': 0.0, 'zmp_x': 0.0, 'zmp_y': 0.0}
        
    def imu_callback(self, msg):
        """
        Process IMU data for balance control
        """
        self.imu_data = msg
        # Extract roll, pitch, yaw
        # Convert quaternion to Euler angles
        w, x, y, z = msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z
        self.balance_feedback['roll'] = math.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z)
        self.balance_feedback['pitch'] = math.asin(2.0 * (w * y - z * x))
        
    def compute_zero_moment_point(self):
        """
        Compute Zero Moment Point for dynamic balance
        """
        # Simplified ZMP calculation
        # In practice, this would use full dynamics model
        zmp_x = self.current_pose.position.x - (self.current_pose.position.z / 9.81) * self.current_twist.linear.x
        zmp_y = self.current_pose.position.y - (self.current_pose.position.z / 9.81) * self.current_twist.linear.y
        
        return zmp_x, zmp_y
    
    def generate_walking_pattern(self, distance, direction=0.0):
        """
        Generate walking pattern for a given distance and direction
        """
        # Calculate number of steps needed
        num_steps = int(abs(distance) / self.step_length)
        if num_steps == 0:
            num_steps = 1  # At least one step
        
        # Generate footstep pattern
        footsteps = []
        current_distance = 0.0
        step_size = distance / num_steps
        
        for i in range(num_steps):
            # Calculate foot position for this step
            step_x = current_distance + step_size * 0.5
            step_y = 0.1 * (-1)**i  # Alternate feet
            step_z = self.step_height
            
            # Apply rotation if needed
            if direction != 0.0:
                rotated_x = step_x * math.cos(direction) - step_y * math.sin(direction)
                rotated_y = step_x * math.sin(direction) + step_y * math.cos(direction)
                step_x, step_y = rotated_x, rotated_y
            
            footsteps.append({
                'x': step_x,
                'y': step_y,
                'z': step_z,
                'step_time': self.step_time
            })
            
            current_distance += step_size
        
        return footsteps
    
    def execute_walking_step(self, step_params):
        """
        Execute a single walking step with balance control
        """
        # Generate joint trajectories for this step
        joint_trajectory = self.generate_joint_trajectory(step_params)
        
        # Execute trajectory with balance feedback
        for point in joint_trajectory:
            # Apply balance corrections based on IMU data
            corrected_joints = self.apply_balance_correction(point)
            
            # Publish joint commands
            self.publish_joint_commands(corrected_joints)
            
            # Wait for next control cycle
            self.node.get_clock().sleep_for(rclpy.time.Duration(seconds=0.01))
    
    def generate_joint_trajectory(self, step_params):
        """
        Generate joint trajectory for a single step
        """
        # Simplified joint trajectory generation
        # In practice, this would use inverse kinematics and dynamics
        trajectory = []
        
        # Break down step into phases: lift, swing, place, balance
        total_points = 100
        for i in range(total_points):
            t = i / (total_points - 1)  # Normalized time [0, 1]
            
            # Generate joint positions for this time step
            joints = {}
            
            # Simplified walking pattern
            joints['left_hip_roll'] = 0.1 * math.sin(math.pi * t)
            joints['right_hip_roll'] = -0.1 * math.sin(math.pi * t)
            
            joints['left_hip_pitch'] = -0.2 * math.sin(math.pi * t)
            joints['right_hip_pitch'] = -0.2 * math.sin(math.pi * t)
            
            joints['left_knee'] = 0.2 * math.sin(2 * math.pi * t)
            joints['right_knee'] = 0.2 * math.sin(2 * math.pi * t)
            
            joints['left_ankle'] = 0.05 * math.sin(2 * math.pi * t)
            joints['right_ankle'] = -0.05 * math.sin(2 * math.pi * t)
            
            trajectory.append(joints)
        
        return trajectory
    
    def apply_balance_correction(self, joints):
        """
        Apply balance corrections based on sensor feedback
        """
        # Get current balance state
        roll = self.balance_feedback['roll']
        pitch = self.balance_feedback['pitch']
        zmp_x, zmp_y = self.compute_zero_moment_point()
        
        # Adjust joints to maintain balance
        corrected_joints = joints.copy()
        
        # Compensate for roll
        corrected_joints['left_hip_roll'] += 2.0 * roll
        corrected_joints['right_hip_roll'] -= 2.0 * roll
        
        # Compensate for pitch
        corrected_joints['left_hip_pitch'] += 1.5 * pitch
        corrected_joints['right_hip_pitch'] += 1.5 * pitch
        
        # Adjust ankle positions based on ZMP
        corrected_joints['left_ankle'] += 0.5 * zmp_y
        corrected_joints['right_ankle'] -= 0.5 * zmp_y
        
        return corrected_joints
    
    def publish_joint_commands(self, joints):
        """
        Publish joint commands to robot
        """
        joint_msg = JointState()
        joint_msg.header.stamp = self.node.get_clock().now().to_msg()
        
        for joint_name, position in joints.items():
            joint_msg.name.append(joint_name)
            joint_msg.position.append(position)
        
        self.joint_pub.publish(joint_msg)

class ControlModule(Node):
    def __init__(self):
        super().__init__('control_module')
        
        # Initialize walking controller
        self.walking_controller = WalkingController(self)
        
        # Action server for walking commands
        self.walking_action_server = ActionServer(
            self,
            WalkingGoal,
            'walk_to_waypoint',
            self.execute_walking_callback
        )
        
        # Navigation command subscriber
        self.nav_cmd_sub = self.create_subscription(
            Twist, 'cmd_vel', self.nav_command_callback, 10
        )
        
    def execute_walking_callback(self, goal_handle):
        """
        Execute walking action server
        """
        self.get_logger().info(f'Executing walking goal: {goal_handle.request}')
        
        # Extract distance and direction from goal
        distance = goal_handle.request.distance
        direction = goal_handle.request.direction
        
        # Generate walking pattern
        footsteps = self.walking_controller.generate_walking_pattern(distance, direction)
        
        # Execute each step in the pattern
        for i, step in enumerate(footsteps):
            self.walking_controller.execute_walking_step(step)
            
            # Publish feedback
            feedback_msg = WalkingFeedback()
            feedback_msg.percentage_completed = float(i + 1) / len(footsteps) * 100.0
            goal_handle.publish_feedback(feedback_msg)
        
        # Complete the action
        result = WalkingResult()
        result.success = True
        goal_handle.succeed()
        
        return result
    
    def nav_command_callback(self, msg):
        """
        Handle velocity commands for navigation
        """
        linear_x = msg.linear.x
        angular_z = msg.angular.z
        
        # Convert to walking parameters
        distance = linear_x * 1.0  # Assuming 1 second duration
        direction = angular_z
        
        # Generate and execute walking pattern
        footsteps = self.walking_controller.generate_walking_pattern(distance, direction)
        for step in footsteps:
            self.walking_controller.execute_walking_step(step)
```

## Integration and Communication

### ROS 2 Launch File

```xml
<!-- launch/humanoid_robot.launch.xml -->
<launch>
  <!-- Parameters -->
  <arg name="use_sim_time" default="false"/>
  
  <!-- Perception node -->
  <node pkg="humanoid_robot" exec="perception_module" name="perception_module">
    <param name="use_sim_time" value="$(var use_sim_time)"/>
    <remap from="camera/image_raw" to="/camera/rgb/image_raw"/>
  </node>
  
  <!-- Planning node -->
  <node pkg="humanoid_robot" exec="planning_module" name="planning_module">
    <param name="use_sim_time" value="$(var use_sim_time)"/>
  </node>
  
  <!-- Control node -->
  <node pkg="humanoid_robot" exec="control_module" name="control_module">
    <param name="use_sim_time" value="$(var use_sim_time)"/>
  </node>
  
  <!-- Isaac ROS Visual SLAM (if available) -->
  <node pkg="isaac_ros_visual_slam" exec="visual_slam_node" name="visual_slam">
    <param name="enable_rectified_pose" value="true"/>
    <param name="map_frame" value="map"/>
    <param name="publish_odom_tf" value="true"/>
  </node>
  
  <!-- VLA Integration Node -->
  <node pkg="humanoid_robot" exec="vla_integration_node" name="vla_integration">
    <param name="use_sim_time" value="$(var use_sim_time)"/>
    <param name="model_path" value="models/vla_model.pt"/>
  </node>
</launch>
```

## Simulation Integration

### Gazebo Model Configuration

```xml
<!-- models/humanoid_robot/model.sdf -->
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="humanoid_robot">
    <!-- Robot base link -->
    <link name="base_link">
      <pose>0 0 1.0 0 0 0</pose>
      <inertial>
        <mass>50.0</mass>
        <inertia>
          <ixx>1.0</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>1.0</iyy>
          <iyz>0.0</iyz>
          <izz>1.0</izz>
        </inertia>
      </inertial>
      
      <visual name="base_visual">
        <geometry>
          <box>
            <size>0.5 0.3 0.8</size>
          </box>
        </geometry>
      </visual>
      
      <collision name="base_collision">
        <geometry>
          <box>
            <size>0.5 0.3 0.8</size>
          </box>
        </geometry>
      </collision>
    </link>
    
    <!-- Legs -->
    <joint name="left_hip_joint" type="revolute">
      <parent>base_link</parent>
      <child>left_thigh</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
        </limit>
      </axis>
    </joint>
    
    <link name="left_thigh">
      <pose>0 -0.15 -0.4 0 0 0</pose>
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <iyy>0.1</iyy>
          <izz>0.05</izz>
        </inertia>
      </inertial>
      <!-- Additional link elements -->
    </link>
    
    <!-- Additional joints and links for arms, head, etc. -->
    
    <!-- Sensors -->
    <sensor name="camera" type="camera">
      <camera name="head_camera">
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
      </camera>
      <always_on>1</always_on>
      <update_rate>30.0</update_rate>
      <visualize>true</visualize>
    </sensor>
    
    <sensor name="imu" type="imu">
      <always_on>1</always_on>
      <update_rate>100</update_rate>
    </sensor>
  </model>
</sdf>
```

## VLA Integration

### Vision-Language-Action Node

```python
# vla_integration_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import torch
import numpy as np
import cv2
from transformers import AutoTokenizer, AutoModel
from PIL import Image as PILImage

class VLAIntegrationNode(Node):
    def __init__(self):
        super().__init__('vla_integration_node')
        
        # Load VLA model
        self.setup_vla_model()
        
        # Subscribers and publishers
        self.command_sub = self.create_subscription(
            String, 'robot_command', self.command_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )
        self.action_pub = self.create_publisher(
            Twist, 'cmd_vel', 10
        )
        
        # Store latest image for processing
        self.latest_image = None
        
    def setup_vla_model(self):
        """
        Load and initialize the VLA model
        """
        try:
            # Load pre-trained model (this is a simplified example)
            # In practice, you would load a specialized VLA model
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
            
            # For this example, we'll use a simple neural network
            # In practice, you would use a sophisticated VLA model
            self.vla_model = self.build_simple_vla_model()
            
            self.get_logger().info('VLA model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load VLA model: {e}')
    
    def build_simple_vla_model(self):
        """
        Build a simplified VLA model for demonstration
        """
        # This is a placeholder - in practice you would load a 
        # sophisticated model that can process vision and language
        import torch.nn as nn
        
        class SimpleVLA(nn.Module):
            def __init__(self):
                super().__init__()
                # Vision processing
                self.vision_conv = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                
                # Language processing
                self.lang_fc = nn.Sequential(
                    nn.Linear(768, 256),  # BERT base output size
                    nn.ReLU()
                )
                
                # Fusion and action generation
                self.fusion = nn.Sequential(
                    nn.Linear(64*120*160 + 256, 512),  # Adjust sizes based on actual dimensions
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 3)  # [linear_x, angular_z, gripper]
                )
            
            def forward(self, vision_input, lang_input):
                # Process vision input
                vision_features = self.vision_conv(vision_input)
                vision_features = vision_features.view(vision_features.size(0), -1)  # Flatten
                
                # Process language input
                lang_features = self.lang_fc(lang_input)
                
                # Concatenate features
                fused_features = torch.cat([vision_features, lang_features], dim=1)
                
                # Generate action
                action = self.fusion(fused_features)
                return action
        
        return SimpleVLA()
    
    def image_callback(self, msg):
        """
        Store the latest image for VLA processing
        """
        try:
            # Convert ROS image to OpenCV
            np_image = np.frombuffer(msg.data, dtype=np.uint8)
            np_image = np_image.reshape((msg.height, msg.width, 3))
            
            # Convert BGR to RGB
            cv_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
            
            # Resize for model input
            cv_image = cv2.resize(cv_image, (640, 480))
            
            # Normalize
            normalized_image = cv_image.astype(np.float32) / 255.0
            normalized_image = np.transpose(normalized_image, (2, 0, 1))  # HWC to CHW
            normalized_image = np.expand_dims(normalized_image, 0)  # Add batch dimension
            
            self.latest_image = normalized_image
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def command_callback(self, msg):
        """
        Process natural language command using VLA
        """
        command = msg.data
        
        if self.latest_image is not None:
            try:
                # Process command with VLA model
                action = self.process_command_with_vla(command, self.latest_image)
                
                # Publish action
                self.publish_action(action)
                
                self.get_logger().info(f'Executed command: {command}')
            except Exception as e:
                self.get_logger().error(f'Error processing command: {e}')
        else:
            self.get_logger().warn('No image available for VLA processing')
    
    def process_command_with_vla(self, command, image):
        """
        Process command and image with VLA model
        """
        # Tokenize command
        tokens = self.tokenizer(
            command, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # Get text embeddings
        with torch.no_grad():
            text_outputs = self.text_encoder(**tokens)
            text_embeddings = text_outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        
        # Convert image to tensor
        image_tensor = torch.from_numpy(image).float()
        
        # Run through VLA model
        with torch.no_grad():
            action_tensor = self.vla_model(image_tensor, text_embeddings)
            action = action_tensor.cpu().numpy()[0]  # Remove batch dimension
        
        return action
    
    def publish_action(self, action):
        """
        Publish action to robot control system
        """
        twist_msg = Twist()
        
        # Map VLA output to Twist command
        twist_msg.linear.x = float(action[0])  # Forward/backward movement
        twist_msg.angular.z = float(action[1])  # Rotation
        
        # Publish the command
        self.action_pub.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    vla_node = VLAIntegrationNode()
    
    try:
        rclpy.spin(vla_node)
    except KeyboardInterrupt:
        pass
    finally:
        vla_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Safety and Validation

### Safety Manager

```python
# safety_manager.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import numpy as np

class SafetyManager(Node):
    def __init__(self):
        super().__init__('safety_manager')
        
        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10
        )
        self.cmd_sub = self.create_subscription(
            Twist, 'cmd_vel', self.command_callback, 10
        )
        
        # Publishers
        self.safety_pub = self.create_publisher(Bool, 'safety_status', 10)
        self.emergency_stop_pub = self.create_publisher(Twist, 'cmd_vel_emergency', 10)
        
        # Safety parameters
        self.safety_threshold = 0.5  # meters for obstacle detection
        self.tilt_threshold = 0.3    # radians for tilt detection
        self.enabled = True
        
        # Current robot state
        self.current_cmd = Twist()
        self.tilt = {'roll': 0.0, 'pitch': 0.0}
        self.obstacle_distances = []
        
        # Start safety monitoring
        self.safety_timer = self.create_timer(0.1, self.safety_check)
    
    def laser_callback(self, msg):
        """
        Process laser scan for obstacle detection
        """
        # Filter out invalid ranges
        valid_ranges = [r for r in msg.ranges if 0 < r < msg.range_max]
        self.obstacle_distances = valid_ranges
    
    def imu_callback(self, msg):
        """
        Process IMU data for tilt detection
        """
        # Convert quaternion to roll/pitch
        w, x, y, z = msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z
        
        # Roll (rotation around x-axis)
        self.tilt['roll'] = np.arctan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z)
        
        # Pitch (rotation around y-axis)
        self.tilt['pitch'] = np.arcsin(2.0 * (w * y - z * x))
    
    def command_callback(self, msg):
        """
        Store the current command for safety processing
        """
        self.current_cmd = msg
    
    def safety_check(self):
        """
        Perform safety checks and apply emergency measures if needed
        """
        if not self.enabled:
            return
        
        # Check for tilt beyond threshold
        tilt_violation = abs(self.tilt['roll']) > self.tilt_threshold or \
                        abs(self.tilt['pitch']) > self.tilt_threshold
        
        # Check for obstacles
        obstacle_violation = any(d < self.safety_threshold for d in self.obstacle_distances) \
                           if self.obstacle_distances else False
        
        # Determine safety status
        is_safe = not (tilt_violation or obstacle_violation)
        
        # Publish safety status
        safety_msg = Bool()
        safety_msg.data = is_safe
        self.safety_pub.publish(safety_msg)
        
        # Trigger emergency stop if unsafe
        if not is_safe:
            self.emergency_stop()
    
    def emergency_stop(self):
        """
        Execute emergency stop procedure
        """
        self.get_logger().warn('EMERGENCY STOP TRIGGERED')
        
        # Publish zero velocity command
        stop_cmd = Twist()
        self.emergency_stop_pub.publish(stop_cmd)
        
        # Disable further commands
        self.enabled = False
        
        # Additional safety measures could be implemented here
        # For example, engaging mechanical brakes, logging incident, etc.
    
    def reset_safety(self):
        """
        Reset safety system after emergency stop
        """
        self.enabled = True
        self.get_logger().info('Safety system reset')
```

This implementation provides a comprehensive foundation for your autonomous humanoid robot project, covering perception, planning, control, VLA integration, and safety systems. Each module is designed to work together through the ROS 2 communication framework, following best practices for robotics software development.