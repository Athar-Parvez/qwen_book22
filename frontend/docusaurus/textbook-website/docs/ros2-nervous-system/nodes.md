---
sidebar_position: 2
---

# ROS 2 Nodes: The Building Blocks of Robotic Systems

## Understanding ROS 2 Nodes

A ROS 2 node is a process that performs computation within a ROS graph. Nodes are the fundamental building blocks of any ROS 2 system, encapsulating functionality into manageable, distributed units.

### Node Properties

Every ROS 2 node includes:

- **Unique Name**: Identifies the node within the ROS graph
- **Parameters**: Configurable values that customize behavior
- **Topics**: Communication channels for publishing and subscribing
- **Services**: Synchronous request/reply communication
- **Actions**: Asynchronous goal-oriented communication patterns

## Creating Nodes

Nodes are typically implemented in C++ or Python using ROS 2 client libraries (rclcpp for C++, rclpy for Python):

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World'
        self.publisher.publish(msg)
```

import InteractiveCodeExample from '@site/src/components/interactive-examples/InteractiveCodeExample';

<InteractiveCodeExample
  title="ROS 2 Node Example"
  language="python"
  description="A simple ROS 2 node that publishes 'Hello World' messages to a topic"
  code={`import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()\n`}
  output={`[INFO] [1634567890.123456789]: Publishing: "Hello World: 0"
[INFO] [1634567890.623456789]: Publishing: "Hello World: 1"
[INFO] [1634567891.123456789]: Publishing: "Hello World: 2"`}
/>

## Node Lifecycle

ROS 2 nodes follow a well-defined lifecycle:

1. **Unconfigured**: Node created but not yet configured
2. **Inactive**: Node configured but not executing
3. **Active**: Node running and performing computations
4. **Finalized**: Node shutting down

This lifecycle enables safer robot state management and graceful failure recovery.

## Node Composition

For improved performance and reduced overhead, multiple nodes can be composed into a single process:

- **Components**: Individual node functionalities that can be combined
- **Composite Nodes**: Multiple components running in one process
- **Benefits**: Reduced inter-process communication overhead

## Best Practices for Node Design

### Modularity
Design nodes with single responsibilities, promoting reuse and maintainability:
- Each node should focus on a specific function
- Separate perception, planning, and control into distinct nodes
- Use composition to create complex behaviors

### Error Handling
Implement robust error handling strategies:
- Graceful degradation when components fail
- Proper exception handling
- Clear logging for debugging

### Resource Management
Optimize resource usage:
- Efficient memory management
- Proper cleanup of resources
- Monitoring of computational load

## Debugging and Monitoring Nodes

ROS 2 provides several tools for node inspection:

- `ros2 node list`: View active nodes
- `ros2 node info <node_name>`: Detailed node information
- `rqt_graph`: Visual representation of the ROS graph
- `ros2 launch`: Managing groups of nodes

## Advanced Node Patterns

### Parameter Server Integration
Nodes can dynamically adjust behavior based on parameters:

```python
self.declare_parameter('frequency', 10)
frequency = self.get_parameter('frequency').value
```

### Node Groups
Organize related nodes for collective control and resource allocation:
- Group nodes for multi-threading
- Share resources between related functions
- Coordinate startup and shutdown sequences

Understanding nodes is crucial for building robust, scalable robotic systems with ROS 2.