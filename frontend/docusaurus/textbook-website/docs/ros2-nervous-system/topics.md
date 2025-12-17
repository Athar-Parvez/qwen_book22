---
sidebar_position: 3
---

# ROS 2 Topics: Asynchronous Message Passing

## Understanding Topics in ROS 2

Topics enable asynchronous communication between nodes through a publish-subscribe pattern. Publishers send messages to topics, and subscribers receive messages from topics without direct connections between them.

### Publish-Subscribe Model

The publish-subscribe model offers several advantages:
- **Decoupling**: Publishers and subscribers operate independently
- **Scalability**: Multiple publishers/subscribers can connect to the same topic
- **Flexibility**: Nodes can join and leave the communication at any time

## Message Types

All messages published to topics must have a defined type with a schema:

- **Standard Messages**: Common types like sensor_msgs, geometry_msgs
- **Custom Messages**: User-defined types for domain-specific data
- **Message Definition Files**: .msg files that define field types and names

Example message definition (String.msg):
```
string data
```

## Quality of Service (QoS) Settings

QoS parameters control reliability and performance characteristics:

- **Reliability Policy**: Reliable (resend lost messages) or best-effort
- **Durability Policy**: Volatile (no historical data) or transient-local
- **History Policy**: Keep-all or keep-latest messages
- **Depth**: Maximum queue size for messages

### QoS Example
```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)
```

## Publisher Implementation

Publishers create and send messages to topics:

```python
import rclpy
from std_msgs.msg import String

def create_publisher(node, msg_type, topic_name, qos_profile):
    publisher = node.create_publisher(msg_type, topic_name, qos_profile)
    return publisher

def publish_message(publisher, message_data):
    msg = String()
    msg.data = message_data
    publisher.publish(msg)
```

## Subscriber Implementation

Subscribers receive and process messages from topics:

```python
def create_subscriber(node, msg_type, topic_name, callback, qos_profile):
    subscriber = node.create_subscription(
        msg_type,
        topic_name,
        callback,
        qos_profile
    )
    return subscriber

def message_callback(msg):
    print(f'Received: {msg.data}')
```

## Topic Management

### Common Commands
- `ros2 topic list`: Show all active topics
- `ros2 topic echo <topic_name>`: Print messages from a topic
- `ros2 topic info <topic_name>`: Display topic information
- `ros2 topic pub <topic_name> <type> <args>`: Publish to a topic from command line

### Topic Namespaces
Topics can be organized hierarchically using namespaces:
- `/sensor_data/lidar/scan`
- `/control/joint_commands`
- `/navigation/goal_pose`

## Performance Considerations

### Bandwidth Management
- Minimize message size to reduce network load
- Use appropriate data compression for large messages
- Consider frequency of message publication

### Latency vs. Throughput
- Balance message frequency with processing capabilities
- Choose appropriate QoS profiles for timing requirements
- Monitor buffer overruns and message drops

## Advanced Topics

### Typed Topic Interfaces
ROS 2 supports type-adapting communications allowing different message types to be connected with adapters.

### Transports
ROS 2 can use different transports for topic communications:
- **DDS**: Default transport for most use cases
- **UDP**: For broadcast communication
- **Shared Memory**: For high-performance intra-process communication

Topics form the backbone of ROS 2 communication, enabling flexible, distributed robotic systems.