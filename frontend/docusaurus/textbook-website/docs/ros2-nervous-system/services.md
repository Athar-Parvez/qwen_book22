---
sidebar_position: 4
---

# ROS 2 Services: Synchronous Request-Reply Communication

## Understanding Services in ROS 2

Services in ROS 2 provide synchronous request-reply communication between nodes. Unlike topics which offer asynchronous data streaming, services establish direct, blocking communication where a client sends a request and waits for a response.

### Service Characteristics

Services are ideal for:
- **One-time Requests**: Actions requiring immediate responses
- **Remote Procedure Calls**: Invoking functionality on another node
- **Blocking Operations**: Tasks that must complete before proceeding
- **State Queries**: Retrieving current state information

## Service Definition

Services consist of two message types:
- **Request**: Data sent from client to server
- **Response**: Data sent from server back to client

Service definition file (GetDistance.srv):
```
# Request
float64 x
float64 y
---
# Response
float64 distance
bool success
```

## Service Server Implementation

Creating a service server involves defining callbacks that process requests:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(
            AddTwoInts, 
            'add_two_ints', 
            self.add_two_ints_callback
        )

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Request: {request.a} + {request.b}')
        return response
```

## Service Client Implementation

Clients call services by sending requests and waiting for responses:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClientAsync(Node):
    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

## Service Management

### Common Commands
- `ros2 service list`: Show all active services
- `ros2 service info <service_name>`: Display service information
- `ros2 service call <service_name> <type> <args>`: Call a service from command line

### Service Interfaces
Services offer typed interfaces ensuring compatible request and response types between clients and servers.

## When to Use Services vs Topics

### Use Services For:
- Operations that should block until complete
- Request-response patterns like querying state
- Operations with clear input/output relationships
- Synchronous coordination between nodes

### Use Topics For:
- Continuous data streams
- Asynchronous notifications
- Broadcasting information to multiple recipients
- Event-driven architectures

## Advanced Service Features

### Service Quality of Service
Services support QoS settings similar to topics:
- **Reliability**: For ensuring service calls complete successfully
- **Deadline**: Time bounds for service completion
- **Liveliness**: Detecting if service providers are alive

### Service Introspection
Monitor service performance and usage:
- Track call frequency and response times
- Log service interactions for debugging
- Monitor for service failures

## Error Handling

### Service Failures
Handle situations where services become unavailable:
```python
if not self.cli.service_is_ready():
    self.get_logger().error('Service not ready')
    return
```

### Timeouts
Set timeouts for service calls to prevent indefinite blocking:
```python
future = self.cli.call_async(request)
try:
    rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
except Exception as e:
    self.get_logger().error(f'Service call failed: {e}')
```

Services provide essential synchronous communication for coordinated robotic operations, complementing the asynchronous nature of topic-based communication.