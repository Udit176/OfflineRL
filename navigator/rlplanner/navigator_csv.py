#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutonomousNavigator ROS2 Node

This node integrates CSV-based scenarios with A* navigation and path following for a robot. It processes scenarios from a CSV file, executing navigation routines based on successful entries. The node handles map processing, path planning, trajectory recording, and robot control to move between specified start and goal positions using both A* and predefined CSV paths.

Key Features:
- Reads and executes successful scenarios from a CSV file.
- Utilizes A* for path planning between points.
- Follows predefined paths from the CSV.
- Records trajectories, computes distances and durations.
- Handles map updates and integrates IMU data for pose estimation.
"""

import sys
import os
import csv
import numpy as np
import time
from copy import copy
import heapq
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist, Quaternion
from sensor_msgs.msg import Imu

rl_csv_path = "/path/to/rl_csv_data.csv"

def retrieve_time(clock_instance):
    return float(clock_instance.now().nanoseconds / 1e9)

class Vertex:
    """Represents a vertex in the navigation graph."""
    def __init__(self, identifier):
        self.identifier = identifier
        self.adjacent_vertices = []
        self.edge_weights = []

    def connect_vertices(self, vertices_list, weights_list):
        self.adjacent_vertices.extend(vertices_list)
        self.edge_weights.extend(weights_list)

class GraphStructure:
    """Manages the graph structure for navigation."""
    def __init__(self, label):
        self.graph_dict = {}
        self.start_vertex = None
        self.end_vertex = None

    def insert_vertex(self, vertex):
        self.graph_dict[vertex.identifier] = vertex

class OptimalPathFinder:
    """Implements the A* algorithm for pathfinding."""
    def __init__(self, graph_data, logger_callback=None):
        self.graph_data = graph_data
        self.logger_callback = logger_callback

    def estimate_cost(self, vertex_a, vertex_b):
        coord_a = tuple(map(int, vertex_a.split(',')))
        coord_b = tuple(map(int, vertex_b.split(',')))
        return np.hypot(coord_b[0] - coord_a[0], coord_b[1] - coord_a[1])

    def compute_optimal_path(self, origin, destination):
        open_set = []
        heapq.heappush(open_set, (0, origin))
        came_from = {}
        g_score = {v: float('inf') for v in self.graph_data.graph_dict}
        g_score[origin] = 0
        f_score = {v: float('inf') for v in self.graph_data.graph_dict}
        f_score[origin] = self.estimate_cost(origin, destination)

        if self.logger_callback:
            self.logger_callback().info(f"Origin: {origin}")
            self.logger_callback().info(f"Destination: {destination}")

        while open_set:
            current_vertex = heapq.heappop(open_set)[1]

            if current_vertex == destination:
                return self.retrace_path(came_from, current_vertex)
            for idx, neighbor in enumerate(self.graph_data.graph_dict[current_vertex].adjacent_vertices):
                tentative_g_score = g_score[current_vertex] + self.graph_data.graph_dict[current_vertex].edge_weights[idx]
                if tentative_g_score < g_score[neighbor.identifier]:
                    came_from[neighbor.identifier] = current_vertex
                    g_score[neighbor.identifier] = tentative_g_score
                    f_score[neighbor.identifier] = tentative_g_score + self.estimate_cost(neighbor.identifier, destination)
                    heapq.heappush(open_set, (f_score[neighbor.identifier], neighbor.identifier))
        return []

    def retrace_path(self, came_from, current_vertex):
        total_path = [current_vertex]
        while current_vertex in came_from:
            current_vertex = came_from[current_vertex]
            total_path.append(current_vertex)
        total_path.reverse()
        return total_path

class TerrainAnalyzer:
    """Processes and analyzes the occupancy grid map."""
    def __init__(self):
        self.occupancy_map = None
        self.binary_image = None
        self.map_ready = False
        self.threshold_occupied = 65
        self.threshold_free = 25
        self.navigation_graph = GraphStructure("navigation_graph")

    def update_map(self, occupancy_data: OccupancyGrid):
        self.occupancy_map = occupancy_data
        map_width = self.occupancy_map.info.width
        map_height = self.occupancy_map.info.height
        map_data = np.array(self.occupancy_map.data).reshape((map_height, map_width))

        unique_vals, counts = np.unique(map_data, return_counts=True)
        for idx in range(len(unique_vals)):
            print(f"STAT: {unique_vals[idx]}:{counts[idx]}")

        binary_map = np.where(map_data > self.threshold_occupied, 1, 0)
        binary_map[map_data == -1] = 1

        self.binary_image = binary_map
        self.map_ready = True
        self.log_info("Map analysis completed.")

    def log_info(self, message):
        print(message)

    def __alter_pixel(self, image_array, x, y, value, absolute_mode):
        if (0 <= x < image_array.shape[0]) and (0 <= y < image_array.shape[1]):
            if absolute_mode:
                image_array[x][y] = value
            else:
                image_array[x][y] += value

    def __expand_obstacle(self, kernel_matrix, image_array, x, y, absolute_mode):
        kernel_half_x = int(kernel_matrix.shape[0] // 2)
        kernel_half_y = int(kernel_matrix.shape[1] // 2)
        if (kernel_half_x == 0) and (kernel_half_y == 0):
            self.__alter_pixel(image_array, x, y, kernel_matrix[0][0], absolute_mode)
        else:
            for kx in range(x - kernel_half_x, x + kernel_half_x + 1):
                for ky in range(y - kernel_half_y, y + kernel_half_y + 1):
                    if 0 <= kx < image_array.shape[0] and 0 <= ky < image_array.shape[1]:
                        self.__alter_pixel(image_array, kx, ky, kernel_matrix[kx - x + kernel_half_x][ky - y + kernel_half_y], absolute_mode)

    def expand_map(self, kernel_matrix, absolute_mode=True):
        self.expanded_image = np.zeros(self.binary_image.shape)
        for x in range(self.binary_image.shape[0]):
            for y in range(self.binary_image.shape[1]):
                if self.binary_image[x][y] == 1:
                    self.__expand_obstacle(kernel_matrix, self.expanded_image, x, y, absolute_mode)
        value_range = np.max(self.expanded_image) - np.min(self.expanded_image)
        if value_range == 0:
            value_range = 1
        self.expanded_image = (self.expanded_image - np.min(self.expanded_image)) / value_range

    def construct_graph_from_map(self):
        for x in range(self.binary_image.shape[0]):
            for y in range(self.binary_image.shape[1]):
                if self.expanded_image[x][y] == 0:
                    vertex = Vertex(f'{x},{y}')
                    self.navigation_graph.insert_vertex(vertex)
        for x in range(self.binary_image.shape[0]):
            for y in range(self.binary_image.shape[1]):
                if self.expanded_image[x][y] == 0:
                    current_vertex = self.navigation_graph.graph_dict[f'{x},{y}']
                    neighbor_vertices = []
                    edge_weights = []
                    neighbor_offsets = [
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1),
                        (-1, -1),
                        (-1, 1),
                        (1, -1),
                        (1, 1)
                    ]
                    for offset in neighbor_offsets:
                        nx, ny = x + offset[0], y + offset[1]
                        if 0 <= nx < self.binary_image.shape[0] and 0 <= ny < self.binary_image.shape[1]:
                            if self.expanded_image[nx][ny] == 0:
                                neighbor_id = f'{nx},{ny}'
                                if neighbor_id in self.navigation_graph.graph_dict:
                                    neighbor_vertex = self.navigation_graph.graph_dict[neighbor_id]
                                    neighbor_vertices.append(neighbor_vertex)
                                    weight = 1 if abs(offset[0]) + abs(offset[1]) == 1 else np.sqrt(2)
                                    edge_weights.append(weight)
                    current_vertex.connect_vertices(neighbor_vertices, edge_weights)

    def create_gaussian_kernel(self, size, sigma=1):
        half_size = int(size) // 2
        x_vals, y_vals = np.mgrid[-half_size:half_size + 1, -half_size:half_size + 1]
        normalizer = 1 / (2.0 * np.pi * sigma ** 2)
        gaussian_matrix = np.exp(-((x_vals ** 2 + y_vals ** 2) / (2.0 * sigma ** 2))) * normalizer
        value_range = np.max(gaussian_matrix) - np.min(gaussian_matrix)
        scaled_matrix = (gaussian_matrix - np.min(gaussian_matrix)) / value_range
        return scaled_matrix

    def create_rectangular_kernel(self, size, fill_value):
        return np.ones((size, size)) * fill_value

    def visualize_path(self, path_sequence):
        path_visual = copy(self.expanded_image)
        for node in path_sequence:
            coords = tuple(map(int, node.split(',')))
            path_visual[coords] = 0.5
        return path_visual

    def generate_graph_map(self):
        node_positions = [tuple(map(int, node_id.split(','))) for node_id in self.navigation_graph.graph_dict]
        if not node_positions:
            self.log_info("Graph is empty. Cannot generate occupancy grid.")
            return None

        min_x = min(pos[0] for pos in node_positions)
        max_x = max(pos[0] for pos in node_positions)
        min_y = min(pos[1] for pos in node_positions)
        max_y = max(pos[1] for pos in node_positions)

        map_height = max_x - min_x + 1
        map_width = max_y - min_y + 1

        self.log_info(f"Graph Map Bounds: x [{min_x}, {max_x}], y [{min_y}, {max_y}]")
        self.log_info(f"Graph Map Dimensions: width={map_width}, height={map_height}")

        graph_map_array = np.ones((map_height, map_width), dtype=np.int8) * 100

        for node_id in self.navigation_graph.graph_dict:
            x, y = map(int, node_id.split(','))
            adjusted_x = x - min_x
            adjusted_y = y - min_y
            graph_map_array[adjusted_x, adjusted_y] = 0

        from nav_msgs.msg import OccupancyGrid
        graph_map_msg = OccupancyGrid()
        graph_map_msg.header.frame_id = 'map'

        graph_map_msg.info.resolution = self.occupancy_map.info.resolution
        graph_map_msg.info.width = map_width
        graph_map_msg.info.height = map_height
        graph_map_msg.info.origin.position.x = self.occupancy_map.info.origin.position.x + min_y * self.occupancy_map.info.resolution
        graph_map_msg.info.origin.position.y = self.occupancy_map.info.origin.position.y + min_x * self.occupancy_map.info.resolution
        graph_map_msg.info.origin.position.z = 0.0
        graph_map_msg.info.origin.orientation = self.occupancy_map.info.origin.orientation

        graph_map_msg.data = graph_map_array.flatten().tolist()

        return graph_map_msg

class AutonomousNavigator(Node):
    """Executes navigation scenarios from a CSV file using A* and predefined paths."""
    def __init__(self, node_name='AutonomousNavigator'):
        super().__init__(node_name)
        self.planned_path = Path()
        self.target_pose = None
        self.robot_pose = None
        self.integrated_pose = None

        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__on_goal_received, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/robot/amcl_pose', self.__on_pose_update, 10)
        self.create_subscription(OccupancyGrid, '/robot/map', self.__on_map_update, 10)
        self.create_subscription(Imu, '/robot/imu/data', self.__on_imu_data, 50)

        self.path_publisher = self.create_publisher(Path, 'global_plan', 10)
        self.velocity_publisher = self.create_publisher(Twist, '/robot/cmd_vel', 10)
        self.graph_map_publisher = self.create_publisher(OccupancyGrid, '/map_astar', 10)
        self.integrated_pose_publisher = self.create_publisher(PoseStamped,'/fused_pose/amcl_imu',50)

        self.execution_rate = self.create_rate(10)

        self.terrain_analyzer = TerrainAnalyzer()

        self.is_path_ready = False
        self.current_waypoint_index = 0

        self.cross_track_kp = 0.4
        self.cross_track_ki = 0.5
        self.cross_track_kd = 0.0
        self.heading_kp = 0.15
        self.heading_ki = 0.5

        self.cross_track_integral = 0.0
        self.heading_integral = 0.0
        self.max_integral_value = 1.0
        self.max_heading_integral = 0.5
        self.max_turn_rate = 0.5

        self.path_lookahead_distance = 0.5

        self.previous_time = time.time()
        self.previous_cross_track_error = 0.0

        self.heading_error_limit = np.deg2rad(15)
        self.final_heading_error_limit = np.deg2rad(5)

        self.last_imu_timestamp = None
        self.last_amcl_timestamp = None
        self.previous_velocity_command = None
        self.last_velocity_command_time = None

        self.csv_data = []
        self.current_csv_index = 0
        self.load_csv_data()
        self.scenario_state = "WAIT_FOR_MAP_AND_POSE"
        self.phase_start_time = None
        self.timeout_duration = 120.0

        self.record_positions = False
        self.trajectory_positions = []
        self.start_record_time = None
        self.last_pose_x = None
        self.last_pose_y = None
        self.accumulated_distance = 0.0

        self.goal_reached = False

    def load_csv_data(self):
        if not os.path.exists(rl_csv_path):
            self.get_logger().warn(f"CSV file not found at {rl_csv_path}")
            return
        with open(rl_csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.csv_data.append(row)
        self.get_logger().info(f"Loaded {len(self.csv_data)} CSV rows.")

    def start_recording_trajectory(self):
        self.trajectory_positions = []
        self.start_record_time = time.time()
        self.record_positions = True
        self.accumulated_distance = 0.0
        self.last_pose_x = None
        self.last_pose_y = None

    def stop_recording_trajectory(self):
        self.record_positions = False
        duration = time.time() - self.start_record_time if self.start_record_time is not None else 0.0
        return duration, self.accumulated_distance, self.trajectory_positions

    def reset_controller(self):
        self.cross_track_integral = 0.0
        self.heading_integral = 0.0

    def __on_map_update(self, data: OccupancyGrid):
        if not self.terrain_analyzer.map_ready:
            self.terrain_analyzer.update_map(data)
            robot_diameter = 0.4
            safety_buffer = 0.2
            total_inflation = robot_diameter + safety_buffer
            kernel_dimension = int(np.ceil(total_inflation / self.terrain_analyzer.occupancy_map.info.resolution))
            if kernel_dimension % 2 == 0:
                kernel_dimension += 1
            inflation_kernel = self.terrain_analyzer.create_rectangular_kernel(kernel_dimension, 1)
            self.terrain_analyzer.expand_map(inflation_kernel, absolute_mode=True)
            self.terrain_analyzer.construct_graph_from_map()
            self.get_logger().info('Completed map processing.')
            self.publish_graph_map()

    def __on_goal_received(self, data):
        self.target_pose = data
        self.get_logger().info(
            f'Received Goal: x={self.target_pose.pose.position.x:.4f}, y={self.target_pose.pose.position.y:.4f}')

    def __on_pose_update(self, data):
        self.robot_pose = PoseStamped()
        self.robot_pose.header = data.header
        self.robot_pose.pose = data.pose.pose

        self.integrated_pose = PoseStamped()
        self.integrated_pose.header = data.header
        self.integrated_pose.pose = data.pose.pose

        self.last_amcl_timestamp = retrieve_time(self.get_clock())

    def __on_imu_data(self, data):
        current_timestamp = retrieve_time(self.get_clock())
        if self.integrated_pose is None:
            self.last_imu_timestamp = current_timestamp
            return

        if self.last_imu_timestamp is None:
            self.last_imu_timestamp = current_timestamp
            return

        time_delta = current_timestamp - self.last_imu_timestamp
        self.last_imu_timestamp = current_timestamp

        angular_velocity_z = data.angular_velocity.z
        delta_yaw = angular_velocity_z * time_delta

        current_orientation = self.integrated_pose.pose.orientation
        current_yaw = self.extract_yaw((
            current_orientation.x,
            current_orientation.y,
            current_orientation.z,
            current_orientation.w
        ))

        updated_yaw = current_yaw + delta_yaw
        new_orientation = self.construct_quaternion(updated_yaw)
        self.integrated_pose.pose.orientation = new_orientation

        if self.previous_velocity_command is not None:
            linear_velocity_x = self.previous_velocity_command.linear.x
            delta_x = (linear_velocity_x * np.cos(current_yaw)) * time_delta
            delta_y = (linear_velocity_x * np.sin(current_yaw)) * time_delta

            self.integrated_pose.pose.position.x += delta_x
            self.integrated_pose.pose.position.y += delta_y

    def publish_graph_map(self):
        graph_map_msg = self.terrain_analyzer.generate_graph_map()
        if graph_map_msg is not None:
            graph_map_msg.header.stamp = self.get_clock().now().to_msg()
            self.graph_map_publisher.publish(graph_map_msg)
            self.get_logger().info('Published graph map.')
        else:
            self.get_logger().warn('Graph map is empty, unable to publish.')

    def extract_yaw(self, quaternion):
        x, y, z, w = quaternion
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw_angle = np.arctan2(siny_cosp, cosy_cosp)
        return yaw_angle

    def construct_quaternion(self, yaw_angle):
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = np.sin(yaw_angle / 2.0)
        q.w = np.cos(yaw_angle / 2.0)
        return q

    def control_robot(self, linear_velocity, angular_velocity):
        cmd = Twist()
        cmd.linear.x = linear_velocity
        cmd.angular.z = angular_velocity
        self.velocity_publisher.publish(cmd)

        self.previous_velocity_command = cmd
        self.last_velocity_command_time = retrieve_time(self.get_clock())

    def convert_world_to_map(self, x_world, y_world):
        resolution = self.terrain_analyzer.occupancy_map.info.resolution
        origin_x = self.terrain_analyzer.occupancy_map.info.origin.position.x
        origin_y = self.terrain_analyzer.occupancy_map.info.origin.position.y
        x_map = int((y_world - origin_y) / resolution)
        y_map = int((x_world - origin_x) / resolution)
        return (x_map, y_map)

    def convert_map_to_world(self, x_map, y_map):
        resolution = self.terrain_analyzer.occupancy_map.info.resolution
        origin_x = self.terrain_analyzer.occupancy_map.info.origin.position.x
        origin_y = self.terrain_analyzer.occupancy_map.info.origin.position.y
        x_world = y_map * resolution + origin_x + resolution / 2.0
        y_world = x_map * resolution + origin_y + resolution / 2.0
        return x_world, y_world

    def locate_nearest_free_node(self, coordinates):
        x, y = coordinates
        max_search_radius = max(self.terrain_analyzer.binary_image.shape)
        for offset in range(max_search_radius):
            for dx in range(-offset, offset + 1):
                for dy in range(-offset, offset + 1):
                    nx, ny = x + dx, y + dy
                    node_id = f"{nx},{ny}"
                    if node_id in self.terrain_analyzer.navigation_graph.graph_dict:
                        return node_id
        return None

    def perform_a_star_planning(self, start_pose, goal_pose):
        path_sequence = Path()
        path_sequence.header.frame_id = 'map'
        self.get_logger().info(
            f'A* Planning:\n> Start: ({start_pose.position.x}, {start_pose.position.y})\n> Goal: ({goal_pose.position.x}, {goal_pose.position.y})')

        start_coords = self.convert_world_to_map(start_pose.position.x, start_pose.position.y)
        goal_coords = self.convert_world_to_map(goal_pose.position.x, goal_pose.position.y)

        self.get_logger().info(f"Start Coordinates: {start_coords}")
        self.get_logger().info(f"Goal Coordinates: {goal_coords}")

        start_node_id = self.locate_nearest_free_node(start_coords)
        goal_node_id = self.locate_nearest_free_node(goal_coords)

        if start_node_id is None or goal_node_id is None:
            self.get_logger().warn('Unable to locate start or goal node in the graph.')
            return None

        path_finder = OptimalPathFinder(self.terrain_analyzer.navigation_graph, logger_callback=self.get_logger)
        node_sequence = path_finder.compute_optimal_path(start_node_id, goal_node_id)

        if not node_sequence:
            self.get_logger().warn('A* algorithm failed to find a path.')
            return None
        else:
            self.get_logger().info(f"Computed Path Length: {len(node_sequence)}")

        for node_id in node_sequence:
            x_idx, y_idx = map(int, node_id.split(','))
            x_world, y_world = self.convert_map_to_world(x_idx, y_idx)
            waypoint = PoseStamped()
            waypoint.header.frame_id = 'map'
            waypoint.pose.position.x = x_world
            waypoint.pose.position.y = y_world
            waypoint.pose.position.z = 0.0
            waypoint.pose.orientation.w = 1.0
            path_sequence.poses.append(waypoint)
        return path_sequence

    def find_closest_waypoint(self, path, current_pose):
        min_distance = float('inf')
        index = self.current_waypoint_index
        robot_x = current_pose.position.x
        robot_y = current_pose.position.y
        for i in range(self.current_waypoint_index, len(path.poses)):
            waypoint = path.poses[i]
            wp_x = waypoint.pose.position.x
            wp_y = waypoint.pose.position.y
            distance = np.hypot(wp_x - robot_x, wp_y - robot_y)
            if distance < min_distance:
                min_distance = distance
                index = i
            if distance > min_distance + 0.5:
                break
        return index

    def follow_path(self, current_pose):
        if self.current_waypoint_index >= len(self.planned_path.poses):
            return 0.0, 0.0

        robot_x = current_pose.position.x
        robot_y = current_pose.position.y
        quaternion = (
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w)
        robot_yaw = self.extract_yaw(quaternion)

        index = self.find_closest_waypoint(self.planned_path, current_pose)
        self.current_waypoint_index = index

        look_ahead_distance = self.path_lookahead_distance
        path_len = len(self.planned_path.poses)
        look_ahead_index = index
        accumulated_distance = 0.0
        for i in range(index, path_len - 1):
            wp_x = self.planned_path.poses[i].pose.position.x
            wp_y = self.planned_path.poses[i].pose.position.y
            next_wp_x = self.planned_path.poses[i + 1].pose.position.x
            next_wp_y = self.planned_path.poses[i + 1].pose.position.y
            segment_distance = np.hypot(next_wp_x - wp_x, next_wp_y - wp_y)
            accumulated_distance += segment_distance
            if accumulated_distance >= look_ahead_distance:
                look_ahead_index = i + 1
                break
        else:
            look_ahead_index = path_len - 1

        goal_x = self.planned_path.poses[-1].pose.position.x
        goal_y = self.planned_path.poses[-1].pose.position.y
        goal_distance = np.hypot(goal_x - robot_x, goal_y - robot_y)

        if goal_distance < 0.1:
            final_orientation = self.planned_path.poses[-1].pose.orientation
            temp_quaternion = (
                final_orientation.x,
                final_orientation.y,
                final_orientation.z,
                final_orientation.w)
            goal_yaw = self.extract_yaw(temp_quaternion)
        else:
            goal_pose = self.planned_path.poses[look_ahead_index].pose
            gx = goal_pose.position.x
            gy = goal_pose.position.y
            goal_yaw = np.arctan2(gy - robot_y, gx - robot_x)

        heading_error = goal_yaw - robot_yaw
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        path_x = self.planned_path.poses[index].pose.position.x
        path_y = self.planned_path.poses[index].pose.position.y
        dx = path_x - robot_x
        dy = path_y - robot_y
        path_direction = np.arctan2(dy, dx)
        cross_track_error = np.sin(robot_yaw - path_direction) * np.hypot(dx, dy)

        dt = time.time() - self.previous_time
        self.previous_time = time.time()
        self.cross_track_integral += cross_track_error * dt
        self.cross_track_integral = np.clip(self.cross_track_integral, -self.max_integral_value, self.max_integral_value)

        self.heading_integral += heading_error * dt
        self.heading_integral = np.clip(self.heading_integral, -self.max_heading_integral, self.max_heading_integral)

        derivative_cross_track_error = (cross_track_error - self.previous_cross_track_error) / dt if dt > 0 else 0.0
        self.previous_cross_track_error = cross_track_error

        is_near_goal_without_stop = goal_distance < 0.2 and abs(heading_error) > self.final_heading_error_limit
        is_near_goal_with_stop = goal_distance < 0.2 and abs(heading_error) <= self.final_heading_error_limit

        if abs(heading_error) > self.heading_error_limit or is_near_goal_without_stop:
            linear_speed = 0.0
            angular_speed = self.heading_kp * heading_error + self.heading_ki * self.heading_integral
            angular_speed = np.clip(angular_speed, -self.max_turn_rate, self.max_turn_rate)
            self.cross_track_integral = 0.0
        else:
            steering_correction = heading_error + np.arctan2(
                self.cross_track_kp * cross_track_error + self.cross_track_ki * self.cross_track_integral + self.cross_track_kd * derivative_cross_track_error,
                1.0)

            max_linear_speed = 0.2
            linear_speed = max_linear_speed
            angular_speed = np.clip(steering_correction * self.heading_kp, -self.max_turn_rate, self.max_turn_rate)
            self.heading_integral = 0.0

        if is_near_goal_with_stop:
            linear_speed = 0.0
            angular_speed = 0.0
            self.is_path_ready = False
            self.reset_controller()
            self.goal_reached = True
            self.get_logger().info(f"Goal reached!")
            self.control_robot(0.0, 0.0)
            return 0.0, 0.0

        return linear_speed, angular_speed

    def set_path(self, path: Path):
        self.planned_path = path
        if path is not None and len(path.poses) > 0:
            self.is_path_ready = True
            self.current_waypoint_index = 0
            self.path_publisher.publish(self.planned_path)
            self.get_logger().info('Published planned path.')
        else:
            self.is_path_ready = False
            self.get_logger().warn('No path to follow.')

    def goto_waypoints(self, path: Path):
        self.set_path(path)
        self.goal_reached = False
        self.start_recording_trajectory()
        start_t = time.time()
        while rclpy.ok() and not self.goal_reached:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.integrated_pose is None:
                continue

            if time.time() - start_t > self.timeout_duration:
                self.get_logger().warn("Timeout reached while following path.")
                self.control_robot(0.0, 0.0)
                break

            linear_speed, angular_speed = (0.0, 0.0)
            if self.is_path_ready:
                linear_speed, angular_speed = self.follow_path(self.integrated_pose.pose)
                self.control_robot(linear_speed, angular_speed)
            else:
                self.control_robot(0.0, 0.0)

            self.integrated_pose_publisher.publish(self.integrated_pose)
            self.record_trajectory_point()

        self.control_robot(0.0, 0.0)
        duration, distance, traj = self.stop_recording_trajectory()
        self.get_logger().info(f"Path follow finished. Duration: {duration:.2f}s, Distance: {distance:.2f}m, ReachedGoal={self.goal_reached}")
        return self.goal_reached, duration, distance

    def record_trajectory_point(self):
        if self.record_positions and self.integrated_pose is not None:
            t = time.time() - self.start_record_time
            x = self.integrated_pose.pose.position.x
            y = self.integrated_pose.pose.position.y
            if self.last_pose_x is not None and self.last_pose_y is not None:
                segment = np.hypot(x - self.last_pose_x, y - self.last_pose_y)
                self.accumulated_distance += segment
            self.last_pose_x = x
            self.last_pose_y = y
            self.trajectory_positions.append((t, x, y))

    def execute_scenario(self, start_str, goal_str, csv_path_str):
        start_coords = self.parse_coordinates(start_str)
        goal_coords = self.parse_coordinates(goal_str)
        csv_path_coords = self.parse_path_list(csv_path_str)

        start_pose = PoseStamped()
        start_pose.header.frame_id = "map"
        start_pose.pose.position.x = start_coords[0]
        start_pose.pose.position.y = start_coords[1]
        start_pose.pose.orientation.w = 1.0

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.pose.position.x = goal_coords[0]
        goal_pose.pose.position.y = goal_coords[1]
        goal_pose.pose.orientation.w = 1.0

        self.get_logger().info("Moving to Start via A*")
        a_star_path = self.perform_a_star_planning(self.integrated_pose.pose, start_pose.pose)
        if a_star_path is None or len(a_star_path.poses) == 0:
            self.get_logger().warn("Cannot reach start position with A*. Skipping scenario.")
            return
        self.goto_waypoints(a_star_path)

        self.get_logger().info("Following CSV path")
        csv_path_msg = Path()
        csv_path_msg.header.frame_id = 'map'
        for (x, y) in csv_path_coords:
            p = PoseStamped()
            p.header.frame_id = 'map'
            p.pose.position.x = x
            p.pose.position.y = y
            p.pose.orientation.w = 1.0
            csv_path_msg.poses.append(p)
        self.goto_waypoints(csv_path_msg)

        self.get_logger().info("Returning to Start via A*")
        a_star_path_back = self.perform_a_star_planning(self.integrated_pose.pose, start_pose.pose)
        if a_star_path_back is not None and len(a_star_path_back.poses) > 0:
            self.goto_waypoints(a_star_path_back)
        else:
            self.get_logger().warn("Cannot return to start. Continuing.")

        self.get_logger().info("From Start to Goal via A*")
        a_star_path_goal = self.perform_a_star_planning(self.integrated_pose.pose, goal_pose.pose)
        if a_star_path_goal is not None and len(a_star_path_goal.poses) > 0:
            self.goto_waypoints(a_star_path_goal)
        else:
            self.get_logger().warn("Cannot reach goal from start now.")

    def parse_coordinates(self, coord_str):
        coord_str = coord_str.strip().strip('"')
        coord_str = coord_str.replace("(", "").replace(")", "")
        parts = coord_str.split(",")
        x = float(parts[0])
        y = float(parts[1])
        return (x, y)

    def parse_path_list(self, path_str):
        path_str = path_str.strip().strip('"')
        path_str = path_str.strip('[]')
        coords = []
        segments = path_str.split("),")
        for seg in segments:
            seg = seg.replace("(", "").replace(")", "")
            seg = seg.strip()
            if seg == "":
                continue
            parts = seg.split(",")
            x = float(parts[0])
            y = float(parts[1])
            coords.append((x, y))
        return coords

    def execute(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.terrain_analyzer.map_ready and self.integrated_pose is not None:
                break
            self.get_logger().info("Waiting for map and pose...")
            time.sleep(1.0)

        self.get_logger().info("Starting CSV scenario execution...")
        for i, row in enumerate(self.csv_data):
            success = row["Success"].strip().lower()
            if success != 'y':
                self.get_logger().info(f"Skipping row {i} because Success={row['Success']}")
                continue

            start_str = row["Start"]
            goal_str = row["Goal"]
            path_str = row["Path"]

            self.get_logger().info(f"Executing scenario {i}: Start={start_str}, Goal={goal_str}")
            self.execute_scenario(start_str, goal_str, path_str)

        self.get_logger().info("All scenarios completed.")

def main(args=None):
    rclpy.init(args=args)
    navigator = AutonomousNavigator(node_name='AutonomousNavigator')

    try:
        navigator.execute()
    except KeyboardInterrupt:
        navigator.get_logger().info("Navigator node interrupted by user.")
    finally:
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
