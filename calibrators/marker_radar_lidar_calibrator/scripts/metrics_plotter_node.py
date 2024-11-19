#!/usr/bin/env python3

import math
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from tier4_calibration_msgs.msg import CalibrationMetrics  # Adjust the import for the new message type


class MetricsPlotter:
    def __init__(self):
        self.fig = None
        self.subplots = {}
        self.metrics_data = {}
        self.m_to_cm = 100

        # Define plot colors and styles
        self.color_distance_o = "C0o-"  # Circle marker, solid line for distance
        self.color_yaw_o = "C1o-"      # Circle marker, solid line for yaw
        self.color_distance = "C0"     # Solid color for distance (e.g., fill areas)
        self.color_yaw = "C1"          # Solid color for yaw (e.g., fill areas)

    def initialize_figure(self, methods):
        self.fig, self.axes = plt.subplots(
            nrows=2 * len(methods), ncols=2, figsize=(8, 6 * len(methods))
        )
        self.fig.canvas.manager.set_window_title("Metrics plotter")
        self.subplots = {method: {} for method in methods}

        for idx, method in enumerate(methods):
            self.subplots[method] = {
                "crossval_distance": self.axes[idx * 2, 0],
                "crossval_yaw": self.axes[idx * 2, 1],
                "average_distance": self.axes[idx * 2 + 1, 0],
                "average_yaw": self.axes[idx * 2 + 1, 1],
            }

        plt.tight_layout()
        plt.pause(0.1)

    def initialize_metrics(self, methods):
        for method in methods:
            if method not in self.metrics_data:
                self.metrics_data[method] = {
                    "num_of_reflectors_list": [],
                    "calibration_distance_error_list": [],
                    "calibration_yaw_error_list": [],
                    "crossval_sample_list": [],
                    "crossval_distance_error_list": [],
                    "crossval_yaw_error_list": [],
                    "std_crossval_distance_error_list": [],
                    "std_crossval_yaw_error_list": [],
                }


    def is_delete_operation(self, method, msg_array):
        if method not in self.metrics_data:
            return False  # Skip if the method is not initialized
        return (
            self.metrics_data[method]["num_of_reflectors_list"]
            and msg_array[0] < self.metrics_data[method]["num_of_reflectors_list"][-1]
        )


    def remove_avg_error_from_list(self, method):
        metrics = self.metrics_data[method]
        for _ in range(min(2, len(metrics["num_of_reflectors_list"]))):
            metrics["calibration_distance_error_list"].pop()
            metrics["calibration_yaw_error_list"].pop()
            metrics["num_of_reflectors_list"].pop()

    def plot_label_and_set_xy_lim(self):
        if not hasattr(self, "axes"):
            return  # Skip if the axes are not initialized

        for method, subplots in self.subplots.items():
            subplots["crossval_distance"].set_title(f"{method}\nCross-validation error: distance")
            subplots["crossval_distance"].set_xlabel("Number of tracks")
            subplots["crossval_distance"].set_ylabel("Distance error [cm]")

            subplots["crossval_yaw"].set_title(f"{method}\nCross-validation error: yaw")
            subplots["crossval_yaw"].set_xlabel("Number of tracks")
            subplots["crossval_yaw"].set_ylabel("Yaw error [deg]")

            subplots["average_distance"].set_title(f"{method}\nAverage error: distance")
            subplots["average_distance"].set_xlabel("Number of tracks")
            subplots["average_distance"].set_ylabel("Distance error [cm]")

            subplots["average_yaw"].set_title(f"{method}\nAverage error: yaw")
            subplots["average_yaw"].set_xlabel("Number of tracks")
            subplots["average_yaw"].set_ylabel("Yaw error [deg]")

        for ax in self.axes.flat:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))


    def update_metrics(self, msg):
        # Extract methods from the incoming message
        methods_in_msg = [method.method_name for method in msg.method_metrics]

        # Initialize figure and metrics for new methods
        if not self.metrics_data or set(self.metrics_data.keys()) != set(methods_in_msg):
            self.initialize_figure(methods_in_msg)
            self.initialize_metrics(methods_in_msg)

        # Update metrics for each method in the message
        for method in msg.method_metrics:
            method_name = method.method_name
            metrics = self.metrics_data[method_name]
            metrics["num_of_reflectors_list"].append(msg.num_of_converged_tracks)
            metrics["calibration_distance_error_list"].append(
                method.calibrated_distance_error * self.m_to_cm
            )
            metrics["calibration_yaw_error_list"].append(method.calibrated_yaw_error)
            metrics["crossval_sample_list"] = msg.num_of_samples
            metrics["crossval_distance_error_list"] = [
                value * self.m_to_cm for value in method.avg_crossval_calibrated_distance_error
            ]
            metrics["crossval_yaw_error_list"] = method.avg_crossval_calibrated_yaw_error
            metrics["std_crossval_distance_error_list"] = [
                value * self.m_to_cm for value in method.std_crossval_calibrated_distance_error
            ]
            metrics["std_crossval_yaw_error_list"] = method.std_crossval_calibrated_yaw_error


    def draw_subplots(self):
        for method, subplots in self.subplots.items():
            metrics = self.metrics_data[method]

            # Clear previous plots
            subplots["crossval_distance"].clear()
            subplots["crossval_yaw"].clear()
            subplots["average_distance"].clear()
            subplots["average_yaw"].clear()

            # Draw cross-validation plots
            if metrics["crossval_sample_list"] and metrics["crossval_distance_error_list"]:
                subplots["crossval_distance"].plot(
                    metrics["crossval_sample_list"],
                    metrics["crossval_distance_error_list"],
                    self.color_distance_o,
                )
            if metrics["crossval_sample_list"] and metrics["crossval_yaw_error_list"]:
                subplots["crossval_yaw"].plot(
                    metrics["crossval_sample_list"],
                    metrics["crossval_yaw_error_list"],
                    self.color_yaw_o,
                )

            # Draw average error plots
            if metrics["num_of_reflectors_list"] and metrics["calibration_distance_error_list"]:
                subplots["average_distance"].plot(
                    metrics["num_of_reflectors_list"],
                    metrics["calibration_distance_error_list"],
                    self.color_distance_o,
                )
            if metrics["num_of_reflectors_list"] and metrics["calibration_yaw_error_list"]:
                subplots["average_yaw"].plot(
                    metrics["num_of_reflectors_list"],
                    metrics["calibration_yaw_error_list"],
                    self.color_yaw_o,
                )

            # Annotate the last value for average distance/yaw
            if metrics["num_of_reflectors_list"]:
                if metrics["calibration_distance_error_list"]:
                    subplots["average_distance"].annotate(
                        f"{metrics['calibration_distance_error_list'][-1]:.2f}",
                        xy=(
                            metrics["num_of_reflectors_list"][-1],
                            metrics["calibration_distance_error_list"][-1],
                        ),
                        color=self.color_distance,
                    )
                if metrics["calibration_yaw_error_list"]:
                    subplots["average_yaw"].annotate(
                        f"{metrics['calibration_yaw_error_list'][-1]:.2f}",
                        xy=(
                            metrics["num_of_reflectors_list"][-1],
                            metrics["calibration_yaw_error_list"][-1],
                        ),
                        color=self.color_yaw,
                    )

            # Annotate the last value for cross-validation distance/yaw
            if metrics["crossval_sample_list"]:
                if metrics["crossval_distance_error_list"]:
                    subplots["crossval_distance"].annotate(
                        f"{metrics['crossval_distance_error_list'][-1]:.2f}",
                        xy=(
                            metrics["crossval_sample_list"][-1],
                            metrics["crossval_distance_error_list"][-1],
                        ),
                        color=self.color_distance,
                    )
                if metrics["crossval_yaw_error_list"]:
                    subplots["crossval_yaw"].annotate(
                        f"{metrics['crossval_yaw_error_list'][-1]:.2f}",
                        xy=(
                            metrics["crossval_sample_list"][-1],
                            metrics["crossval_yaw_error_list"][-1],
                        ),
                        color=self.color_yaw,
                    )

            # Draw standard deviation bands for cross-validation
            if (
                metrics["crossval_sample_list"]
                and metrics["crossval_distance_error_list"]
                and metrics["std_crossval_distance_error_list"]
            ):
                subplots["crossval_distance"].fill_between(
                    metrics["crossval_sample_list"],
                    np.array(metrics["crossval_distance_error_list"])
                    - np.array(metrics["std_crossval_distance_error_list"]),
                    np.array(metrics["crossval_distance_error_list"])
                    + np.array(metrics["std_crossval_distance_error_list"]),
                    color=self.color_distance,
                    alpha=0.3,
                )
            if (
                metrics["crossval_sample_list"]
                and metrics["crossval_yaw_error_list"]
                and metrics["std_crossval_yaw_error_list"]
            ):
                subplots["crossval_yaw"].fill_between(
                    metrics["crossval_sample_list"],
                    np.array(metrics["crossval_yaw_error_list"])
                    - np.array(metrics["std_crossval_yaw_error_list"]),
                    np.array(metrics["crossval_yaw_error_list"])
                    + np.array(metrics["std_crossval_yaw_error_list"]),
                    color=self.color_yaw,
                    alpha=0.3,
                )


    def draw_with_msg(self, msg):
        methods_in_msg = [method.method_name for method in msg.method_metrics]

        # Initialize missing methods dynamically
        if not self.metrics_data or set(self.metrics_data.keys()) != set(methods_in_msg):
            self.initialize_figure(methods_in_msg)
            self.initialize_metrics(methods_in_msg)

        for method in methods_in_msg:
            if self.is_delete_operation(method, [msg.num_of_converged_tracks]):
                self.remove_avg_error_from_list(method)

        self.update_metrics(msg)
        self.draw_subplots()
        self.plot_label_and_set_xy_lim()
        plt.tight_layout()
        plt.pause(0.1)




class MetricsPlotterNode(Node):
    def __init__(self):
        super().__init__("plot_metric")
        self.subscription = self.create_subscription(
            CalibrationMetrics, "calibration_metrics", self.listener_callback, 10
        )
        self.metrics_plotter = MetricsPlotter()

    def listener_callback(self, msg):
        self.metrics_plotter.draw_with_msg(msg)


def main(args=None):
    rclpy.init(args=args)
    metrics_plotter_node = MetricsPlotterNode()
    rclpy.spin(metrics_plotter_node)
    metrics_plotter_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
