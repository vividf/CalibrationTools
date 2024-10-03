// Copyright 2024 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <autoware/universe_utils/geometry/geometry.hpp>
#include <marker_radar_lidar_calibrator/marker_radar_lidar_calibrator.hpp>
#include <marker_radar_lidar_calibrator/track.hpp>
#include <marker_radar_lidar_calibrator/transformation_estimator.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tier4_ground_plane_utils/ground_plane_utils.hpp>

#include <pcl/ModelCoefficients.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/utils.h>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <tuple>

#define UPDATE_PARAM(PARAM_STRUCT, NAME) update_param(parameters, #NAME, PARAM_STRUCT.NAME)

namespace
{
template <typename T>
void update_param(
  const std::vector<rclcpp::Parameter> & parameters, const std::string & name, T & value)
{
  auto it = std::find_if(
    parameters.cbegin(), parameters.cend(),
    [&name](const rclcpp::Parameter & parameter) { return parameter.get_name() == name; });
  if (it != parameters.cend()) {
    value = it->template get_value<T>();
    RCLCPP_INFO_STREAM(
      rclcpp::get_logger("marker_radar_lidar_calibrator"),
      "Setting parameter [" << name << "] to " << value);
  }
}
}  // namespace

namespace marker_radar_lidar_calibrator
{
rcl_interfaces::msg::SetParametersResult ExtrinsicReflectorBasedCalibrator::paramCallback(
  const std::vector<rclcpp::Parameter> & parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  Parameters p = parameters_;

  try {
    UPDATE_PARAM(p, radar_optimization_frame);
    UPDATE_PARAM(p, use_lidar_initial_crop_box_filter);
    UPDATE_PARAM(p, lidar_initial_crop_box_min_x);
    UPDATE_PARAM(p, lidar_initial_crop_box_min_y);
    UPDATE_PARAM(p, lidar_initial_crop_box_min_z);
    UPDATE_PARAM(p, lidar_initial_crop_box_max_x);
    UPDATE_PARAM(p, lidar_initial_crop_box_max_y);
    UPDATE_PARAM(p, lidar_initial_crop_box_max_z);
    UPDATE_PARAM(p, use_radar_initial_crop_box_filter);
    UPDATE_PARAM(p, radar_initial_crop_box_min_x);
    UPDATE_PARAM(p, radar_initial_crop_box_min_y);
    UPDATE_PARAM(p, radar_initial_crop_box_min_z);
    UPDATE_PARAM(p, radar_initial_crop_box_max_x);
    UPDATE_PARAM(p, radar_initial_crop_box_max_y);
    UPDATE_PARAM(p, radar_initial_crop_box_max_z);
    UPDATE_PARAM(p, lidar_background_model_leaf_size);
    UPDATE_PARAM(p, radar_background_model_leaf_size);

    UPDATE_PARAM(p, max_calibration_range);
    UPDATE_PARAM(p, background_model_timeout);
    UPDATE_PARAM(p, min_foreground_distance);
    UPDATE_PARAM(p, background_extraction_timeout);
    UPDATE_PARAM(p, ransac_threshold);
    UPDATE_PARAM(p, ransac_max_iterations);
    UPDATE_PARAM(p, lidar_cluster_max_tolerance);
    UPDATE_PARAM(p, lidar_cluster_min_points);
    UPDATE_PARAM(p, lidar_cluster_max_points);
    UPDATE_PARAM(p, radar_cluster_max_tolerance);
    UPDATE_PARAM(p, radar_cluster_min_points);
    UPDATE_PARAM(p, radar_cluster_max_points);
    UPDATE_PARAM(p, reflector_radius);
    UPDATE_PARAM(p, reflector_max_height);
    UPDATE_PARAM(p, max_matching_distance);
    UPDATE_PARAM(p, max_initial_calibration_translation_error);
    UPDATE_PARAM(p, max_initial_calibration_rotation_error);
    UPDATE_PARAM(p, max_number_of_combination_samples);
    // transaction succeeds, now assign values
    parameters_ = p;
  } catch (const rclcpp::exceptions::InvalidParameterTypeException & e) {
    result.successful = false;
    result.reason = e.what();
  }

  return result;
}

ExtrinsicReflectorBasedCalibrator::ExtrinsicReflectorBasedCalibrator(
  const rclcpp::NodeOptions & options)
: Node("marker_radar_lidar_calibrator_node", options), tf_broadcaster_(this)
{
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  transform_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  parameters_.radar_optimization_frame =
    this->declare_parameter<std::string>("radar_optimization_frame");

  parameters_.use_lidar_initial_crop_box_filter =
    this->declare_parameter<bool>("use_lidar_initial_crop_box_filter", true);
  parameters_.lidar_initial_crop_box_min_x =
    this->declare_parameter<double>("lidar_initial_crop_box_min_x", -50.0);
  parameters_.lidar_initial_crop_box_min_y =
    this->declare_parameter<double>("lidar_initial_crop_box_min_y", -50.0);
  parameters_.lidar_initial_crop_box_min_z =
    this->declare_parameter<double>("lidar_initial_crop_box_min_z", -50.0);
  parameters_.lidar_initial_crop_box_max_x =
    this->declare_parameter<double>("lidar_initial_crop_box_max_x", 50.0);
  parameters_.lidar_initial_crop_box_max_y =
    this->declare_parameter<double>("lidar_initial_crop_box_max_y", 50.0);
  parameters_.lidar_initial_crop_box_max_z =
    this->declare_parameter<double>("lidar_initial_crop_box_max_z", 50.0);

  parameters_.use_radar_initial_crop_box_filter =
    this->declare_parameter<bool>("use_radar_initial_crop_box_filter", true);
  parameters_.radar_initial_crop_box_min_x =
    this->declare_parameter<double>("radar_initial_crop_box_min_x", -50.0);
  parameters_.radar_initial_crop_box_min_y =
    this->declare_parameter<double>("radar_initial_crop_box_min_y", -50.0);
  parameters_.radar_initial_crop_box_min_z =
    this->declare_parameter<double>("radar_initial_crop_box_min_z", -50.0);
  parameters_.radar_initial_crop_box_max_x =
    this->declare_parameter<double>("radar_initial_crop_box_max_x", 50.0);
  parameters_.radar_initial_crop_box_max_y =
    this->declare_parameter<double>("radar_initial_crop_box_max_y", 50.0);
  parameters_.radar_initial_crop_box_max_z =
    this->declare_parameter<double>("radar_initial_crop_box_max_z", 50.0);

  double calibration_max_range = this->declare_parameter<double>("calibration_max_range", 60.0);
  lidar_background_model_.min_point_ =
    Eigen::Vector4f(-calibration_max_range, -calibration_max_range, -calibration_max_range, 1.f);
  lidar_background_model_.max_point_ =
    Eigen::Vector4f(calibration_max_range, calibration_max_range, calibration_max_range, 1.f);
  radar_background_model_.min_point_ =
    Eigen::Vector4f(-calibration_max_range, -calibration_max_range, -calibration_max_range, 1.f);
  radar_background_model_.max_point_ =
    Eigen::Vector4f(calibration_max_range, calibration_max_range, calibration_max_range, 1.f);

  lidar_background_model_.leaf_size_ =
    this->declare_parameter<double>("lidar_background_model_leaf_size", 0.1);
  radar_background_model_.leaf_size_ =
    this->declare_parameter<double>("radar_background_model_leaf_size", 0.1);
  parameters_.max_calibration_range =
    this->declare_parameter<double>("max_calibration_range", 20.0);
  parameters_.background_model_timeout =
    this->declare_parameter<double>("background_model_timeout", 1.0);
  parameters_.min_foreground_distance =
    this->declare_parameter<double>("min_foreground_distance", 0.4);
  parameters_.background_extraction_timeout =
    this->declare_parameter<double>("background_extraction_timeout", 3.0);
  parameters_.ransac_threshold = this->declare_parameter<double>("ransac_threshold", 0.2);
  parameters_.ransac_max_iterations = this->declare_parameter<int>("ransac_max_iterations", 100);
  parameters_.lidar_cluster_max_tolerance =
    this->declare_parameter<double>("lidar_cluster_max_tolerance", 0.5);
  parameters_.lidar_cluster_min_points =
    this->declare_parameter<int>("lidar_cluster_min_points", 3);
  parameters_.lidar_cluster_max_points =
    this->declare_parameter<int>("lidar_cluster_max_points", 500);
  parameters_.radar_cluster_max_tolerance =
    this->declare_parameter<double>("radar_cluster_max_tolerance", 0.5);
  parameters_.radar_cluster_min_points =
    this->declare_parameter<int>("radar_cluster_min_points", 1);
  parameters_.radar_cluster_max_points =
    this->declare_parameter<int>("radar_cluster_max_points", 10);
  parameters_.reflector_radius = this->declare_parameter<double>("reflector_radius", 0.1);
  parameters_.reflector_max_height = this->declare_parameter<double>("reflector_max_height", 1.2);
  parameters_.max_matching_distance = this->declare_parameter<double>("max_matching_distance", 1.0);
  parameters_.max_number_of_combination_samples = static_cast<std::size_t>(
    this->declare_parameter<int>("max_number_of_combination_samples", 10000));

  auto msg_type = this->declare_parameter<std::string>("msg_type");
  auto transformation_type = this->declare_parameter<std::string>("transformation_type");

  if (msg_type == "radar_tracks") {
    msg_type_ = MsgType::radar_tracks;
  } else if (msg_type == "radar_scan") {
    msg_type_ = MsgType::radar_scan;
  } else if (msg_type == "radar_cloud") {
    msg_type_ = MsgType::radar_cloud;
  } else {
    throw std::runtime_error("Invalid param value: " + msg_type);
  }

  if (transformation_type == "svd_2d") {
    transformation_type_ = TransformationType::svd_2d;
  } else if (transformation_type == "yaw_only_rotation_2d") {
    transformation_type_ = TransformationType::yaw_only_rotation_2d;
  } else if (transformation_type == "svd_3d") {
    transformation_type_ = TransformationType::svd_3d;
  } else if (transformation_type == "zero_roll_3d") {
    transformation_type_ = TransformationType::zero_roll_3d;
  } else {
    throw std::runtime_error("Invalid param value: " + transformation_type);
  }

  double initial_lidar_cov = this->declare_parameter<double>("initial_lidar_cov", 0.5);
  double initial_radar_cov = this->declare_parameter<double>("initial_radar_cov", 2.0);
  double lidar_measurement_cov = this->declare_parameter<double>("lidar_measurement_cov", 0.03);
  double radar_measurement_cov = this->declare_parameter<double>("radar_measurement_cov", 0.05);
  double lidar_process_cov = this->declare_parameter<double>("lidar_process_cov", 0.01);
  double radar_process_cov = this->declare_parameter<double>("radar_process_cov", 0.01);
  double lidar_convergence_thresh =
    this->declare_parameter<double>("lidar_convergence_thresh", 0.03);
  double radar_convergence_thresh =
    this->declare_parameter<double>("radar_convergence_thresh", 0.03);
  double timeout_thresh = this->declare_parameter<double>("timeout_thresh", 3.0);

  parameters_.max_initial_calibration_translation_error =
    this->declare_parameter<double>("max_initial_calibration_translation_error", 1.0);
  parameters_.max_initial_calibration_rotation_error =
    this->declare_parameter<double>("max_initial_calibration_rotation_error", 45.0);

  factory_ptr_ = std::make_shared<TrackFactory>(
    initial_lidar_cov, initial_radar_cov, lidar_measurement_cov, radar_measurement_cov,
    lidar_process_cov, radar_process_cov, lidar_convergence_thresh, radar_convergence_thresh,
    timeout_thresh, parameters_.max_matching_distance);

  lidar_background_pub_ =
    this->create_publisher<sensor_msgs::msg::PointCloud2>("lidar_background_pointcloud", 10);
  lidar_foreground_pub_ =
    this->create_publisher<sensor_msgs::msg::PointCloud2>("lidar_foreground_pointcloud", 10);
  lidar_colored_clusters_pub_ =
    this->create_publisher<sensor_msgs::msg::PointCloud2>("lidar_colored_clusters", 10);
  lidar_detections_pub_ =
    this->create_publisher<visualization_msgs::msg::MarkerArray>("lidar_detection_markers", 10);
  lidar_plane_pub_ =
    this->create_publisher<sensor_msgs::msg::PointCloud2>("lidar_plane_pointcloud", 10);

  radar_background_pub_ =
    this->create_publisher<sensor_msgs::msg::PointCloud2>("radar_background_pointcloud", 10);
  radar_foreground_pub_ =
    this->create_publisher<sensor_msgs::msg::PointCloud2>("radar_foreground_pointcloud", 10);
  radar_detections_pub_ =
    this->create_publisher<visualization_msgs::msg::MarkerArray>("radar_detection_markers", 10);
  matches_markers_pub_ =
    this->create_publisher<visualization_msgs::msg::MarkerArray>("matches_markers", 10);
  tracking_markers_pub_ =
    this->create_publisher<visualization_msgs::msg::MarkerArray>("tracking_markers", 10);
  text_markers_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("text_markers", 10);
  metrics_pub_ =
    this->create_publisher<std_msgs::msg::Float32MultiArray>("calibration_metrics", 10);

  lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "input_lidar_pointcloud", rclcpp::SensorDataQoS(),
    std::bind(&ExtrinsicReflectorBasedCalibrator::lidarCallback, this, std::placeholders::_1));

  if (msg_type_ == MsgType::radar_tracks) {
    radar_tracks_sub_ = this->create_subscription<radar_msgs::msg::RadarTracks>(
      "input_radar_msg", rclcpp::SensorDataQoS(),
      std::bind(
        &ExtrinsicReflectorBasedCalibrator::radarTracksCallback, this, std::placeholders::_1));
  } else if (msg_type_ == MsgType::radar_scan) {
    radar_scan_sub_ = this->create_subscription<radar_msgs::msg::RadarScan>(
      "input_radar_msg", rclcpp::SensorDataQoS(),
      std::bind(
        &ExtrinsicReflectorBasedCalibrator::radarScanCallback, this, std::placeholders::_1));
  } else if (msg_type_ == MsgType::radar_cloud) {
    radar_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "input_radar_msg", rclcpp::SensorDataQoS(),
      std::bind(
        &ExtrinsicReflectorBasedCalibrator::radarCloudCallback, this, std::placeholders::_1));
  }

  // The service server runs in a dedicated thread
  calibration_api_srv_callback_group_ =
    create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  calibration_ui_srv_callback_group_ =
    create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  set_param_res_ = add_on_set_parameters_callback(
    std::bind(&ExtrinsicReflectorBasedCalibrator::paramCallback, this, std::placeholders::_1));

  calibration_request_server_ =
    this->create_service<tier4_calibration_msgs::srv::ExtrinsicCalibrator>(
      "extrinsic_calibration",
      std::bind(
        &ExtrinsicReflectorBasedCalibrator::requestReceivedCallback, this, std::placeholders::_1,
        std::placeholders::_2),
      rmw_qos_profile_services_default, calibration_api_srv_callback_group_);

  background_model_service_server_ = this->create_service<std_srvs::srv::Empty>(
    "extract_background_model",
    std::bind(
      &ExtrinsicReflectorBasedCalibrator::backgroundModelRequestCallback, this,
      std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default, calibration_ui_srv_callback_group_);

  timer_ = rclcpp::create_timer(
    this, get_clock(), std::chrono::seconds(1),
    std::bind(&ExtrinsicReflectorBasedCalibrator::timerCallback, this));
}

void ExtrinsicReflectorBasedCalibrator::requestReceivedCallback(
  [[maybe_unused]] const std::shared_ptr<tier4_calibration_msgs::srv::ExtrinsicCalibrator::Request>
    request,
  const std::shared_ptr<tier4_calibration_msgs::srv::ExtrinsicCalibrator::Response> response)
{
  using std::chrono_literals::operator""s;

  // Loop until the calibration finishes
  while (rclcpp::ok()) {
    rclcpp::sleep_for(1s);
    std::unique_lock<std::mutex> lock(mutex_);

    if (send_calibration_) {
      break;
    }
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 60000, "Waiting for the calibration to end");
  }

  std::unique_lock<std::mutex> lock(mutex_);

  std::stringstream ss;
  ss << "Calibration successful. distance_score=" << calibration_distance_score_
     << " yaw_score=" << calibration_yaw_score_;

  tier4_calibration_msgs::msg::CalibrationResult result;
  result.message.data = ss.str();
  result.score = calibration_distance_score_;
  result.success = true;
  result.transform_stamped = tf2::eigenToTransform(calibrated_radar_to_lidar_eigen_);
  result.transform_stamped.header.frame_id = radar_frame_;
  result.transform_stamped.child_frame_id = lidar_frame_;

  response->results.emplace_back(result);
}

void ExtrinsicReflectorBasedCalibrator::timerCallback()
{
  if (
    lidar_background_model_.valid_ && radar_background_model_.valid_ && !tracking_service_server_) {
    tracking_service_server_ = this->create_service<std_srvs::srv::Empty>(
      "add_lidar_radar_pair",
      std::bind(
        &ExtrinsicReflectorBasedCalibrator::trackingRequestCallback, this, std::placeholders::_1,
        std::placeholders::_2),
      rmw_qos_profile_services_default, calibration_ui_srv_callback_group_);
  }

  if (calibration_valid_ && !send_calibration_service_server_) {
    send_calibration_service_server_ = this->create_service<std_srvs::srv::Empty>(
      "send_calibration",
      std::bind(
        &ExtrinsicReflectorBasedCalibrator::sendCalibrationCallback, this, std::placeholders::_1,
        std::placeholders::_2),
      rmw_qos_profile_services_default, calibration_ui_srv_callback_group_);
  }

  if (converged_tracks_.size() > 0 && !delete_track_service_server_) {
    delete_track_service_server_ = this->create_service<std_srvs::srv::Empty>(
      "delete_lidar_radar_pair",
      std::bind(
        &ExtrinsicReflectorBasedCalibrator::deleteTrackRequestCallback, this, std::placeholders::_1,
        std::placeholders::_2),
      rmw_qos_profile_services_default, calibration_ui_srv_callback_group_);
  }
}

void ExtrinsicReflectorBasedCalibrator::backgroundModelRequestCallback(
  [[maybe_unused]] const std::shared_ptr<std_srvs::srv::Empty::Request> request,
  [[maybe_unused]] const std::shared_ptr<std_srvs::srv::Empty::Response> response)
{
  using std::chrono_literals::operator""s;

  {
    std::unique_lock<std::mutex> lock(mutex_);
    extract_lidar_background_model_ = true;
    extract_radar_background_model_ = true;
  }

  while (rclcpp::ok()) {
    rclcpp::sleep_for(1s);
    std::unique_lock<std::mutex> lock(mutex_);

    if (lidar_background_model_.valid_ && radar_background_model_.valid_) {
      break;
    }

    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000, "Waiting to extract the background model");
  }

  RCLCPP_INFO(this->get_logger(), "Background model estimated");
}

void ExtrinsicReflectorBasedCalibrator::trackingRequestCallback(
  [[maybe_unused]] const std::shared_ptr<std_srvs::srv::Empty::Request> request,
  [[maybe_unused]] const std::shared_ptr<std_srvs::srv::Empty::Response> response)
{
  using std::chrono_literals::operator""s;

  {
    std::unique_lock<std::mutex> lock(mutex_);
    tracking_active_ = true;
    current_new_tracks_ = 0;
  }

  while (rclcpp::ok()) {
    rclcpp::sleep_for(1s);
    std::unique_lock<std::mutex> lock(mutex_);

    if (!tracking_active_ && current_new_tracks_) {
      break;
    }

    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "Waiting for detections to converge. Active tracks=%lu. New tracks=%d", active_tracks_.size(),
      current_new_tracks_);
  }

  RCLCPP_INFO(this->get_logger(), "New converged detections: %d", current_new_tracks_);
}

void ExtrinsicReflectorBasedCalibrator::deleteTrackRequestCallback(
  [[maybe_unused]] const std::shared_ptr<std_srvs::srv::Empty::Request> request,
  [[maybe_unused]] const std::shared_ptr<std_srvs::srv::Empty::Response> response)
{
  using std::chrono_literals::operator""s;

  if (converged_tracks_.size() > 0) {
    converged_tracks_.pop_back();
    calibrateSensors();
    visualizeTrackMarkers();
    deleteTrackMarkers();
    drawCalibrationStatusText();
    RCLCPP_INFO(
      this->get_logger(), "The last track was successfully deleted. Remaining converged tracks: %d",
      static_cast<int>(converged_tracks_.size()));
    // sleep for 1s for the plotter node to finish plotting.
    rclcpp::sleep_for(1s);
  } else {
    RCLCPP_WARN(this->get_logger(), "There are no converged tracks available");
  }
}

void ExtrinsicReflectorBasedCalibrator::sendCalibrationCallback(
  [[maybe_unused]] const std::shared_ptr<std_srvs::srv::Empty::Request> request,
  [[maybe_unused]] const std::shared_ptr<std_srvs::srv::Empty::Response> response)
{
  std::unique_lock<std::mutex> lock(mutex_);
  send_calibration_ = true;
}

void ExtrinsicReflectorBasedCalibrator::lidarCallback(
  const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  RCLCPP_INFO(this->get_logger(), "lidarCallback");
  std::vector<Eigen::Vector3d> radar_detections;
  if (msg_type_ == MsgType::radar_tracks) {
    if (!latest_radar_tracks_msgs_ || latest_radar_tracks_msgs_->tracks.size() == 0) {
      RCLCPP_INFO(this->get_logger(), "There were no radar tracks");
      return;
    }
    pcl::PointCloud<common_types::PointType>::Ptr radar_pointcloud_ptr =
      extractRadarPointcloud(latest_radar_tracks_msgs_);
    radar_detections = extractRadarReflectors(radar_pointcloud_ptr);
    latest_radar_tracks_msgs_->tracks.clear();
  } else if (msg_type_ == MsgType::radar_scan) {
    if (!latest_radar_scan_msgs_ || latest_radar_scan_msgs_->returns.size() == 0) {
      if (latest_radar_scan_msgs_->returns.size() == 0)
        RCLCPP_INFO(this->get_logger(), "There were no radar scans");
      return;
    }
    pcl::PointCloud<common_types::PointType>::Ptr radar_pointcloud_ptr =
      extractRadarPointcloud(latest_radar_scan_msgs_);
    radar_detections = extractRadarReflectors(radar_pointcloud_ptr);
    latest_radar_scan_msgs_->returns.clear();
  } else {
    if (!latest_radar_cloud_msgs_) {
      RCLCPP_INFO(this->get_logger(), "There were no radar pointclouds");
      return;
    }
    pcl::PointCloud<common_types::PointType>::Ptr radar_pointcloud_ptr =
      extractRadarPointcloud(latest_radar_cloud_msgs_);
    radar_detections = extractRadarReflectors(radar_pointcloud_ptr);
  }

  auto lidar_detections = extractLidarReflectors(msg);
  auto matches = matchDetections(lidar_detections, radar_detections);

  bool is_track_converged = trackMatches(matches, msg->header.stamp);
  if (is_track_converged) calibrateSensors();
  visualizationMarkers(lidar_detections, radar_detections, matches);
  visualizeTrackMarkers();
  drawCalibrationStatusText();

  RCLCPP_INFO(
    this->get_logger(),
    "Lidar detections: %lu Radar detections: %lu Matches: %lu Active tracks; %lu Converged tracks: "
    "%lu",
    lidar_detections.size(), radar_detections.size(), matches.size(), active_tracks_.size(),
    converged_tracks_.size());
}

void ExtrinsicReflectorBasedCalibrator::radarTracksCallback(
  const radar_msgs::msg::RadarTracks::SharedPtr msg)
{
  if (!latest_radar_tracks_msgs_) {
    latest_radar_tracks_msgs_ = msg;
  } else {
    latest_radar_tracks_msgs_->header = msg->header;
    latest_radar_tracks_msgs_->tracks.insert(
      latest_radar_tracks_msgs_->tracks.end(), msg->tracks.begin(), msg->tracks.end());
  }
}

void ExtrinsicReflectorBasedCalibrator::radarScanCallback(
  const radar_msgs::msg::RadarScan::SharedPtr msg)
{
  if (!latest_radar_scan_msgs_) {
    latest_radar_scan_msgs_ = msg;
  } else {
    latest_radar_scan_msgs_->header = msg->header;
    latest_radar_scan_msgs_->returns.insert(
      latest_radar_scan_msgs_->returns.end(), msg->returns.begin(), msg->returns.end());
  }
}

void ExtrinsicReflectorBasedCalibrator::radarCloudCallback(
  const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  latest_radar_cloud_msgs_ = msg;
}

std::vector<Eigen::Vector3d> ExtrinsicReflectorBasedCalibrator::extractLidarReflectors(
  const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  lidar_frame_ = msg->header.frame_id;
  lidar_header_ = msg->header;
  bool extract_background_model;
  bool valid_background_model;
  std::vector<Eigen::Vector3d> detections;

  {
    std::unique_lock<std::mutex> lock(mutex_);
    extract_background_model = extract_lidar_background_model_;
    valid_background_model = lidar_background_model_.valid_;
  }

  pcl::PointCloud<common_types::PointType>::Ptr lidar_pointcloud_ptr(
    new pcl::PointCloud<common_types::PointType>);
  pcl::fromROSMsg(*msg, *lidar_pointcloud_ptr);

  if (parameters_.use_lidar_initial_crop_box_filter) {
    pcl::CropBox<common_types::PointType> box_filter;
    pcl::PointCloud<common_types::PointType>::Ptr tmp_lidar_pointcloud_ptr(
      new pcl::PointCloud<common_types::PointType>);
    RCLCPP_INFO(this->get_logger(), "pre lidar_pointcloud_ptr=%lu", lidar_pointcloud_ptr->size());
    RCLCPP_WARN(
      this->get_logger(), "crop box parameters=%f | %f | %f",
      parameters_.lidar_initial_crop_box_min_x, parameters_.lidar_initial_crop_box_min_y,
      parameters_.lidar_initial_crop_box_min_z);
    RCLCPP_WARN(
      this->get_logger(), "crop box parameters=%f | %f | %f",
      parameters_.lidar_initial_crop_box_max_x, parameters_.lidar_initial_crop_box_max_y,
      parameters_.lidar_initial_crop_box_max_z);
    box_filter.setMin(Eigen::Vector4f(
      parameters_.lidar_initial_crop_box_min_x, parameters_.lidar_initial_crop_box_min_y,
      parameters_.lidar_initial_crop_box_min_z, 1.0));
    box_filter.setMax(Eigen::Vector4f(
      parameters_.lidar_initial_crop_box_max_x, parameters_.lidar_initial_crop_box_max_y,
      parameters_.lidar_initial_crop_box_max_z, 1.0));
    box_filter.setInputCloud(lidar_pointcloud_ptr);
    box_filter.filter(*tmp_lidar_pointcloud_ptr);
    lidar_pointcloud_ptr.swap(tmp_lidar_pointcloud_ptr);
    RCLCPP_INFO(this->get_logger(), "lidar_pointcloud_ptr=%lu", lidar_pointcloud_ptr->size());
  }

  if (extract_background_model && !valid_background_model) {
    extractBackgroundModel(
      lidar_pointcloud_ptr, msg->header, latest_updated_lidar_header_, first_lidar_header_,
      lidar_background_model_);
    return detections;
  }

  if (!valid_background_model) {
    return detections;
  }

  pcl::PointCloud<common_types::PointType>::Ptr ground_plane_inliers_ptr;
  bool status;

  if (first_time_) {
    Eigen::Isometry3d initial_base_to_lidar_transform;

    try {
      rclcpp::Time t = rclcpp::Time(0);
      rclcpp::Duration timeout = rclcpp::Duration::from_seconds(1.0);

      initial_base_to_lidar_transform = tf2::transformToEigen(
        tf_buffer_->lookupTransform("base_link", "hesai_top", t, timeout).transform);
    } catch (tf2::TransformException & ex) {
      RCLCPP_WARN(this->get_logger(), "could not get initial tf. %s", ex.what());
    }

    tier4_ground_plane_utils::GroundPlaneExtractorParameters parameters;
    parameters.verbose_ = true;
    parameters.use_crop_box_filter_ = false;
    parameters.use_pca_rough_normal_ = false;
    parameters.max_inlier_distance_ = 0.01;
    parameters.min_plane_points_ = 300;
    parameters.min_plane_points_percentage_ = 1.0;
    parameters.max_cos_distance_ = 0.5;
    parameters.max_iterations_ = 1000;
    parameters.remove_outliers_ = false;
    parameters.initial_base_to_lidar_transform_ = initial_base_to_lidar_transform;
    std::vector<Eigen::Vector4d> outlier_models;

    std::tie(status, ground_model_, ground_plane_inliers_ptr) =
      tier4_ground_plane_utils::extractGroundPlane(
        lidar_pointcloud_ptr, parameters, outlier_models);

    first_time_ = false;
    std::cout << "####### ground_model_: ######" << ground_model_.matrix() << std::endl;
  }

  pcl::PointCloud<common_types::PointType>::Ptr foreground_pointcloud_ptr;
  extractForegroundPoints(
    lidar_pointcloud_ptr, lidar_background_model_, false, foreground_pointcloud_ptr, ground_model_);

  auto clusters = extractClusters(
    foreground_pointcloud_ptr, parameters_.lidar_cluster_max_tolerance,
    parameters_.lidar_cluster_min_points, parameters_.lidar_cluster_max_points);
  detections = findReflectorsFromClusters(clusters, ground_model_);

  // Visualization
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_clusters_pointcloud_ptr(
    new pcl::PointCloud<pcl::PointXYZRGB>());

  Eigen::Vector3i colors[7] = {{0, 0, 255},   {0, 128, 255}, {0, 200, 200}, {0, 255, 0},
                               {200, 200, 0}, {255, 0, 0},   {255, 0, 255}};

  std::size_t colored_clusters_pointcloud_size = 0;

  for (const auto & cluster : clusters) {
    colored_clusters_pointcloud_size += cluster->size();
  }

  colored_clusters_pointcloud_ptr->reserve(colored_clusters_pointcloud_size);

  for (std::size_t i = 0; i < clusters.size(); i++) {
    const auto & cluster = clusters[i];
    pcl::PointXYZRGB colored_p;  // cSpell:ignore XYZRGB
    auto & color = colors[i % 7];

    for (const auto & p : cluster->points) {
      colored_p.getArray3fMap() = p.getArray3fMap();
      colored_p.r = color.x();
      colored_p.b = color.y();
      colored_p.g = color.z();
      colored_clusters_pointcloud_ptr->push_back(colored_p);
    }
  }

  colored_clusters_pointcloud_ptr->width = colored_clusters_pointcloud_ptr->size();
  colored_clusters_pointcloud_ptr->height = 1;

  RCLCPP_INFO(
    this->get_logger(), "Colored clusters size=%lu", colored_clusters_pointcloud_ptr->size());

  sensor_msgs::msg::PointCloud2 background_msg;
  pcl::toROSMsg(*lidar_background_model_.pointcloud_, background_msg);
  background_msg.header = lidar_header_;
  lidar_background_pub_->publish(background_msg);

  sensor_msgs::msg::PointCloud2 foreground_msg;
  pcl::toROSMsg(*foreground_pointcloud_ptr, foreground_msg);
  foreground_msg.header = lidar_header_;
  lidar_foreground_pub_->publish(foreground_msg);

  sensor_msgs::msg::PointCloud2 colored_clusters_msg;
  pcl::toROSMsg(*colored_clusters_pointcloud_ptr, colored_clusters_msg);
  colored_clusters_msg.header = lidar_header_;
  lidar_colored_clusters_pub_->publish(colored_clusters_msg);

  if (ground_plane_inliers_ptr) {
    sensor_msgs::msg::PointCloud2 plane_msg;
    pcl::toROSMsg(*ground_plane_inliers_ptr, plane_msg);
    plane_msg.header = lidar_header_;
    lidar_plane_pub_->publish(plane_msg);
  }

  return detections;
}

template <typename RadarMsgType>
pcl::PointCloud<common_types::PointType>::Ptr
ExtrinsicReflectorBasedCalibrator::extractRadarPointcloud(const std::shared_ptr<RadarMsgType> & msg)
{
  static_assert(
    std::is_same<RadarMsgType, radar_msgs::msg::RadarTracks>::value ||
      std::is_same<RadarMsgType, radar_msgs::msg::RadarScan>::value ||
      std::is_same<RadarMsgType, sensor_msgs::msg::PointCloud2>::value,
    "Unsupported message type");

  radar_frame_ = msg->header.frame_id;
  radar_header_ = msg->header;
  auto radar_pointcloud_ptr = std::make_shared<pcl::PointCloud<common_types::PointType>>();

  if constexpr (std::is_same<RadarMsgType, radar_msgs::msg::RadarTracks>::value) {
    radar_pointcloud_ptr->reserve(msg->tracks.size());
    for (const auto & track : msg->tracks) {
      radar_pointcloud_ptr->emplace_back(track.position.x, track.position.y, track.position.z);
    }
  } else if constexpr (std::is_same<RadarMsgType, radar_msgs::msg::RadarScan>::value) {
    radar_pointcloud_ptr->reserve(msg->returns.size());
    for (const auto & radar_return : msg->returns) {
      float range = radar_return.range;
      float azimuth = radar_return.azimuth;
      float elevation = radar_return.elevation;

      float x = range * std::cos(azimuth) * std::cos(elevation);
      float y = range * std::sin(azimuth) * std::cos(elevation);
      float z = range * std::sin(elevation);

      radar_pointcloud_ptr->emplace_back(x, y, z);
    }
  } else if constexpr (std::is_same<RadarMsgType, sensor_msgs::msg::PointCloud2>::value) {
    pcl::fromROSMsg(*msg, *radar_pointcloud_ptr);
  }

  return radar_pointcloud_ptr;
}

std::vector<Eigen::Vector3d> ExtrinsicReflectorBasedCalibrator::extractRadarReflectors(
  pcl::PointCloud<common_types::PointType>::Ptr radar_pointcloud_ptr)
{
  bool extract_background_model;
  bool valid_background_model;
  std::vector<Eigen::Vector3d> detections;

  {
    std::unique_lock<std::mutex> lock(mutex_);
    extract_background_model = extract_radar_background_model_;
    valid_background_model = radar_background_model_.valid_;
  }

  if (parameters_.use_radar_initial_crop_box_filter) {
    pcl::CropBox<common_types::PointType> box_filter;
    pcl::PointCloud<common_types::PointType>::Ptr tmp_radar_pointcloud_ptr(
      new pcl::PointCloud<common_types::PointType>);
    box_filter.setMin(Eigen::Vector4f(
      parameters_.radar_initial_crop_box_min_x, parameters_.radar_initial_crop_box_min_y,
      parameters_.radar_initial_crop_box_min_z, 1.0));
    box_filter.setMax(Eigen::Vector4f(
      parameters_.radar_initial_crop_box_max_x, parameters_.radar_initial_crop_box_max_y,
      parameters_.radar_initial_crop_box_max_z, 1.0));
    box_filter.setInputCloud(radar_pointcloud_ptr);
    box_filter.filter(*tmp_radar_pointcloud_ptr);
    radar_pointcloud_ptr.swap(tmp_radar_pointcloud_ptr);
  }

  if (extract_background_model && !valid_background_model) {
    extractBackgroundModel(
      radar_pointcloud_ptr, radar_header_, latest_updated_radar_header_, first_radar_header_,
      radar_background_model_);
    return detections;
  }

  if (!valid_background_model) {
    return detections;
  }

  pcl::PointCloud<common_types::PointType>::Ptr foreground_pointcloud_ptr;
  Eigen::Vector4d ground_model;

  extractForegroundPoints(
    radar_pointcloud_ptr, radar_background_model_, false, foreground_pointcloud_ptr, ground_model);
  auto clusters = extractClusters(
    foreground_pointcloud_ptr, parameters_.radar_cluster_max_tolerance,
    parameters_.radar_cluster_min_points, parameters_.radar_cluster_max_points);

  detections.reserve(clusters.size());

  RCLCPP_INFO(this->get_logger(), "Extracting radar reflectors from clusters");

  for (const auto & cluster : clusters) {
    Eigen::Vector3d p_avg = Eigen::Vector3d::Zero();

    for (const auto & p : cluster->points) {
      p_avg += Eigen::Vector3d(p.x, p.y, p.z);
    }

    p_avg /= cluster->points.size();
    RCLCPP_INFO(
      this->get_logger(), "\t Radar reflector id=%lu size=%lu center: x=%.2f y=%.2f z=%.2f",
      detections.size(), cluster->points.size(), p_avg.x(), p_avg.y(), p_avg.z());

    detections.emplace_back(p_avg);
  }

  sensor_msgs::msg::PointCloud2 background_msg;
  pcl::toROSMsg(*radar_background_model_.pointcloud_, background_msg);
  background_msg.header = radar_header_;
  radar_background_pub_->publish(background_msg);

  sensor_msgs::msg::PointCloud2 foreground_msg;
  pcl::toROSMsg(*radar_pointcloud_ptr, foreground_msg);
  foreground_msg.header = radar_header_;
  radar_foreground_pub_->publish(foreground_msg);

  return detections;
}

void ExtrinsicReflectorBasedCalibrator::extractBackgroundModel(
  const pcl::PointCloud<common_types::PointType>::Ptr & sensor_pointcloud_ptr,
  const std_msgs::msg::Header & current_header, std_msgs::msg::Header & last_updated_header,
  std_msgs::msg::Header & first_header, BackgroundModel & background_model)
{
  // Initialize background model in the first iteration
  if (background_model.set_.size() == 0) {
    background_model.min_point_ = Eigen::Vector4f(
      -parameters_.max_calibration_range, -parameters_.max_calibration_range,
      -parameters_.max_calibration_range, 1.0);
    background_model.max_point_ = Eigen::Vector4f(
      parameters_.max_calibration_range, parameters_.max_calibration_range,
      parameters_.max_calibration_range, 1.0);
    last_updated_header = current_header;
    first_header = current_header;

    RCLCPP_INFO(this->get_logger(), "Background model estimation: first iteration");
    RCLCPP_INFO(
      this->get_logger(), "\t min_point: x=%.2f y=%.2f z=%.2f w=%.2f",
      background_model.min_point_.x(), background_model.min_point_.y(),
      background_model.min_point_.z(), background_model.min_point_.w());
    RCLCPP_INFO(
      this->get_logger(), "\t max_point: x=%.2f y=%.2f z=%.2f w=%.2f",
      background_model.max_point_.x(), background_model.max_point_.y(),
      background_model.max_point_.z(), background_model.max_point_.w());
  }

  index_t x_cells = (background_model.max_point_.x() - background_model.min_point_.x()) /
                    background_model.leaf_size_;
  index_t y_cells = (background_model.max_point_.y() - background_model.min_point_.y()) /
                    background_model.leaf_size_;
  index_t z_cells = (background_model.max_point_.z() - background_model.min_point_.z()) /
                    background_model.leaf_size_;
  background_model.pointcloud_->points.reserve(x_cells * y_cells * z_cells);
  index_t prev_num_points = background_model.set_.size();

  for (const auto & p : sensor_pointcloud_ptr->points) {
    index_t x_index =
      static_cast<index_t>((p.x - background_model.min_point_.x()) / background_model.leaf_size_);
    index_t y_index =
      static_cast<index_t>((p.y - background_model.min_point_.y()) / background_model.leaf_size_);
    index_t z_index =
      static_cast<index_t>((p.z - background_model.min_point_.z()) / background_model.leaf_size_);
    index_t index = z_index * y_cells * x_cells + y_index * x_cells + x_index;
    const auto & it = background_model.set_.emplace(index);

    if (it.second) {
      common_types::PointType p_center;
      p_center.x = background_model.min_point_.x() + background_model.leaf_size_ * (x_index + 0.5f);
      p_center.y = background_model.min_point_.y() + background_model.leaf_size_ * (y_index + 0.5f);
      p_center.z = background_model.min_point_.z() + background_model.leaf_size_ * (z_index + 0.5f);
      background_model.pointcloud_->push_back(p_center);
    }
  }

  double time_since_last_start =
    (rclcpp::Time(current_header.stamp) - rclcpp::Time(first_header.stamp)).seconds();

  if (
    background_model.set_.size() > prev_num_points &&
    time_since_last_start < parameters_.background_extraction_timeout) {
    RCLCPP_INFO(
      this->get_logger(), "Current points in the background model: %lu",
      background_model.set_.size());
    last_updated_header = current_header;
    return;
  }

  double time_since_last_update =
    (rclcpp::Time(current_header.stamp) - rclcpp::Time(last_updated_header.stamp)).seconds();
  if (
    time_since_last_update < parameters_.background_model_timeout &&
    time_since_last_update >= 0.0 &&
    time_since_last_start < parameters_.background_extraction_timeout) {
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000, "Waiting for timeout (%.2f)",
      time_since_last_update);
    return;
  }

  background_model.tree_.setInputCloud(background_model.pointcloud_);
  background_model.pointcloud_->points.shrink_to_fit();

  RCLCPP_INFO(this->get_logger(), "Finished background model initialization");

  {
    std::unique_lock<std::mutex> lock(mutex_);
    background_model.valid_ = true;
  }
}

void ExtrinsicReflectorBasedCalibrator::extractForegroundPoints(
  const pcl::PointCloud<common_types::PointType>::Ptr & sensor_pointcloud_ptr,
  const BackgroundModel & background_model, bool use_ransac,
  pcl::PointCloud<common_types::PointType>::Ptr & foreground_pointcloud_ptr,
  Eigen::Vector4d & ground_model)
{
  RCLCPP_INFO(this->get_logger(), "Extracting foreground");
  RCLCPP_INFO(this->get_logger(), "\t initial points: %lu", sensor_pointcloud_ptr->size());

  // Crop box
  pcl::PointCloud<common_types::PointType>::Ptr cropped_pointcloud_ptr(
    new pcl::PointCloud<common_types::PointType>);
  pcl::CropBox<common_types::PointType> crop_filter;
  crop_filter.setMin(background_model.min_point_);
  crop_filter.setMax(background_model.max_point_);
  crop_filter.setInputCloud(sensor_pointcloud_ptr);
  crop_filter.filter(*cropped_pointcloud_ptr);
  RCLCPP_INFO(this->get_logger(), "\t cropped points: %lu", cropped_pointcloud_ptr->size());

  // Fast hash
  pcl::PointCloud<common_types::PointType>::Ptr voxel_filtered_pointcloud_ptr(
    new pcl::PointCloud<common_types::PointType>);
  voxel_filtered_pointcloud_ptr->reserve(cropped_pointcloud_ptr->size());

  index_t x_cells = (background_model.max_point_.x() - background_model.min_point_.x()) /
                    background_model.leaf_size_;
  index_t y_cells = (background_model.max_point_.y() - background_model.min_point_.y()) /
                    background_model.leaf_size_;

  for (const auto & p : cropped_pointcloud_ptr->points) {
    index_t x_index =
      static_cast<index_t>((p.x - background_model.min_point_.x()) / background_model.leaf_size_);
    index_t y_index =
      static_cast<index_t>((p.y - background_model.min_point_.y()) / background_model.leaf_size_);
    index_t z_index =
      static_cast<index_t>((p.z - background_model.min_point_.z()) / background_model.leaf_size_);
    index_t index = z_index * y_cells * x_cells + y_index * x_cells + x_index;

    if (background_model.set_.count(index) == 0) {
      voxel_filtered_pointcloud_ptr->emplace_back(p);
    }
  }
  RCLCPP_INFO(
    this->get_logger(), "\t voxel filtered points: %lu", voxel_filtered_pointcloud_ptr->size());

  // K-search
  pcl::PointCloud<common_types::PointType>::Ptr tree_filtered_pointcloud_ptr(
    new pcl::PointCloud<common_types::PointType>);
  tree_filtered_pointcloud_ptr->reserve(voxel_filtered_pointcloud_ptr->size());
  float min_foreground_square_distance =
    parameters_.min_foreground_distance * parameters_.min_foreground_distance;

  for (const auto & p : voxel_filtered_pointcloud_ptr->points) {
    std::vector<int> indexes;
    std::vector<float> square_distances;

    if (background_model.tree_.nearestKSearch(p, 1, indexes, square_distances) > 0) {
      if (square_distances.size() == 0 || square_distances[0] >= min_foreground_square_distance) {
        tree_filtered_pointcloud_ptr->emplace_back(p);
      }
    }
  }

  RCLCPP_INFO(
    this->get_logger(), "\t tree filtered points: %lu", tree_filtered_pointcloud_ptr->size());

  if (!use_ransac) {
    foreground_pointcloud_ptr = tree_filtered_pointcloud_ptr;
    return;
  }

  // Plane ransac (since the orientation changes slightly between data, this one does not use the
  // background model)
  pcl::ModelCoefficients::Ptr coefficients_ptr(new pcl::ModelCoefficients);
  pcl::PointCloud<common_types::PointType>::Ptr ransac_filtered_pointcloud_ptr(
    new pcl::PointCloud<common_types::PointType>);
  pcl::PointIndices::Ptr inliers_ptr(new pcl::PointIndices);
  pcl::SACSegmentation<common_types::PointType> seg;
  pcl::ExtractIndices<common_types::PointType> extract;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);  // cSpell:ignore SACMODEL
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(parameters_.ransac_threshold);
  seg.setMaxIterations(parameters_.ransac_max_iterations);
  seg.setInputCloud(sensor_pointcloud_ptr);
  seg.segment(*inliers_ptr, *coefficients_ptr);

  ransac_filtered_pointcloud_ptr->reserve(tree_filtered_pointcloud_ptr->size());

  for (const auto & p : tree_filtered_pointcloud_ptr->points) {
    if (
      p.x * coefficients_ptr->values[0] + p.y * coefficients_ptr->values[1] +
        p.z * coefficients_ptr->values[2] + coefficients_ptr->values[3] >
      parameters_.ransac_threshold) {
      ransac_filtered_pointcloud_ptr->emplace_back(p);
    }
  }

  RCLCPP_INFO(
    this->get_logger(), "\t ransac filtered points: %lu", ransac_filtered_pointcloud_ptr->size());

  foreground_pointcloud_ptr = ransac_filtered_pointcloud_ptr;
  ground_model = Eigen::Vector4d(
    coefficients_ptr->values[0], coefficients_ptr->values[1], coefficients_ptr->values[2],
    coefficients_ptr->values[3]);
}

std::vector<pcl::PointCloud<common_types::PointType>::Ptr>
ExtrinsicReflectorBasedCalibrator::extractClusters(
  const pcl::PointCloud<common_types::PointType>::Ptr & foreground_pointcloud_ptr,
  const double cluster_max_tolerance, const int cluster_min_points, const int cluster_max_points)
{
  pcl::search::KdTree<common_types::PointType>::Ptr tree_ptr(
    new pcl::search::KdTree<common_types::PointType>);
  tree_ptr->setInputCloud(foreground_pointcloud_ptr);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<common_types::PointType> cluster_extractor;
  cluster_extractor.setClusterTolerance(cluster_max_tolerance);
  cluster_extractor.setMinClusterSize(cluster_min_points);
  cluster_extractor.setMaxClusterSize(cluster_max_points);
  cluster_extractor.setSearchMethod(tree_ptr);
  cluster_extractor.setInputCloud(foreground_pointcloud_ptr);
  cluster_extractor.extract(cluster_indices);

  RCLCPP_INFO(
    this->get_logger(), "Cluster extraction input size: %lu", foreground_pointcloud_ptr->size());

  std::vector<pcl::PointCloud<common_types::PointType>::Ptr> cluster_vector;

  for (const auto & cluster : cluster_indices) {
    pcl::PointCloud<common_types::PointType>::Ptr cluster_pointcloud_ptr(
      new pcl::PointCloud<common_types::PointType>);
    cluster_pointcloud_ptr->reserve(cluster.indices.size());

    for (const auto & idx : cluster.indices) {
      cluster_pointcloud_ptr->push_back((*foreground_pointcloud_ptr)[idx]);
    }

    cluster_pointcloud_ptr->width = cluster_pointcloud_ptr->size();
    cluster_pointcloud_ptr->height = 1;
    cluster_pointcloud_ptr->is_dense = true;
    RCLCPP_INFO(
      this->get_logger(), "\t found cluster of size: %lu", cluster_pointcloud_ptr->size());

    cluster_vector.push_back(cluster_pointcloud_ptr);
  }

  return cluster_vector;
}

std::vector<Eigen::Vector3d> ExtrinsicReflectorBasedCalibrator::findReflectorsFromClusters(
  const std::vector<pcl::PointCloud<common_types::PointType>::Ptr> & clusters,
  const Eigen::Vector4d & ground_model)
{
  std::vector<Eigen::Vector3d> reflector_centers;
  RCLCPP_INFO(this->get_logger(), "Extracting lidar reflectors from clusters");

  for (const auto & cluster_pointcloud_ptr : clusters) {
    float max_h = -std::numeric_limits<float>::max();
    common_types::PointType highest_point;

    for (const auto & p : cluster_pointcloud_ptr->points) {
      float height =
        p.x * ground_model.x() + p.y * ground_model.y() + p.z * ground_model.z() + ground_model.w();
      if (height > max_h) {
        max_h = height;
        highest_point = p;
      }
    }

    if (max_h > parameters_.reflector_max_height) {
      continue;
    }

    pcl::search::KdTree<common_types::PointType>::Ptr tree_ptr(
      new pcl::search::KdTree<common_types::PointType>);
    tree_ptr->setInputCloud(cluster_pointcloud_ptr);

    std::vector<int> indexes;
    std::vector<float> squared_distances;

    if (
      tree_ptr->radiusSearch(
        highest_point, parameters_.reflector_radius, indexes, squared_distances) > 0) {
      Eigen::Vector3d center = Eigen::Vector3d::Zero();

      for (const auto & index : indexes) {
        const auto & p = cluster_pointcloud_ptr->points[index];
        center += Eigen::Vector3d(p.x, p.y, p.z);
      }

      center /= indexes.size();
      RCLCPP_INFO(
        this->get_logger(), "\t Lidar reflector id=%lu size=%lu center: x=%.2f y=%.2f z=%.2f",
        reflector_centers.size(), indexes.size(), center.x(), center.y(), center.z());
      reflector_centers.push_back(center);
    }
  }

  return reflector_centers;
}

bool ExtrinsicReflectorBasedCalibrator::checkInitialTransforms()
{
  if (lidar_frame_ == "" || radar_frame_ == "") {
    return false;
  }

  if (got_initial_transform_) {
    return true;
  }

  try {
    rclcpp::Time t = rclcpp::Time(0);
    rclcpp::Duration timeout = rclcpp::Duration::from_seconds(1.0);

    initial_radar_to_lidar_msg_ =
      tf_buffer_->lookupTransform(radar_frame_, lidar_frame_, t, timeout).transform;

    initial_radar_to_lidar_eigen_ = tf2::transformToEigen(initial_radar_to_lidar_msg_);
    calibrated_radar_to_lidar_eigen_ = initial_radar_to_lidar_eigen_;

    radar_optimization_to_lidar_msg_ =
      tf_buffer_->lookupTransform(parameters_.radar_optimization_frame, lidar_frame_, t, timeout)
        .transform;

    radar_optimization_to_lidar_eigen_ = tf2::transformToEigen(radar_optimization_to_lidar_msg_);

    initial_radar_optimization_to_radar_msg_ =
      tf_buffer_->lookupTransform(parameters_.radar_optimization_frame, radar_frame_, t, timeout)
        .transform;

    initial_radar_optimization_to_radar_eigen_ =
      tf2::transformToEigen(initial_radar_optimization_to_radar_msg_);

    got_initial_transform_ = true;
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN(this->get_logger(), "could not get initial tf. %s", ex.what());
    return false;
  }

  return true;
}

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>
ExtrinsicReflectorBasedCalibrator::matchDetections(
  const std::vector<Eigen::Vector3d> & lidar_detections,
  const std::vector<Eigen::Vector3d> & radar_detections)
{
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> matched_detections;

  if (lidar_detections.size() == 0 || radar_detections.size() == 0 || !checkInitialTransforms()) {
    return matched_detections;
  }

  // Lidar transformed detections
  std::vector<Eigen::Vector3d> lidar_detections_transformed;
  const auto radar_to_lidar_transform = initial_radar_to_lidar_eigen_;

  std::transform(
    lidar_detections.cbegin(), lidar_detections.cend(),
    std::back_inserter(lidar_detections_transformed),
    [&radar_to_lidar_transform,
     &transformation_type = this->transformation_type_](const auto & lidar_detection) {
      auto transformed_point = radar_to_lidar_transform * lidar_detection;
      if (
        transformation_type == TransformationType::yaw_only_rotation_2d ||
        transformation_type == TransformationType::svd_2d) {
        transformed_point.z() = 0.f;
      }
      return transformed_point;
    });

  RCLCPP_INFO(
    this->get_logger(),
    "Lidar reflectors in radar coordinate system (using the initial transformation)");
  for (std::size_t lidar_index = 0; lidar_index < lidar_detections.size(); lidar_index++) {
    const auto & lidar_detection = lidar_detections_transformed[lidar_index];
    RCLCPP_INFO(
      this->get_logger(), "\t Lidar reflector (rcs) id=%lu size=%lu center: x=%.2f y=%.2f z=%.2f",
      lidar_index, lidar_detections.size(), lidar_detection.x(), lidar_detection.y(),
      lidar_detection.z());
  }

  std::vector<std::size_t> lidar_to_radar_closest_idx, radar_to_lidar_closest_idx;
  lidar_to_radar_closest_idx.resize(lidar_detections.size());
  radar_to_lidar_closest_idx.resize(radar_detections.size());

  RCLCPP_INFO(this->get_logger(), "Matching each lidar detections to its closest radar detection");

  for (std::size_t lidar_index = 0; lidar_index < lidar_detections.size(); lidar_index++) {
    float closest_distance = std::numeric_limits<float>::max();
    std::size_t closest_index = 0;

    for (std::size_t radar_index = 0; radar_index < radar_detections.size(); radar_index++) {
      float distance =
        (lidar_detections_transformed[lidar_index] - radar_detections[radar_index]).norm();

      if (distance < closest_distance) {
        closest_distance = distance;
        closest_index = radar_index;
      }
    }

    RCLCPP_INFO(
      this->get_logger(), "\tClosest radar to lidar=%lu is %lu with distance %f", lidar_index,
      closest_index, closest_distance);
    lidar_to_radar_closest_idx[lidar_index] = closest_index;
  }

  RCLCPP_INFO(this->get_logger(), "Matching each radar detections to its closest lidar detection");

  for (std::size_t radar_index = 0; radar_index < radar_detections.size(); radar_index++) {
    float closest_distance = std::numeric_limits<float>::max();
    std::size_t closest_index = 0;

    for (std::size_t lidar_index = 0; lidar_index < lidar_detections.size(); lidar_index++) {
      float distance =
        (lidar_detections_transformed[lidar_index] - radar_detections[radar_index]).norm();

      if (distance < closest_distance) {
        closest_distance = distance;
        closest_index = lidar_index;
      }
    }

    RCLCPP_INFO(
      this->get_logger(), "\tClosest lidar to radar=%lu is %lu with distance %f", radar_index,
      closest_index, closest_distance);

    radar_to_lidar_closest_idx[radar_index] = closest_index;
  }

  for (std::size_t lidar_index = 0; lidar_index < lidar_detections.size(); lidar_index++) {
    std::size_t closest_radar_index = lidar_to_radar_closest_idx[lidar_index];
    RCLCPP_INFO(
      this->get_logger(), "lidar_index = %lu / %lu", lidar_index, lidar_detections.size());
    RCLCPP_INFO(
      this->get_logger(), "closest_radar_index = %lu / %lu", closest_radar_index,
      radar_detections.size());
    float distance =
      (lidar_detections_transformed[lidar_index] - radar_detections[closest_radar_index]).norm();
    if (
      radar_to_lidar_closest_idx[closest_radar_index] == lidar_index &&
      distance < parameters_.max_matching_distance) {
      matched_detections.emplace_back(
        lidar_detections[lidar_index], radar_detections[closest_radar_index]);
    }
  }

  return matched_detections;
}

bool ExtrinsicReflectorBasedCalibrator::trackMatches(
  const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> & matches,
  builtin_interfaces::msg::Time & current_time)
{
  std::unique_lock<std::mutex> lock(mutex_);

  if (!tracking_active_) {
    return false;
  }

  // Check if active tracks expired
  auto timed_out_end = std::remove_if(
    active_tracks_.begin(), active_tracks_.end(),
    [&current_time](const auto & track) { return track.timedOut(current_time); });

  active_tracks_.erase(timed_out_end, active_tracks_.end());

  // Update tracks
  for (const auto & [lidar_detection, radar_detection] : matches) {
    if (std::any_of(
          converged_tracks_.begin(), converged_tracks_.end(),
          [&lidar_detection, &radar_detection](auto & track) {
            return track.partialMatch(lidar_detection, radar_detection);
          })) {
      continue;
    }

    bool new_track = std::all_of(
      active_tracks_.begin(), active_tracks_.end(),
      [&lidar_detection, &radar_detection](auto & track) {
        return !track.match(lidar_detection, radar_detection);
      });

    if (new_track) {
      active_tracks_.push_back(
        factory_ptr_->makeTrack(lidar_detection, radar_detection, current_time));
    } else {
      std::for_each(
        active_tracks_.begin(), active_tracks_.end(),
        [&lidar_detection, &radar_detection](auto & track) {
          track.updateIfMatch(lidar_detection, radar_detection);
        });
    }
  }

  // Move converged tracks
  std::copy_if(
    active_tracks_.begin(), active_tracks_.end(), std::back_inserter(converged_tracks_),
    [](auto & track) { return track.converged(); });

  current_new_tracks_ += std::transform_reduce(
    active_tracks_.begin(), active_tracks_.end(), 0, std::plus{},
    [](auto & track) { return track.converged(); });

  bool is_track_converged = std::any_of(
    active_tracks_.begin(), active_tracks_.end(), [](auto & track) { return track.converged(); });

  auto converged_end = std::remove_if(
    active_tracks_.begin(), active_tracks_.end(),
    [&current_time](auto & track) { return track.converged(); });

  active_tracks_.erase(converged_end, active_tracks_.end());

  if (current_new_tracks_ > 0 && active_tracks_.size() == 0) {
    tracking_active_ = false;
  }

  return is_track_converged;
}

std::tuple<double, double> ExtrinsicReflectorBasedCalibrator::get2DRotationDelta(
  std::vector<Track> converged_tracks, bool is_crossval)
{
  double delta_cos_sum = 0.0;
  double delta_sin_sum = 0.0;

  for (std::size_t track_index = 0; track_index < converged_tracks.size(); track_index++) {
    auto track = converged_tracks[track_index];
    // lidar coordinates
    const auto & lidar_estimation = track.getLidarEstimation();
    // to radar coordinates
    const auto & lidar_transformed_estimation = initial_radar_to_lidar_eigen_ * lidar_estimation;
    const auto & radar_estimation_rcs = track.getRadarEstimation();

    const double lidar_transformed_norm = lidar_transformed_estimation.norm();
    const double lidar_transformed_cos = lidar_transformed_estimation.x() / lidar_transformed_norm;
    const double lidar_transformed_sin = lidar_transformed_estimation.y() / lidar_transformed_norm;

    const double radar_norm = radar_estimation_rcs.norm();
    const double radar_cos = radar_estimation_rcs.x() / radar_norm;
    const double radar_sin = radar_estimation_rcs.y() / radar_norm;

    // sin(a-b) = sin(a)*cos(b) - cos(a)*sin(b)
    // cos(a-b) = cos(a)*cos(b) + sin(a)*sin(b)
    // a = lidar, b = radar
    double delta_angle_sin = lidar_transformed_sin * radar_cos - lidar_transformed_cos * radar_sin;
    double delta_angle_cos = lidar_transformed_cos * radar_cos + lidar_transformed_sin * radar_sin;
    delta_sin_sum += delta_angle_sin;
    delta_cos_sum += delta_angle_cos;

    if (!is_crossval) {
      // logging
      RCLCPP_INFO_STREAM(this->get_logger(), "lidar_estimation:\n" << lidar_estimation.matrix());
      RCLCPP_INFO_STREAM(
        this->get_logger(), "lidar_transformed_estimation:\n"
                              << lidar_transformed_estimation.matrix());
      RCLCPP_INFO_STREAM(
        this->get_logger(), "radar_estimation_rcs:\n"
                              << radar_estimation_rcs.matrix());
    }
  }
  double delta_cos = delta_cos_sum / converged_tracks.size();
  double delta_sin = -delta_sin_sum / converged_tracks.size();

  return {delta_cos, delta_sin};
}

std::tuple<
  pcl::PointCloud<common_types::PointType>::Ptr, pcl::PointCloud<common_types::PointType>::Ptr>
ExtrinsicReflectorBasedCalibrator::getPointsSet()
{
  // Note: ocs=radar optimization coordinate system rcs=radar coordinate system
  pcl::PointCloud<common_types::PointType>::Ptr lidar_points_ocs(
    new pcl::PointCloud<common_types::PointType>);
  pcl::PointCloud<common_types::PointType>::Ptr radar_points_rcs(
    new pcl::PointCloud<common_types::PointType>);
  lidar_points_ocs->reserve(converged_tracks_.size());
  radar_points_rcs->reserve(converged_tracks_.size());

  auto eigen_to_pcl_2d = [](const auto & p) { return common_types::PointType(p.x(), p.y(), 0.0); };
  auto eigen_to_pcl_3d = [](const auto & p) {
    return common_types::PointType(p.x(), p.y(), p.z());
  };

  for (std::size_t track_index = 0; track_index < converged_tracks_.size(); track_index++) {
    auto track = converged_tracks_[track_index];
    // lidar coordinates
    const auto & lidar_estimation = track.getLidarEstimation();
    // to radar optimization coordinates
    const auto & lidar_estimation_ocs = radar_optimization_to_lidar_eigen_ * lidar_estimation;
    // to radar coordinates
    const auto & lidar_transformed_estimation = initial_radar_to_lidar_eigen_ * lidar_estimation;
    const auto & radar_estimation_rcs = track.getRadarEstimation();

    if (transformation_type_ == TransformationType::svd_2d) {
      lidar_points_ocs->emplace_back(eigen_to_pcl_2d(lidar_estimation_ocs));
      radar_points_rcs->emplace_back(eigen_to_pcl_2d(radar_estimation_rcs));
    } else {
      lidar_points_ocs->emplace_back(eigen_to_pcl_3d(lidar_estimation_ocs));
      radar_points_rcs->emplace_back(eigen_to_pcl_3d(radar_estimation_rcs));
    }
    // logging
    RCLCPP_INFO_STREAM(this->get_logger(), "lidar_estimation:\n" << lidar_estimation.matrix());
    RCLCPP_INFO_STREAM(
      this->get_logger(), "lidar_transformed_estimation:\n"
                            << lidar_transformed_estimation.matrix());
    RCLCPP_INFO_STREAM(
      this->get_logger(), "radar_estimation_rcs:\n"
                            << radar_estimation_rcs.matrix());
  }
  return {lidar_points_ocs, radar_points_rcs};
}

std::pair<double, double> ExtrinsicReflectorBasedCalibrator::computeCalibrationError(
  const Eigen::Isometry3d & radar_to_lidar_isometry)
{
  double distance_error = 0.0;
  double yaw_error = 0.0;

  for (auto & track : converged_tracks_) {
    auto lidar_estimation = track.getLidarEstimation();
    auto radar_estimation = track.getRadarEstimation();
    auto lidar_estimation_transformed = radar_to_lidar_isometry * lidar_estimation;

    if (
      transformation_type_ == TransformationType::yaw_only_rotation_2d ||
      transformation_type_ == TransformationType::svd_2d) {
      lidar_estimation_transformed.z() = 0.0;
      radar_estimation.z() = 0.0;
    }
    distance_error += (lidar_estimation_transformed - radar_estimation).norm();
    yaw_error += getYawError(lidar_estimation_transformed, radar_estimation);
  }

  distance_error /= static_cast<double>(converged_tracks_.size());
  yaw_error *= 180.0 / (M_PI * static_cast<double>(converged_tracks_.size()));

  return std::make_pair(distance_error, yaw_error);
}

TransformationResult ExtrinsicReflectorBasedCalibrator::estimateTransformation()
{
  TransformationResult transformation_result;
  TransformationEstimator estimator(
    initial_radar_to_lidar_eigen_, initial_radar_optimization_to_radar_eigen_,
    radar_optimization_to_lidar_eigen_);

  if (transformation_type_ == TransformationType::yaw_only_rotation_2d) {
    auto [delta_cos, delta_sin] = get2DRotationDelta(converged_tracks_, false);
    estimator.setDelta(delta_cos, delta_sin);
    estimator.estimateYawOnlyTransformation();
    transformation_result.calibrated_radar_to_lidar_transformation = estimator.getTransformation();
    RCLCPP_INFO_STREAM(
      this->get_logger(), "Initial radar->lidar transform:\n"
                            << initial_radar_to_lidar_eigen_.matrix());
    RCLCPP_INFO_STREAM(
      this->get_logger(),
      "Pure rotation calibration radar->lidar transform:\n"
        << transformation_result.calibrated_radar_to_lidar_transformation.matrix());
  } else if (transformation_type_ == TransformationType::svd_2d) {
    std::tie(transformation_result.lidar_points_ocs, transformation_result.radar_points_rcs) =
      getPointsSet();
    estimator.setPoints(
      transformation_result.lidar_points_ocs, transformation_result.radar_points_rcs);
    estimator.estimateSVDTransformation(transformation_type_);
    transformation_result.calibrated_radar_to_lidar_transformation = estimator.getTransformation();
    RCLCPP_INFO_STREAM(
      this->get_logger(), "Initial radar->lidar transform:\n"
                            << initial_radar_to_lidar_eigen_.matrix());
    RCLCPP_INFO_STREAM(
      this->get_logger(),
      "2D calibration radar->lidar transform:\n"
        << transformation_result.calibrated_radar_to_lidar_transformation.matrix());
  } else if (
    transformation_type_ == TransformationType::svd_3d ||
    transformation_type_ == TransformationType::zero_roll_3d) {
    std::tie(transformation_result.lidar_points_ocs, transformation_result.radar_points_rcs) =
      getPointsSet();
    estimator.setPoints(
      transformation_result.lidar_points_ocs, transformation_result.radar_points_rcs);

    if (transformation_type_ == TransformationType::zero_roll_3d)
      estimator.estimateZeroRollTransformation();
    else if (transformation_type_ == TransformationType::svd_3d)
      estimator.estimateSVDTransformation(transformation_type_);

    transformation_result.calibrated_radar_to_lidar_transformation = estimator.getTransformation();
    RCLCPP_INFO_STREAM(
      this->get_logger(), "Initial radar->lidar transform:\n"
                            << initial_radar_to_lidar_eigen_.matrix());
    RCLCPP_INFO_STREAM(
      this->get_logger(),
      "3D calibration radar->lidar transform:\n"
        << transformation_result.calibrated_radar_to_lidar_transformation.matrix());
  }

  return transformation_result;
}

void ExtrinsicReflectorBasedCalibrator::evaluateTransformation(
  Eigen::Isometry3d calibrated_radar_to_lidar_transformation)
{
  // Estimate the pre & post calibration error
  auto [initial_distance_error, initial_yaw_error] =
    computeCalibrationError(initial_radar_to_lidar_eigen_);
  auto [calibrated_distance_error, calibrated_yaw_error] =
    computeCalibrationError(calibrated_radar_to_lidar_transformation);

  RCLCPP_INFO(
    this->get_logger(),
    "Initial calibration error: detection2detection.distance=%.4fm yaw=%.4f degrees",
    initial_distance_error, initial_yaw_error);
  RCLCPP_INFO(
    this->get_logger(),
    "Final calibration error: detection2detection.distance=%.4fm yaw=%.4f degrees",
    calibrated_distance_error, calibrated_yaw_error);

  auto compute_transformation_difference =
    [](const Eigen::Isometry3d & t1, const Eigen::Isometry3d & t2) -> std::pair<double, double> {
    double translation_difference = (t2.inverse() * t1).translation().norm();
    double rotation_difference =
      std::acos(std::min(1.0, 0.5 * ((t2.rotation().inverse() * t1.rotation()).trace() - 1.0)));

    return std::make_pair(translation_difference, rotation_difference);
  };

  auto [calibrated_translation_difference, calibrated_rotation_difference] =
    compute_transformation_difference(
      initial_radar_to_lidar_eigen_, calibrated_radar_to_lidar_transformation);

  std::unique_lock<std::mutex> lock(mutex_);
  if (
    calibrated_translation_difference < parameters_.max_initial_calibration_translation_error &&
    calibrated_rotation_difference < parameters_.max_initial_calibration_rotation_error) {
    calibrated_radar_to_lidar_eigen_ = calibrated_radar_to_lidar_transformation;
    calibration_valid_ = true;
    calibration_distance_score_ = calibrated_distance_error;
    calibration_yaw_score_ = calibrated_yaw_error;
  } else {
    RCLCPP_WARN(
      this->get_logger(),
      "The calibrated poses differ considerably with the initial calibration. This may be either a "
      "fault of the algorithm or a bad calibration initialization");
  }
  output_metrics_.push_back(static_cast<float>(converged_tracks_.size()));
  output_metrics_.push_back(static_cast<float>(calibrated_distance_error));
  output_metrics_.push_back(static_cast<float>(calibrated_yaw_error));
}

void ExtrinsicReflectorBasedCalibrator::findCombinations(
  std::size_t n, std::size_t k, std::vector<std::size_t> & curr, std::size_t first_num,
  std::vector<std::vector<std::size_t>> & combinations)
{
  auto curr_size = curr.size();
  if (curr_size == k) {
    combinations.push_back(curr);
    return;
  }

  auto need = k - curr_size;
  auto remain = n - first_num + 1;
  auto available = remain - need;

  for (auto num = first_num; num <= first_num + available; num++) {
    curr.push_back(num);
    findCombinations(n, k, curr, num + 1, combinations);
    curr.pop_back();
  }

  return;
}

void ExtrinsicReflectorBasedCalibrator::selectCombinations(
  std::size_t tracks_size, std::size_t num_of_samples,
  std::vector<std::vector<std::size_t>> & combinations)
{
  RCLCPP_INFO(
    this->get_logger(),
    "Current number of combinations is: %zu, converged_tracks_size: %zu, num_of_samples: %zu",
    combinations.size(), tracks_size, num_of_samples);

  // random select the combinations if the number of combinations is too large
  if (combinations.size() > parameters_.max_number_of_combination_samples) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::shuffle(combinations.begin(), combinations.end(), mt);
    combinations.resize(parameters_.max_number_of_combination_samples);
    RCLCPP_WARN(
      this->get_logger(),
      "The number of combinations is set to: %zu, because it exceeds the maximum number of "
      "combination samples: %zu",
      combinations.size(), parameters_.max_number_of_combination_samples);
  }
}

void ExtrinsicReflectorBasedCalibrator::evaluateCombinations(
  std::vector<std::vector<std::size_t>> & combinations, std::size_t num_of_samples,
  TransformationResult transformation_result)
{
  TransformationEstimator crossval_estimator(
    initial_radar_to_lidar_eigen_, initial_radar_optimization_to_radar_eigen_,
    radar_optimization_to_lidar_eigen_);

  pcl::PointCloud<common_types::PointType>::Ptr crossval_lidar_points_ocs(
    new pcl::PointCloud<common_types::PointType>);
  pcl::PointCloud<common_types::PointType>::Ptr crossval_radar_points_rcs(
    new pcl::PointCloud<common_types::PointType>);
  std::vector<Track> crossval_converged_tracks_;
  crossval_lidar_points_ocs->reserve(num_of_samples);
  crossval_radar_points_rcs->reserve(num_of_samples);
  crossval_converged_tracks_.reserve(num_of_samples);

  double total_crossval_calibrated_distance_error = 0.0;
  double total_crossval_calibrated_yaw_error = 0.0;
  std::vector<double> crossval_calibrated_distance_error_vector;
  std::vector<double> crossval_calibrated_yaw_error_vector;

  for (const auto & combination : combinations) {
    if (transformation_type_ == TransformationType::yaw_only_rotation_2d) {
      crossval_converged_tracks_.clear();

      for (std::size_t i = 0; i < combination.size(); i++) {
        crossval_converged_tracks_.push_back(converged_tracks_[i]);
      }
      auto [delta_cos, delta_sin] = get2DRotationDelta(crossval_converged_tracks_, true);

      crossval_estimator.setDelta(delta_cos, delta_sin);
      crossval_estimator.estimateYawOnlyTransformation();
    } else {
      crossval_lidar_points_ocs->clear();
      crossval_radar_points_rcs->clear();

      // calculate the transformation.
      for (std::size_t i = 0; i < combination.size(); i++) {
        crossval_lidar_points_ocs->emplace_back(
          transformation_result.lidar_points_ocs->points[combination[i]]);
        crossval_radar_points_rcs->emplace_back(
          transformation_result.radar_points_rcs->points[combination[i]]);
      }
      crossval_estimator.setPoints(crossval_lidar_points_ocs, crossval_radar_points_rcs);
      if (transformation_type_ == TransformationType::zero_roll_3d)
        crossval_estimator.estimateZeroRollTransformation();
      else
        crossval_estimator.estimateSVDTransformation(transformation_type_);
    }

    Eigen::Isometry3d crossval_calibrated_radar_to_lidar_transformation =
      crossval_estimator.getTransformation();
    // calculate the error.
    auto [crossval_calibrated_distance_error, crossval_calibrated_yaw_error] =
      computeCalibrationError(crossval_calibrated_radar_to_lidar_transformation);

    total_crossval_calibrated_distance_error += crossval_calibrated_distance_error;
    total_crossval_calibrated_yaw_error += crossval_calibrated_yaw_error;
    crossval_calibrated_distance_error_vector.push_back(crossval_calibrated_distance_error);
    crossval_calibrated_yaw_error_vector.push_back(crossval_calibrated_yaw_error);
  }

  auto calculate_std = [](std::vector<double> & data, double mean) -> double {
    double sum = 0.0;
    for (std::size_t i = 0; i < data.size(); i++) {
      sum += (data[i] - mean) * (data[i] - mean);
    }
    double variance = sum / data.size();
    return std::sqrt(variance);
  };

  double avg_crossval_calibrated_distance_error =
    total_crossval_calibrated_distance_error / combinations.size();
  double avg_crossval_calibrated_yaw_error =
    total_crossval_calibrated_yaw_error / combinations.size();
  output_metrics_.push_back(static_cast<float>(num_of_samples));
  output_metrics_.push_back(static_cast<float>(avg_crossval_calibrated_distance_error));
  output_metrics_.push_back(static_cast<float>(avg_crossval_calibrated_yaw_error));

  double std_crossval_calibrated_distance_error = calculate_std(
    crossval_calibrated_distance_error_vector, avg_crossval_calibrated_distance_error);
  double std_crossval_calibrated_yaw_error =
    calculate_std(crossval_calibrated_yaw_error_vector, avg_crossval_calibrated_yaw_error);
  output_metrics_.push_back(static_cast<float>(std_crossval_calibrated_distance_error));
  output_metrics_.push_back(static_cast<float>(std_crossval_calibrated_yaw_error));
}

void ExtrinsicReflectorBasedCalibrator::crossValEvaluation(
  TransformationResult transformation_result)
{
  auto tracks_size = converged_tracks_.size();
  if (tracks_size <= 3) return;

  for (std::size_t num_of_samples = 3; num_of_samples < tracks_size; num_of_samples++) {
    std::vector<std::vector<std::size_t>> combinations;
    std::vector<std::size_t> curr;

    findCombinations(tracks_size - 1, num_of_samples, curr, 0, combinations);
    selectCombinations(tracks_size, num_of_samples, combinations);
    evaluateCombinations(combinations, num_of_samples, transformation_result);
  }
}

void ExtrinsicReflectorBasedCalibrator::publishMetrics()
{
  // The final format of the output metrics is
  // num of reflectors,  calibration_distance_error, calibration_yaw_error,
  // sample, avg_crossval_dis_error, avg_crossval_yaw_error, std_crossval_dis_error,
  // std_crossval_yaw_error. sample, ....
  std_msgs::msg::Float32MultiArray calibration_metrics_msg = std_msgs::msg::Float32MultiArray();
  calibration_metrics_msg.data = output_metrics_;
  metrics_pub_->publish(calibration_metrics_msg);
}

void ExtrinsicReflectorBasedCalibrator::calibrateSensors()
{
  if (!checkInitialTransforms() || converged_tracks_.size() == 0) {
    if (converged_tracks_.size() == 0) {
      std_msgs::msg::Float32MultiArray calibration_metrics_msg = std_msgs::msg::Float32MultiArray();
      calibration_metrics_msg.data = {0, 0, 0, 0, 0};
      metrics_pub_->publish(calibration_metrics_msg);
    }
    return;
  }
  output_metrics_.clear();
  auto transformation_result = estimateTransformation();
  evaluateTransformation(transformation_result.calibrated_radar_to_lidar_transformation);
  crossValEvaluation(transformation_result);
  publishMetrics();
}

void ExtrinsicReflectorBasedCalibrator::visualizationMarkers(
  const std::vector<Eigen::Vector3d> & lidar_detections,
  const std::vector<Eigen::Vector3d> & radar_detections,
  const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> & matched_detections)
{
  visualization_msgs::msg::MarkerArray lidar_detections_marker_array;

  for (std::size_t detection_index = 0; detection_index < lidar_detections.size();
       detection_index++) {
    const auto & detection_center = lidar_detections[detection_index];
    visualization_msgs::msg::Marker marker;
    marker.header = lidar_header_;
    marker.lifetime = rclcpp::Duration::from_seconds(0.5);
    marker.id = detection_index;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = detection_center.x();
    marker.pose.position.y = detection_center.y();
    marker.pose.position.z = detection_center.z();
    marker.pose.orientation.w = 1.0;
    marker.scale.x = parameters_.reflector_radius;
    marker.scale.y = parameters_.reflector_radius;
    marker.scale.z = parameters_.reflector_radius;
    marker.color.a = 0.6;
    marker.color.r = 0.0;
    marker.color.g = 0.0;
    marker.color.b = 1.0;
    lidar_detections_marker_array.markers.push_back(marker);
  }

  lidar_detections_pub_->publish(lidar_detections_marker_array);

  visualization_msgs::msg::MarkerArray radar_detections_marker_array;

  for (std::size_t detection_index = 0; detection_index < radar_detections.size();
       detection_index++) {
    const auto & detection_center = radar_detections[detection_index];
    visualization_msgs::msg::Marker marker;
    marker.header = radar_header_;
    marker.id = detection_index;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.lifetime = rclcpp::Duration::from_seconds(0.5);
    marker.ns = "center";
    marker.pose.position.x = detection_center.x();
    marker.pose.position.y = detection_center.y();
    marker.pose.position.z = detection_center.z();
    marker.pose.orientation.w = 1.0;
    marker.scale.x = parameters_.reflector_radius;
    marker.scale.y = parameters_.reflector_radius;
    marker.scale.z = parameters_.reflector_radius;
    marker.color.a = 0.6;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 1.0;
    radar_detections_marker_array.markers.push_back(marker);

    geometry_msgs::msg::Point p1, p2;
    p1.z -= 0.5;
    p2.z += 0.5;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.ns = "line";
    marker.scale.x = 0.2 * parameters_.reflector_radius;
    marker.scale.y = 0.2 * parameters_.reflector_radius;
    marker.scale.z = 0.2 * parameters_.reflector_radius;
    marker.points.push_back(p1);
    marker.points.push_back(p2);
    radar_detections_marker_array.markers.push_back(marker);
  }

  radar_detections_pub_->publish(radar_detections_marker_array);

  visualization_msgs::msg::MarkerArray matches_marker_array;

  for (std::size_t match_index = 0; match_index < matched_detections.size(); match_index++) {
    const auto & [lidar_detection, radar_detection] = matched_detections[match_index];
    const auto lidar_detection_transformed = initial_radar_to_lidar_eigen_ * lidar_detection;

    visualization_msgs::msg::Marker marker;
    marker.header = radar_header_;
    marker.id = match_index;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.lifetime = rclcpp::Duration::from_seconds(0.5);
    marker.ns = "match";
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.02;
    marker.scale.y = 0.02;
    marker.scale.z = 0.02;
    marker.color.a = 0.6;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 1.0;
    marker.points.push_back(eigenToPointMsg(lidar_detection_transformed));
    marker.points.push_back(eigenToPointMsg(radar_detection));
    matches_marker_array.markers.push_back(marker);
  }

  matches_markers_pub_->publish(matches_marker_array);
}

void ExtrinsicReflectorBasedCalibrator::visualizeTrackMarkers()
{
  auto add_track_markers = [&](
                             const Eigen::Vector3d & lidar_estimation,
                             const Eigen::Vector3d & radar_estimation_transformed,
                             const std::string ns, const std_msgs::msg::ColorRGBA & color,
                             std::vector<visualization_msgs::msg::Marker> & markers) {
    visualization_msgs::msg::Marker marker;

    marker.header = lidar_header_;
    marker.id = markers.size();
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.ns = ns;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.2 * parameters_.reflector_radius;
    marker.scale.y = 0.2 * parameters_.reflector_radius;
    marker.scale.z = 0.2 * parameters_.reflector_radius;
    marker.color = color;
    marker.points.push_back(eigenToPointMsg(radar_estimation_transformed));
    marker.points.push_back(eigenToPointMsg(lidar_estimation));
    markers.push_back(marker);

    marker.id = markers.size();
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.pose.position = eigenToPointMsg(radar_estimation_transformed);
    marker.pose.orientation.w = 1.0;
    marker.scale.x = parameters_.reflector_radius;
    marker.scale.y = parameters_.reflector_radius;
    marker.scale.z = parameters_.reflector_radius;
    marker.points.clear();
    markers.push_back(marker);

    marker.id = markers.size();
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.scale.x = 0.2 * parameters_.reflector_radius;
    marker.scale.y = 0.2 * parameters_.reflector_radius;
    marker.scale.z = 0.2 * parameters_.reflector_radius;
    marker.points.push_back(eigenToPointMsg(Eigen::Vector3d(0.0, 0.0, -0.5)));
    marker.points.push_back(eigenToPointMsg(Eigen::Vector3d(0.0, 0.0, 0.5)));
    markers.push_back(marker);

    marker.id = markers.size();
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.pose.position = eigenToPointMsg(lidar_estimation);
    marker.pose.orientation.w = 1.0;
    marker.scale.x = parameters_.reflector_radius;
    marker.scale.y = parameters_.reflector_radius;
    marker.scale.z = parameters_.reflector_radius;
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 1.0;
    marker.points.clear();
    markers.push_back(marker);
  };

  // Visualization
  visualization_msgs::msg::MarkerArray tracking_marker_array;
  std_msgs::msg::ColorRGBA initial_color;
  initial_color.r = 1.0;
  initial_color.a = 1.0;

  std_msgs::msg::ColorRGBA calibrated_color;
  calibrated_color.g = 1.0;
  calibrated_color.a = 1.0;

  for (std::size_t track_index = 0; track_index < converged_tracks_.size(); track_index++) {
    auto & track = converged_tracks_[track_index];
    const auto & lidar_estimation = track.getLidarEstimation();
    const auto & radar_estimation = track.getRadarEstimation();

    const auto initial_radar_estimation_transformed =
      initial_radar_to_lidar_eigen_.inverse() * radar_estimation;
    const auto calibrated_radar_estimation_transformed =
      calibrated_radar_to_lidar_eigen_.inverse() * radar_estimation;

    add_track_markers(
      lidar_estimation, initial_radar_estimation_transformed, "initial", initial_color,
      tracking_marker_array.markers);
    add_track_markers(
      lidar_estimation, calibrated_radar_estimation_transformed, "calibrated", calibrated_color,
      tracking_marker_array.markers);
  }

  tracking_markers_pub_->publish(tracking_marker_array);
}

void ExtrinsicReflectorBasedCalibrator::deleteTrackMarkers()
{
  visualization_msgs::msg::MarkerArray tracking_marker_array;
  visualization_msgs::msg::Marker marker;
  auto deleted_id_start = converged_tracks_.size() * MARKER_SIZE_PER_TRACK;
  // delete the latest initial marker
  std::string ns = "initial";
  for (int i = 0; i < 4; i++) {
    marker.id = deleted_id_start + i;
    marker.ns = ns;
    marker.action = visualization_msgs::msg::Marker::DELETE;
    tracking_marker_array.markers.push_back(marker);
  }
  // delete the latest calibrated marker
  ns = "calibrated";
  for (int i = 4; i < 8; i++) {
    marker.id = deleted_id_start + i;
    marker.ns = ns;
    marker.action = visualization_msgs::msg::Marker::DELETE;
    tracking_marker_array.markers.push_back(marker);
  }

  tracking_markers_pub_->publish(tracking_marker_array);
}

void ExtrinsicReflectorBasedCalibrator::drawCalibrationStatusText()
{
  auto to_string_with_precision = [](const float value, const int n = 2) -> std::string {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << value;
    return out.str();
  };

  visualization_msgs::msg::Marker text_marker;

  text_marker.id = 0;
  text_marker.header = lidar_header_;
  text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  text_marker.color.r = 1.0;
  text_marker.color.g = 1.0;
  text_marker.color.b = 1.0;
  text_marker.color.a = 1.0;
  text_marker.ns = "calibration_status";
  text_marker.scale.z = 0.3;

  // show the latest cross validation results which is located in the last two elements of the
  // metrics vector show the latest calibration result, which is located in the 2nd and 3rd index of
  // the metrics vector
  constexpr double m_to_cm = 100.0;

  if (converged_tracks_.size() == 0) {
    text_marker.text = " pairs=" + std::to_string(converged_tracks_.size());
  } else {
    text_marker.text =
      " pairs=" + std::to_string(converged_tracks_.size()) +
      "\n average_distance_error[cm]=" + to_string_with_precision(output_metrics_[1] * m_to_cm) +
      "\n average_yaw_error[deg]=" + to_string_with_precision(output_metrics_[2]);

    if (converged_tracks_.size() > 3) {
      text_marker.text +=
        "\n crossval_distance_error[cm]=" +
        to_string_with_precision(output_metrics_[output_metrics_.size() - 4] * m_to_cm) +
        "\n crossval_yaw_error[deg]=" +
        to_string_with_precision(output_metrics_[output_metrics_.size() - 3]);
    }
  }

  text_marker.pose.position.x = 1.0;
  text_marker.pose.position.y = 1.0;
  text_marker.pose.position.z = 1.0;
  text_marker.pose.orientation.x = 0.0;
  text_marker.pose.orientation.y = 0.0;
  text_marker.pose.orientation.z = 0.0;
  text_marker.pose.orientation.w = 1.0;

  text_markers_pub_->publish(text_marker);
}

geometry_msgs::msg::Point ExtrinsicReflectorBasedCalibrator::eigenToPointMsg(
  const Eigen::Vector3d & p_eigen)
{
  geometry_msgs::msg::Point p;
  p.x = p_eigen.x();
  p.y = p_eigen.y();
  p.z = p_eigen.z();
  return p;
}

double ExtrinsicReflectorBasedCalibrator::getYawError(
  const Eigen::Vector3d & v1, const Eigen::Vector3d & v2)
{
  return std::abs(std::acos(v1.dot(v2) / (v1.norm() * v2.norm())));
}

}  // namespace marker_radar_lidar_calibrator
