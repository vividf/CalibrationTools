// Copyright 2024 Tier IV, Inc.
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

#include "marker_radar_lidar_calibrator/types.hpp"

#include <Eigen/Dense>
#include <marker_radar_lidar_calibrator/utils.hpp>

#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace marker_radar_lidar_calibrator
{
std::string toString(TransformationType type)
{
  switch (type) {
    case TransformationType::svd_2d:
      return "svd_2d";
    case TransformationType::yaw_only_rotation_2d:
      return "yaw_only_rotation_2d";
    case TransformationType::svd_3d:
      return "svd_3d";
    case TransformationType::zero_roll_3d:
      return "zero_roll_3d";
    default:
      return "unknown";
  }
}

std::string toStringWithPrecision(const float value, const int n)
{
  std::ostringstream out;
  out.precision(n);
  out << std::fixed << value;
  return out.str();
}

geometry_msgs::msg::Point eigenToPointMsg(const Eigen::Vector3d & p_eigen)
{
  geometry_msgs::msg::Point p;
  p.x = p_eigen.x();
  p.y = p_eigen.y();
  p.z = p_eigen.z();
  return p;
}

void updateTrackIds(std::vector<Track> & converged_tracks)
{
  for (size_t i = 0; i < converged_tracks.size(); ++i) {
    converged_tracks[i].id = i + 1;  // Reassign IDs starting from 1
  }
}

std::pair<double, double> computeCalibrationError(
  std::vector<Track> & converged_tracks, TransformationType transformation_type,
  const Eigen::Isometry3d & radar_to_lidar_isometry)
{
  double total_distance_error = 0.0;
  double total_yaw_error = 0.0;

  for (auto & track : converged_tracks) {
    auto lidar_estimation_transformed = radar_to_lidar_isometry * track.lidar_estimation;

    auto distance_error =
      getDistanceError(transformation_type, lidar_estimation_transformed, track.radar_estimation);
    auto yaw_error = getYawError(lidar_estimation_transformed, track.radar_estimation);

    track.distance_error = distance_error;
    track.yaw_error = yaw_error * 180.0 / (M_PI);
    total_distance_error += distance_error;
    total_yaw_error += yaw_error;
  }

  total_distance_error /= static_cast<double>(converged_tracks.size());
  total_yaw_error *= 180.0 / (M_PI * static_cast<double>(converged_tracks.size()));

  return std::make_pair(total_distance_error, total_yaw_error);
}

double getDistanceError(
  TransformationType transformation_type, Eigen::Vector3d v1, Eigen::Vector3d v2)
{
  if (
    transformation_type == TransformationType::svd_2d ||
    transformation_type == TransformationType::yaw_only_rotation_2d) {
    v1.z() = 0.0;
    v2.z() = 0.0;
  }
  return (v1 - v2).norm();
}

double getYawError(Eigen::Vector3d v1, Eigen::Vector3d v2)
{
  v1.z() = 0.0;
  v2.z() = 0.0;
  return std::abs(std::acos(v1.dot(v2) / (v1.norm() * v2.norm())));
}

void findCombinations(
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

void selectCombinations(
  const rclcpp::Logger & logger, std::size_t tracks_size, std::size_t num_of_samples,
  std::vector<std::vector<std::size_t>> & combinations, size_t max_number_of_combination_samples)
{
  RCLCPP_INFO(
    logger,
    "Current number of combinations is: %zu, converged_tracks_size: %zu, num_of_samples: %zu",
    combinations.size(), tracks_size, num_of_samples);

  // random select the combinations if the number of combinations is too large
  if (combinations.size() > max_number_of_combination_samples) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::shuffle(combinations.begin(), combinations.end(), mt);
    combinations.resize(max_number_of_combination_samples);
    RCLCPP_WARN(
      logger,
      "The number of combinations is set to: %zu, because it exceeds the maximum number of "
      "combination samples: %zu",
      combinations.size(), max_number_of_combination_samples);
  }
}

void parseHeader(
  std::ifstream & file, const std::string & header_name, std_msgs::msg::Header & header)
{
  std::string line;

  // Parse the header section.
  while (std::getline(file, line)) {
    if (!line.empty()) break;  // Skip blank lines.
  }

  if (line != header_name) {
    throw std::runtime_error("Failed to find " + header_name + " section.");
  }

  if (!std::getline(file, line) || line.rfind("stamp_sec ", 0) != 0) {
    throw std::runtime_error("Missing or invalid stamp_sec in " + header_name + " section.");
  }
  header.stamp.sec = std::stoi(line.substr(10));

  if (!std::getline(file, line) || line.rfind("stamp_nanosec ", 0) != 0) {
    throw std::runtime_error("Missing or invalid stamp_nanosec in " + header_name + " section.");
  }
  header.stamp.nanosec = std::stoi(line.substr(14));

  if (!std::getline(file, line) || line.rfind("frame_id ", 0) != 0) {
    throw std::runtime_error("Missing or invalid frame_id in " + header_name + " section.");
  }
  header.frame_id = line.substr(9);
}

void parseConvergedTracks(std::ifstream & file, std::vector<Track> & converged_tracks)
{
  std::string line;

  // Skip the header line.
  if (!std::getline(file, line)) {
    throw std::runtime_error("File is empty or missing the header line.");
  }

  // Parse the point cloud data.
  while (std::getline(file, line) && !line.empty()) {
    if (line.find("matrix:") != std::string::npos) break;  // Stop if a matrix section starts.

    std::istringstream stream(line);
    std::vector<double> values;
    double value;

    while (stream >> value) {  // Parse values.
      values.push_back(value);
    }

    if (values.size() != 6) {
      throw std::runtime_error("Invalid number of values in line: " + line);
    }

    Track track;
    track.id = converged_tracks.size() + 1;
    track.lidar_estimation = Eigen::Vector3d(values[0], values[1], values[2]);
    track.radar_estimation = Eigen::Vector3d(values[3], values[4], values[5]);

    converged_tracks.emplace_back(std::move(track));
  }
}

}  // namespace marker_radar_lidar_calibrator
