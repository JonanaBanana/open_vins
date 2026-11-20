/*
 * GeoTNav addition to OpenVINS: a downward rangefinder as a horizontal-plane constraint.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_MSCKF_RANGE_PLANE_H
#define OV_MSCKF_RANGE_PLANE_H

#include <Eigen/Eigen>
#include <cmath>
#include <deque>
#include <memory>
#include <mutex>
#include <string>

#include "utils/colors.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"

namespace ov_msckf {

/**
 * @brief Options for the rangefinder plane constraint.
 *
 * The ground is taken as a plane perpendicular to gravity at the rangefinder's slant range along its
 * beam. Every MSCKF feature is constrained to that plane, which gives the features' depths, and so the
 * scale, a metric reference. See RangePlane in UpdaterMSCKF.
 *
 * The estimator settings (use_range, range_sigma, ...) are in estimator_config.yaml. The sensor is
 * described, like the IMU and cameras, in a separate file named by relative_config_range (read only when
 * use_range is on), under "range0":
 *
 *     range0:
 *       T_imu_range: 4x4, the rangefinder frame in the IMU frame; the beam is its +x axis (ROS convention)
 *       rostopic: topic name
 *       msg_type: range (sensor_msgs/Range) or laserscan (sensor_msgs/LaserScan, first range)
 *       time_offset: t_imu = t_range + time_offset [s]
 *
 * Requires feat_rep_msckf GLOBAL_3D. Readings are fed by the ROS 2 node (run_subscribe_msckf) only.
 */
struct RangeOptions {
  /// Use the constraint
  bool enabled = false;
  /// Topic of the rangefinder (range0: rostopic)
  std::string topic = "/range";
  /// Message type on that topic (range0: msg_type): "range" (sensor_msgs/Range) or "laserscan"
  /// (sensor_msgs/LaserScan, the first range). Readings outside the message's own limits or not finite are
  /// invalid.
  std::string msg_type = "range";
  /// Standard deviation of each feature's own error: how far it sits off the plane (vegetation, buildings,
  /// slopes) [m]. With sigma_shared 0 it also has to cover the range noise.
  double sigma = 3.0;
  /// Standard deviation of the error shared by every row that uses the same reading: range noise plus how
  /// far the beam's footprint is off the plane (canopy, roofs) [m]. A row's variance is
  /// sigma^2 + N * sigma_shared^2, with N the rows sharing its reading in one update, so one reading never
  /// counts for more than one measurement of sigma_shared. 0: rows are independent.
  double sigma_shared = 0.0;
  /// Residuals beyond huber_k sigmas get a proportionally larger sigma (bounded influence)
  double huber_k = 3.0;
  /// Valid ranges are in (min, max] [m], on top of the message's own limits. Default: no limits beyond the
  /// message's. Set max below a sensor's "no return" value if it reports one as a range.
  double min = 0.0;
  double max = INFINITY;
  /// Tilt of the beam from vertical beyond which the plane is not used [deg]
  double max_tilt_deg = 60.0;
  /// Largest gap between the two ranges interpolated at a clone's time [s]
  double max_gap = 0.1;
  /// Height above the plane below which no row is added [m]. Near the ground (take-off, landing) the
  /// features are close and small height errors are large relative to the height. Default: no limit.
  double min_height = 0.0;
  /// The rangefinder frame in the IMU frame (range0: T_imu_range, required). The beam is the frame's +x axis.
  Eigen::Matrix4d T_imu_range = Eigen::Matrix4d::Identity();
  /// Derived from T_imu_range: beam direction (its rotation's first column) and the beam's origin, in the
  /// IMU frame
  Eigen::Vector3d dir_in_imu = Eigen::Vector3d(1, 0, 0);
  Eigen::Vector3d p_in_imu = Eigen::Vector3d::Zero();
  /// Time offset of the range stamps (range0: time_offset): t_imu = t_range + time_offset [s]
  double time_offset = 0.0;

  void print_and_load(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
    if (parser != nullptr) {
      parser->parse_config("use_range", enabled, false);
      if (enabled) {
        parser->parse_external("relative_config_range", "range0", "T_imu_range", T_imu_range, true);
        parser->parse_external("relative_config_range", "range0", "rostopic", topic, true);
        parser->parse_external("relative_config_range", "range0", "msg_type", msg_type, false);
        parser->parse_external("relative_config_range", "range0", "time_offset", time_offset, false);
        if (msg_type != "range" && msg_type != "laserscan") {
          PRINT_ERROR(RED "invalid range0 msg_type: %s\n" RESET, msg_type.c_str());
          PRINT_ERROR(RED "please select a valid type: range (sensor_msgs/Range), laserscan (sensor_msgs/LaserScan)\n" RESET);
          std::exit(EXIT_FAILURE);
        }
        const Eigen::Matrix3d R = T_imu_range.block<3, 3>(0, 0);
        if ((R.transpose() * R - Eigen::Matrix3d::Identity()).norm() > 1e-3 || R.determinant() < 0.0 ||
            (T_imu_range.row(3) - Eigen::RowVector4d(0, 0, 0, 1)).norm() > 1e-9) {
          PRINT_ERROR(RED "range0 T_imu_range is not a rigid transform (rotation not orthonormal or last row not [0 0 0 1])\n" RESET);
          std::exit(EXIT_FAILURE);
        }
        dir_in_imu = R.col(0).normalized();
        p_in_imu = T_imu_range.block<3, 1>(0, 3);
      }
      parser->parse_config("range_sigma", sigma, false);
      parser->parse_config("range_sigma_shared", sigma_shared, false);
      parser->parse_config("range_huber_k", huber_k, false);
      parser->parse_config("range_min", min, false);
      parser->parse_config("range_max", max, false);
      parser->parse_config("range_max_tilt_deg", max_tilt_deg, false);
      parser->parse_config("range_max_gap", max_gap, false);
      parser->parse_config("range_min_height", min_height, false);
    }
    if (!enabled) {
      PRINT_INFO("RANGE PLANE: off\n");
      return;
    }
    PRINT_INFO("RANGE PLANE: on, topic %s (%s), sigma %.2f m (shared %.2f m), valid (%.1f, %.1f] m, tilt <= %.0f deg, height >= %.1f m, "
               "beam [%.2f %.2f %.2f] from [%.3f %.3f %.3f] m in the IMU frame, time offset %.4f s\n",
               topic.c_str(), msg_type.c_str(), sigma, sigma_shared, min, max, max_tilt_deg, min_height, dir_in_imu(0), dir_in_imu(1),
               dir_in_imu(2), p_in_imu(0), p_in_imu(1), p_in_imu(2), time_offset);
  }
};

/**
 * @brief A reading as the message reports it: NaN (invalid) if not finite or outside the message's own
 * limits [msg_min, msg_max] (sensor_msgs/Range: min_range, max_range; LaserScan: range_min, range_max).
 * Limits that are not finite or not positive are ignored.
 */
inline double range_within_msg_limits(double range, double msg_min, double msg_max) {
  if (!std::isfinite(range))
    return NAN;
  if (std::isfinite(msg_min) && msg_min > 0.0 && range < msg_min)
    return NAN;
  if (std::isfinite(msg_max) && msg_max > 0.0 && range > msg_max)
    return NAN;
  return range;
}

/**
 * @brief Thread-safe buffer of rangefinder readings (stamps in the IMU clock), interpolated at any time.
 */
class RangeBuffer {
public:
  explicit RangeBuffer(const RangeOptions &options) : _options(options) {}

  /// Adds a reading; invalid ones (no return, out of range) are kept as gaps.
  void feed(double time, double range) {
    std::lock_guard<std::mutex> lck(_mtx);
    if (!_data.empty() && time <= _data.back().first)
      return;
    const bool valid = std::isfinite(range) && range > _options.min && range <= _options.max;
    _data.emplace_back(time, valid ? range : NAN);
    while (!_data.empty() && _data.front().first < time - 30.0)
      _data.pop_front();
  }

  /// The range at `time`, interpolated between two valid readings at most max_gap apart.
  bool get(double time, double &range) const {
    std::lock_guard<std::mutex> lck(_mtx);
    if (_data.size() < 2 || time < _data.front().first || time > _data.back().first)
      return false;
    auto hi = std::lower_bound(_data.begin(), _data.end(), time,
                               [](const std::pair<double, double> &a, double t) { return a.first < t; });
    if (hi == _data.begin())
      hi++;
    const auto lo = std::prev(hi);
    if (std::isnan(lo->second) || std::isnan(hi->second) || hi->first - lo->first > _options.max_gap)
      return false;
    const double a = (time - lo->first) / (hi->first - lo->first);
    range = lo->second + a * (hi->second - lo->second);
    return true;
  }

  /// Time of the newest reading (valid or not); false if none was received yet.
  bool latest_time(double &time) const {
    std::lock_guard<std::mutex> lck(_mtx);
    if (_data.empty())
      return false;
    time = _data.back().first;
    return true;
  }

private:
  RangeOptions _options;
  mutable std::mutex _mtx;
  std::deque<std::pair<double, double>> _data;
};

} // namespace ov_msckf

#endif // OV_MSCKF_RANGE_PLANE_H
