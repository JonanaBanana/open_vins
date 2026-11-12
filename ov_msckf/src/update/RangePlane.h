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

#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"

namespace ov_msckf {

/**
 * @brief Options for the rangefinder plane constraint (estimator_config.yaml).
 *
 * The ground is taken as a plane perpendicular to gravity at the rangefinder's slant range along its
 * beam. Every MSCKF feature is constrained to that plane, which gives the features' depths, and so the
 * scale, a metric reference. See RangePlane in UpdaterMSCKF.
 */
struct RangeOptions {
  /// Use the constraint
  bool enabled = false;
  /// Topic of the rangefinder (sensor_msgs/LaserScan, the first range)
  std::string topic = "/alt_raw";
  /// Standard deviation of the predicted slant range: range noise plus how far features sit off the
  /// plane (vegetation, buildings, slopes) [m]
  double sigma = 3.0;
  /// Residuals beyond huber_k sigmas get a proportionally larger sigma (bounded influence)
  double huber_k = 3.0;
  /// Valid ranges are in (min, max] [m]; the SF30/D reports 250 m without a return
  double min = 0.0;
  double max = 190.0;
  /// Tilt of the beam from vertical beyond which the plane is not used [deg]
  double max_tilt_deg = 60.0;
  /// Largest gap between the two ranges interpolated at a clone's time [s]
  double max_gap = 0.1;
  /// Height above the plane below which no row is added: near the ground (take-off, landing) the
  /// features are close and small height errors are large relative to the height [m]
  double min_height = 5.0;
  /// Beam direction in the IMU frame (body-fixed; down for /imu_apps in FLU)
  Eigen::Vector3d dir_in_imu = Eigen::Vector3d(0, 0, -1);

  void print_and_load(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
    if (parser != nullptr) {
      parser->parse_config("use_range", enabled, false);
      parser->parse_config("range_topic", topic, false);
      parser->parse_config("range_sigma", sigma, false);
      parser->parse_config("range_huber_k", huber_k, false);
      parser->parse_config("range_min", min, false);
      parser->parse_config("range_max", max, false);
      parser->parse_config("range_max_tilt_deg", max_tilt_deg, false);
      parser->parse_config("range_max_gap", max_gap, false);
      parser->parse_config("range_min_height", min_height, false);
      std::vector<double> dir = {dir_in_imu(0), dir_in_imu(1), dir_in_imu(2)};
      parser->parse_config("range_dir_in_imu", dir, false);
      dir_in_imu << dir.at(0), dir.at(1), dir.at(2);
      dir_in_imu.normalize();
    }
    PRINT_INFO("RANGE PLANE: %s, topic %s, sigma %.2f m, valid (%.1f, %.1f] m, tilt <= %.0f deg, height >= %.1f m, beam [%.2f %.2f %.2f]\n",
               enabled ? "on" : "off", topic.c_str(), sigma, min, max, max_tilt_deg, min_height, dir_in_imu(0), dir_in_imu(1),
               dir_in_imu(2));
  }
};

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

private:
  RangeOptions _options;
  mutable std::mutex _mtx;
  std::deque<std::pair<double, double>> _data;
};

} // namespace ov_msckf

#endif // OV_MSCKF_RANGE_PLANE_H
