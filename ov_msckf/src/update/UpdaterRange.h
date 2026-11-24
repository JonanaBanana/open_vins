/*
 * GeoTNav addition to OpenVINS: the rangefinder plane constraint, shared by the MSCKF and SLAM updaters.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_MSCKF_UPDATER_RANGE_H
#define OV_MSCKF_UPDATER_RANGE_H

#include <Eigen/Eigen>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "RangePlane.h"

namespace ov_type {
class Type;
} // namespace ov_type

namespace ov_msckf {

class State;

/**
 * @brief The rangefinder plane constraint: one row per feature relating the measured slant range to the
 * clone's height above the horizontal plane through the feature.
 *
 * The ground is a plane perpendicular to gravity. A reading at a clone (camera time t, IMU time t + dt) gives
 * the plane's height under the beam; a feature on that plane predicts the slant range
 * h = (p_I,z + l . (R_GtoI e_z) - p_F,z) / -b_z (l: the rangefinder's position in the IMU frame, b_z: the beam's
 * vertical component in the global frame). See RangeOptions for the settings.
 */
class UpdaterRange {
public:
  UpdaterRange(std::shared_ptr<RangeBuffer> buffer, const RangeOptions &options) : _buffer(buffer), _options(options) {}

  /// A feature's range row before whitening
  struct Row {
    double clone_time = -1.0;      // clone (camera clock) whose reading is used: rows sharing it share the reading
    std::shared_ptr<ov_type::Type> clone;  // that clone
    double range = 0.0;            // the reading [m]
    double residual = 0.0;         // measured minus predicted slant range [m]
    Eigen::RowVector3d dh_dtheta;  // w.r.t. the clone's orientation error
    double dh_dpz = 0.0;           // w.r.t. the clone's z position
    double dh_dpfz = 0.0;          // w.r.t. the feature's global z position
    double cos_tilt = 1.0;         // beam's tilt from vertical
  };

  const RangeOptions &options() const { return _options; }

  /// The newest clone among the feature's measurement times with a valid reading that is not an outlier; false if none
  bool find_reading(std::shared_ptr<State> state, const std::unordered_map<size_t, std::vector<double>> &timestamps, double &clone_time,
                    double &range) const;

  /**
   * @brief Height of the plane under the beam at a clone, z = p_S,z + range b_z; false if the beam is tilted
   * beyond max_tilt_deg or the measured height is below min_height
   */
  bool plane_height(std::shared_ptr<State> state, double clone_time, double range, double &z_plane) const;

  /**
   * @brief Whether the reading at a clone is an outlier (outlier_max_dev): its plane height differs by more than
   * that from the median plane height of the other clones' readings. Needs at least 4 of them; false if off.
   */
  bool reading_is_outlier(std::shared_ptr<State> state, double clone_time, double range) const;

  /// Largest distance between the clones the feature is seen from, over its distance from the newest one
  double parallax(std::shared_ptr<State> state, const std::unordered_map<size_t, std::vector<double>> &timestamps,
                  const Eigen::Vector3d &p_FinG) const;

  /**
   * @brief The range row of a feature at p_FinG (global), seen at the given times.
   *
   * False (no row) if there is no valid reading, the clone is not in Hx_order, the beam is too tilted or too low,
   * or, with gates, the feature's parallax is below min_parallax.
   */
  bool compute_row(std::shared_ptr<State> state, const std::unordered_map<size_t, std::vector<double>> &timestamps,
                   const Eigen::Vector3d &p_FinG, const std::vector<std::shared_ptr<ov_type::Type>> &Hx_order, bool gates, Row &row);

  /**
   * @brief Append a range row to a feature's system, whitened to sigma_pix, with standard deviation
   * sqrt(sigma^2 + n_shared sigma_shared^2) and a Huber-type clamp beyond huber_k of it.
   *
   * dpfg_dlambda (3 x H_f.cols()) is the derivative of the feature's global position w.r.t. its parameters
   * (identity for a global 3D position); dpfg_dx / dpfg_dx_order are its derivatives w.r.t. state variables
   * (e.g. an anchor pose), which must be in Hx_order.
   */
  void append_row(const Row &row, int n_shared, double sigma_pix, const std::vector<std::shared_ptr<ov_type::Type>> &Hx_order,
                  const Eigen::MatrixXd &dpfg_dlambda, const std::vector<Eigen::MatrixXd> &dpfg_dx,
                  const std::vector<std::shared_ptr<ov_type::Type>> &dpfg_dx_order, Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x,
                  Eigen::VectorXd &res);

  /// Counters for the periodic log
  long n_missing = 0;        // no valid reading for the feature
  long n_low_parallax = 0;   // rows skipped by min_parallax
  long n_slam_full = 0;      // SLAM features created from all their observations plus the range row
  long n_slam_plane = 0;     // SLAM features created on the plane from one bearing and the range

private:
  /// Column of a variable in a Jacobian ordered as Hx_order; -1 if absent
  static int column_of(const std::vector<std::shared_ptr<ov_type::Type>> &Hx_order, const std::shared_ptr<ov_type::Type> &var);

  std::shared_ptr<RangeBuffer> _buffer;
  RangeOptions _options;

  /// Clone times whose reading was rejected as an outlier (each counted once; pruned with the window)
  mutable std::set<double> _rejected;
  mutable long _n_rejected = 0;

  // Statistics of the rows, logged every 5000 rows
  long _rows = 0;
  long _huber = 0;
  double _res_sum = 0.0;
  double _res_sq = 0.0;
  long _shared_sum = 0;
  int _shared_max = 0;
  double _bin_sum[4] = {0, 0, 0, 0};
  long _bin_n[4] = {0, 0, 0, 0};
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_RANGE_H
