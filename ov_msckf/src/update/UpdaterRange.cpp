/*
 * Addition to OpenVINS (fork): the rangefinder plane constraint, shared by the MSCKF and SLAM updaters.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "UpdaterRange.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "UpdaterHelper.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/Landmark.h"
#include "types/PoseJPL.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

bool UpdaterRange::find_reading(std::shared_ptr<State> state, const std::unordered_map<size_t, std::vector<double>> &timestamps,
                                double &clone_time, double &range) const {
  std::vector<double> times;
  for (const auto &pair : timestamps) {
    times.insert(times.end(), pair.second.begin(), pair.second.end());
  }
  std::sort(times.rbegin(), times.rend());
  const double dt_cam_to_imu = state->_calib_dt_CAMtoIMU->value()(0);
  for (double t : times) {
    if (state->_clones_IMU.find(t) != state->_clones_IMU.end() && _buffer->get(t + dt_cam_to_imu, range) &&
        !reading_is_outlier(state, t, range)) {
      clone_time = t;
      return true;
    }
  }
  return false;
}

bool UpdaterRange::reading_is_outlier(std::shared_ptr<State> state, double clone_time, double range) const {
  if (_options.outlier_max_dev <= 0.0)
    return false;
  double z0 = 0.0;
  if (!plane_height(state, clone_time, range, z0))
    return false;  // not usable anyway (tilt, height)

  // Plane heights under the beam at the other clones: the ground height each reading implies, with the vehicle's
  // climb and attitude taken out
  const double dt_cam_to_imu = state->_calib_dt_CAMtoIMU->value()(0);
  std::vector<double> z;
  for (const auto &clone : state->_clones_IMU) {
    double r = 0.0, zc = 0.0;
    if (clone.first != clone_time && _buffer->get(clone.first + dt_cam_to_imu, r) && plane_height(state, clone.first, r, zc))
      z.push_back(zc);
  }
  if (z.size() < 4)
    return false;  // too few to judge
  std::nth_element(z.begin(), z.begin() + z.size() / 2, z.end());
  const bool outlier = std::abs(z0 - z[z.size() / 2]) > _options.outlier_max_dev;

  // Count each rejected reading once; forget those older than the window
  if (outlier && _rejected.insert(clone_time).second)
    _n_rejected++;
  const double oldest = state->_clones_IMU.begin()->first;
  while (!_rejected.empty() && *_rejected.begin() < oldest)
    _rejected.erase(_rejected.begin());
  return outlier;
}

bool UpdaterRange::plane_height(std::shared_ptr<State> state, double clone_time, double range, double &z_plane) const {
  std::shared_ptr<PoseJPL> clone = state->_clones_IMU.at(clone_time);
  const Eigen::Vector3d b = clone->Rot().transpose() * _options.dir_in_imu;
  if (-b(2) < std::cos(_options.max_tilt_deg * M_PI / 180.0) || range * -b(2) < _options.min_height)
    return false;
  const Eigen::Vector3d p_S = clone->pos() + clone->Rot().transpose() * _options.p_in_imu;
  z_plane = p_S(2) + range * b(2);
  return true;
}

int UpdaterRange::column_of(const std::vector<std::shared_ptr<Type>> &Hx_order, const std::shared_ptr<Type> &var) {
  int col = 0;
  for (const auto &v : Hx_order) {
    if (v.get() == var.get())
      return col;
    col += v->size();
  }
  return -1;
}

bool UpdaterRange::compute_row(std::shared_ptr<State> state, const std::unordered_map<size_t, std::vector<double>> &timestamps,
                               const Eigen::Vector3d &p_FinG, const std::vector<std::shared_ptr<Type>> &Hx_order, Row &row) {

  // The newest clone the feature is seen from that has a valid range (clones are at camera times; the
  // ranges are stamped in the IMU clock).
  double range = 0.0;
  double clone_time = -1.0;
  if (!find_reading(state, timestamps, clone_time, range)) {
    n_missing++;
    return false;
  }
  if (column_of(Hx_order, state->_clones_IMU.at(clone_time)) < 0)
    return false;
  return row_at(state, clone_time, range, p_FinG(2), row);
}

bool UpdaterRange::row_at(std::shared_ptr<State> state, double clone_time, double range, double z_plane, Row &row) const {
  std::shared_ptr<PoseJPL> clone = state->_clones_IMU.at(clone_time);

  // Beam in the global frame (z up): b = R_GtoI^T d, so b_z = d . (R_GtoI e_z). The beam starts at the
  // rangefinder, p_S = p_I + R_GtoI^T l (l: its position in the IMU frame), so its height above the
  // horizontal plane through the feature is H = p_I,z + l . (R_GtoI e_z) - p_F,z, and the predicted slant
  // range is H / -b_z.
  const Eigen::Vector3d &d = _options.dir_in_imu;
  const Eigen::Vector3d &l = _options.p_in_imu;
  const double b_z = d.dot(clone->Rot().col(2));
  const double height = clone->pos()(2) + l.dot(clone->Rot().col(2)) - z_plane;
  if (-b_z < std::cos(_options.max_tilt_deg * M_PI / 180.0) || height <= 0.0 || range * -b_z < _options.min_height) {
    return false;
  }
  row.residual = range - height / -b_z;

  // Jacobians at the linearisation point (first estimates, as the visual rows). With
  // R_GtoI = (I - [dtheta]x) R_hat, R_GtoI e_z = c + [c]x dtheta (c = R_hat e_z), so
  // d b_z / d dtheta = d^T [c]x and d H / d dtheta = l^T [c]x; with h = -H / b_z:
  // d h / d dtheta = -(1 / b_z) l^T [c]x + (H / b_z^2) d^T [c]x.
  const Eigen::Matrix3d R_lin = state->_options.do_fej ? clone->Rot_fej() : clone->Rot();
  const Eigen::Vector3d p_lin = state->_options.do_fej ? clone->pos_fej() : clone->pos();
  const Eigen::Matrix3d c_skew = skew_x(R_lin.col(2));
  const double b_z_lin = d.dot(R_lin.col(2));
  const double height_lin = p_lin(2) + l.dot(R_lin.col(2)) - z_plane;
  row.clone_time = clone_time;
  row.clone = clone;
  row.range = range;
  row.dh_dtheta = (height_lin / (b_z_lin * b_z_lin)) * d.transpose() * c_skew - (1.0 / b_z_lin) * l.transpose() * c_skew;
  row.dh_dpz = -1.0 / b_z_lin;
  row.dh_dpfz = 1.0 / b_z_lin;
  row.cos_tilt = -b_z;
  return true;
}

bool UpdaterRange::ground_ready(std::shared_ptr<State> state) const { return _options.plane_state && state->_ground != nullptr; }

bool UpdaterRange::ground_row(std::shared_ptr<State> state, const Eigen::Vector3d &p_FinG, Row &row) const {
  if (!ground_ready(state))
    return false;
  // Not near the ground (take-off, landing)
  const double d = state->_ground->value()(0);
  if (state->_imu->pos()(2) - d < _options.min_height)
    return false;
  row = Row();
  row.residual = d - p_FinG(2);  // measured 0 = p_F,z - d
  row.dh_dpfz = 1.0;
  row.ground = state->_ground;
  row.dh_dground = -1.0;
  row.cos_tilt = 1.0;
  return true;
}

void UpdaterRange::update_ground(std::shared_ptr<State> state, double clone_time, double rate) {
  if (!_options.plane_state || state->_clones_IMU.find(clone_time) == state->_clones_IMU.end())
    return;
  const double dt_cam_to_imu = state->_calib_dt_CAMtoIMU->value()(0);
  double range = 0.0, z_reading = 0.0;
  if (!_buffer->get(clone_time + dt_cam_to_imu, range) || !plane_height(state, clone_time, range, z_reading))
    return;
  std::shared_ptr<PoseJPL> clone = state->_clones_IMU.at(clone_time);

  // First reading: the ground is where it says, with the reading's noise and the clone's uncertainty
  if (state->_ground == nullptr) {
    Row row;
    if (!row_at(state, clone_time, range, z_reading, row))
      return;
    auto ground = std::make_shared<Vec>(1);
    ground->set_value(Eigen::VectorXd::Constant(1, z_reading));
    ground->set_fej(Eigen::VectorXd::Constant(1, z_reading));
    Eigen::MatrixXd H_R = Eigen::MatrixXd::Zero(1, 6), H_L(1, 1), R(1, 1);
    H_R.block(0, 0, 1, 3) = row.dh_dtheta;
    H_R(0, 5) = row.dh_dpz;
    H_L(0, 0) = row.dh_dpfz;  // the plane height enters as the feature's z did
    R(0, 0) = _options.reading_sigma * _options.reading_sigma;
    Eigen::VectorXd res = Eigen::VectorXd::Constant(1, row.residual);
    StateHelper::initialize_invertible(state, ground, {clone}, H_R, H_L, R, res);
    state->_ground = ground;
    _ground_last_time = clone_time;
    _ground_last_xy = clone->pos().head(2);
    PRINT_INFO("[GROUND] initialised at %.2f m (sigma %.2f m)\n", ground->value()(0),
               std::sqrt(StateHelper::get_marginal_covariance(state, {ground})(0, 0)));
    return;
  }

  // Terrain noise for the horizontal distance travelled since the last update
  std::shared_ptr<Vec> ground = state->_ground;
  const double dist = (clone->pos().head(2) - _ground_last_xy).norm();
  state->_Cov(ground->id(), ground->id()) += _options.terrain_sigma * _options.terrain_sigma * dist;
  _ground_last_time = clone_time;
  _ground_last_xy = clone->pos().head(2);

  // The reading against the ground height
  Row row;
  if (!row_at(state, clone_time, range, ground->value()(0), row))
    return;
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, 7);
  H.block(0, 0, 1, 3) = row.dh_dtheta;
  H(0, 5) = row.dh_dpz;
  H(0, 6) = row.dh_dpfz;  // w.r.t. the ground height
  std::vector<std::shared_ptr<Type>> H_order = {clone, ground};
  // Noise: the reading's, plus its timing error at the current angular rate (the beam tilts by w dt)
  double var = _options.reading_sigma * _options.reading_sigma;
  if (rate > 0.0 && _options.timing_sigma > 0.0) {
    const double s = row.dh_dtheta.norm() * rate * _options.timing_sigma;
    var += s * s;
  }
  // Gate: reject a reading that is both statistically (4 sigma of the innovation) and absolutely (outlier_max_dev)
  // far from the prediction; otherwise bound its influence (Huber)
  const Eigen::MatrixXd P = StateHelper::get_marginal_covariance(state, H_order);
  const double S = (H * P * H.transpose())(0, 0) + var;
  const double r = row.residual;
  if (r * r > 16.0 * S && (_options.outlier_max_dev <= 0.0 || std::abs(r) > _options.outlier_max_dev)) {
    _ground_rejected++;
    return;
  }
  if (std::abs(r) > _options.huber_k * std::sqrt(S))
    var += (r * r / (_options.huber_k * _options.huber_k)) - S;
  Eigen::MatrixXd R = Eigen::MatrixXd::Constant(1, 1, var);
  Eigen::VectorXd res = Eigen::VectorXd::Constant(1, r);
  StateHelper::EKFUpdate(state, H_order, H, res, R);
  _ground_used++;

  if (_ground_log_time < 0 || clone_time - _ground_log_time > 10.0) {
    _ground_log_time = clone_time;
    PRINT_INFO("[GROUND] height %.2f m (sigma %.2f m), vehicle %.1f m above; readings used %ld, rejected %ld\n", ground->value()(0),
               std::sqrt(StateHelper::get_marginal_covariance(state, {ground})(0, 0)), clone->pos()(2) - ground->value()(0), _ground_used,
               _ground_rejected);
  }
}

void UpdaterRange::append_row(const Row &row, int n_shared, double sigma_pix, const std::vector<std::shared_ptr<Type>> &Hx_order,
                              const Eigen::MatrixXd &dpfg_dlambda, const std::vector<Eigen::MatrixXd> &dpfg_dx,
                              const std::vector<std::shared_ptr<Type>> &dpfg_dx_order, Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x,
                              Eigen::VectorXd &res) {

  // Per-feature noise plus the shared part, counted once for all n_shared rows using this reading; then
  // whiten to the pixel noise the rest of the update assumes, with a Huber-type clamp beyond huber_k sigmas.
  double sigma = std::sqrt(_options.sigma * _options.sigma + n_shared * _options.sigma_shared * _options.sigma_shared);
  if (std::abs(row.residual) > _options.huber_k * sigma) {
    sigma = std::abs(row.residual) / _options.huber_k;
    _huber++;
  }
  const double w = sigma_pix / sigma;

  const Eigen::Index n = H_x.rows();
  H_x.conservativeResize(n + 1, Eigen::NoChange);
  H_x.row(n).setZero();
  H_f.conservativeResize(n + 1, Eigen::NoChange);
  H_f.row(n).setZero();
  res.conservativeResize(n + 1);

  // The clone the reading is at: orientation (3) then position (3) (PLANESTATE point-on-plane rows have none)
  if (row.clone != nullptr) {
    const int col = column_of(Hx_order, row.clone);
    assert(col >= 0);
    H_x.block(n, col, 1, 3) += w * row.dh_dtheta;
    H_x(n, col + 5) += w * row.dh_dpz;
  }
  // PLANESTATE: the ground height
  if (row.ground != nullptr) {
    const int col = column_of(Hx_order, row.ground);
    assert(col >= 0);
    H_x(n, col) += w * row.dh_dground;
  }

  // The feature: d h / d p_FinG = [0 0 dh_dpfz], chained through its representation
  H_f.row(n) = w * row.dh_dpfz * dpfg_dlambda.row(2);
  for (size_t k = 0; k < dpfg_dx_order.size(); k++) {
    const int c = column_of(Hx_order, dpfg_dx_order.at(k));
    assert(c >= 0);
    H_x.block(n, c, 1, dpfg_dx_order.at(k)->size()) += w * row.dh_dpfz * dpfg_dx.at(k).row(2);
  }
  res(n) = w * row.residual;

  // Statistics
  const double residual = row.residual;
  _rows++;
  _res_sum += residual;
  _res_sq += residual * residual;
  _shared_sum += n_shared;
  _shared_max = std::max(_shared_max, n_shared);
  // By tilt (<10, 10-20, 20-30, >=30 deg): a plane rotated wrongly would show as a residual that changes
  // systematically with the tilt.
  const int bin = std::min(3, (int)(std::acos(std::min(1.0, row.cos_tilt)) * 180.0 / M_PI / 10.0));
  _bin_sum[bin] += residual;
  _bin_n[bin]++;
  if (_rows % 5000 == 0) {
    PRINT_INFO("[RANGE] %ld rows: residual mean %+.2f m, rms %.2f m over the last 5000; rows per reading mean %.1f, max %d; "
               "Huber %.1f %%; no range %ld, outlier readings %ld; SLAM features from range %ld full, %ld on plane; "
               "mean by tilt <10/10-20/20-30/30+ deg: %+.2f %+.2f %+.2f %+.2f m (%ld %ld %ld %ld rows in all)\n",
               _rows, _res_sum / 5000.0, std::sqrt(_res_sq / 5000.0), _shared_sum / 5000.0, _shared_max, 100.0 * _huber / _rows, n_missing,
               _n_rejected, n_slam_full, n_slam_plane, _bin_sum[0] / std::max(1L, _bin_n[0]),
               _bin_sum[1] / std::max(1L, _bin_n[1]), _bin_sum[2] / std::max(1L, _bin_n[2]), _bin_sum[3] / std::max(1L, _bin_n[3]),
               _bin_n[0], _bin_n[1], _bin_n[2], _bin_n[3]);
    _res_sum = 0.0;
    _res_sq = 0.0;
    _shared_sum = 0;
    _shared_max = 0;
  }
}
