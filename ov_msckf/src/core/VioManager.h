/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_MSCKF_VIOMANAGER_H
#define OV_MSCKF_VIOMANAGER_H

#include <Eigen/StdVector>
#include <algorithm>
#include <atomic>
#include <boost/filesystem.hpp>
#include <fstream>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>

#include "VioManagerOptions.h"

namespace ov_core {
struct ImuData;
struct CameraData;
class TrackBase;
class FeatureInitializer;
class Feature;
} // namespace ov_core
namespace ov_init {
class InertialInitializer;
} // namespace ov_init

namespace ov_msckf {

class State;
class StateHelper;
class UpdaterMSCKF;
class UpdaterSLAM;
class UpdaterRange;
class UpdaterZeroVelocity;
class Propagator;

/**
 * @brief Core class that manages the entire system
 *
 * This class contains the state and other algorithms needed for the MSCKF to work.
 * We feed in measurements into this class and send them to their respective algorithms.
 * If we have measurements to propagate or update with, this class will call on our state to do that.
 */
class VioManager {

public:
  /**
   * @brief Default constructor, will load all configuration variables
   * @param params_ Parameters loaded from either ROS or CMDLINE
   */
  VioManager(VioManagerOptions &params_);

  /**
   * @brief Feed function for inertial data
   * @param message Contains our timestamp and inertial information
   */
  void feed_measurement_imu(const ov_core::ImuData &message);

  /**
   * @brief Feed function for camera measurements
   * @param message Contains our timestamp, images, and camera ids
   */
  void feed_measurement_camera(const ov_core::CameraData &message) { track_image_and_update(message); }

  /**
   * @brief Fork: feed function for a downward rangefinder (used by the plane constraint if enabled)
   * @param timestamp Time of the reading, in the range sensor's clock (time_offset in relative_config_range converts it to the IMU's)
   * @param range Slant range along the beam [m]; invalid readings are kept as gaps
   */
  void feed_measurement_range(double timestamp, double range) {
    if (range_buffer != nullptr)
      range_buffer->feed(timestamp + params.range_options.time_offset, range);
  }

  /**
   * @brief Feed function for a synchronized simulated cameras
   * @param timestamp Time that this image was collected
   * @param camids Camera ids that we have simulated measurements for
   * @param feats Raw uv simulated measurements
   */
  void feed_measurement_simulation(double timestamp, const std::vector<int> &camids,
                                   const std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> &feats);

  /**
   * @brief Given a state, this will initialize our IMU state.
   * @param imustate State in the MSCKF ordering: [time(sec),q_GtoI,p_IinG,v_IinG,b_gyro,b_accel]
   */
  void initialize_with_gt(Eigen::Matrix<double, 17, 1> imustate);

  /// If we are initialized or not
  bool initialized() { return is_initialized_vio && timelastupdate != -1; }

  /// Timestamp that the system was initialized at
  double initialized_time() { return startup_time; }

  /// Accessor for current system parameters
  VioManagerOptions get_params() { return params; }

  /// Accessor to get the current state
  std::shared_ptr<State> get_state() { return state; }

  /// Accessor to get the current propagator
  std::shared_ptr<Propagator> get_propagator() { return propagator; }

  /// Get a nice visualization image of what tracks we have
  cv::Mat get_historical_viz_image();

  /// Returns 3d SLAM features in the global frame
  std::vector<Eigen::Vector3d> get_features_SLAM();

  /// Returns 3d ARUCO features in the global frame
  std::vector<Eigen::Vector3d> get_features_ARUCO();

  /// Returns 3d features used in the last update in global frame
  std::vector<Eigen::Vector3d> get_good_features_MSCKF() { return good_features_MSCKF; }

  /// Return the image used when projecting the active tracks
  void get_active_image(double &timestamp, cv::Mat &image) {
    timestamp = active_tracks_time;
    image = active_image;
  }

  /// Returns active tracked features in the current frame
  void get_active_tracks(double &timestamp, std::unordered_map<size_t, Eigen::Vector3d> &feat_posinG,
                         std::unordered_map<size_t, Eigen::Vector3d> &feat_tracks_uvd) {
    timestamp = active_tracks_time;
    feat_posinG = active_tracks_posinG;
    feat_tracks_uvd = active_tracks_uvd;
  }

protected:
  /**
   * @brief Given a new set of camera images, this will track them.
   *
   * If we are having stereo tracking, we should call stereo tracking functions.
   * Otherwise we will try to track on each of the images passed.
   *
   * @param message Contains our timestamp, images, and camera ids
   */
  void track_image_and_update(const ov_core::CameraData &message);

  /**
   * @brief This will do the propagation and feature updates to the state
   * @param message Contains our timestamp, images, and camera ids
   */
  void do_feature_propagate_update(const ov_core::CameraData &message);

  /**
   * @brief This function will try to initialize the state.
   *
   * This should call on our initializer and try to init the state.
   * In the future we should call the structure-from-motion code from here.
   * This function could also be repurposed to re-initialize the system after failure.
   *
   * @param message Contains our timestamp, images, and camera ids
   * @return True if we have successfully initialized
   */
  bool try_to_initialize(const ov_core::CameraData &message);

  /**
   * @brief This function will will re-triangulate all features in the current frame
   *
   * For all features that are currently being tracked by the system, this will re-triangulate them.
   * This is useful for downstream applications which need the current pointcloud of points (e.g. loop closure).
   * This will try to triangulate *all* points, not just ones that have been used in the update.
   *
   * @param message Contains our timestamp, images, and camera ids
   */
  void retriangulate_active_tracks(const ov_core::CameraData &message);

  /// Manager parameters
  VioManagerOptions params;

  /// Our master state object :D
  std::shared_ptr<State> state;

  /// Propagator of our state
  std::shared_ptr<Propagator> propagator;

  /// Our sparse feature tracker (klt or descriptor)
  std::shared_ptr<ov_core::TrackBase> trackFEATS;

  /// Our aruoc tracker
  std::shared_ptr<ov_core::TrackBase> trackARUCO;

  /// State initializer
  std::shared_ptr<ov_init::InertialInitializer> initializer;

  /// Boolean if we are initialized or not
  bool is_initialized_vio = false;

  /// Our MSCKF feature updater
  std::shared_ptr<UpdaterMSCKF> updaterMSCKF;

  /// Fork: rangefinder readings for the plane constraint (null if it is off)
  std::shared_ptr<RangeBuffer> range_buffer;
  /// Fork: the range constraint shared by the MSCKF and SLAM updaters (null if it is off)
  std::shared_ptr<UpdaterRange> range_updater;
  /// State time (IMU clock) of the last "no range readings" warning
  double range_last_warn_time = -1;
  /// Fork: tracked frames since the last keyframe (keyframe_every_n_frames); the first one after start is a keyframe
  int frames_since_keyframe = 1 << 20;
  /// Fork: whether the frame being processed is a keyframe
  bool frame_is_keyframe = true;
  /// Fork: times of the clones that are keyframes (with num_recent_clones)
  std::set<double> keyframe_times;

  /// Fork: whether this tracked frame is a keyframe (counts it)
  bool is_keyframe();

  /// Fork: keyframes plus the newest frames as clones (keyframe_every_n_frames > 1 and num_recent_clones > 0)
  bool use_recent_clones() const { return params.keyframe_every_n_frames > 1 && params.num_recent_clones > 0; }

  /**
   * @brief Fork: the clone to marginalize after this update, with use_recent_clones: the newest non-keyframe
   * older than the num_recent_clones newest clones, else the oldest clone if the window is over max_clones
   * @param time Time of the clone, -1 for none
   * @param oldest Whether it is the oldest clone (its features are then used in the update, as upstream)
   */
  void choose_marg_clone(double &time, bool &oldest);

  /// State time of the last SLAM feature height log
  double range_last_slam_log = -1;

  /// Fork: triangulates candidate tracks for slam_min_parallax (with the triangulation settings, fi_*)
  std::shared_ptr<ov_core::FeatureInitializer> promote_init;

  /// Fork: ids of the SLAM features chosen in this update that were promoted early (slam_min_parallax); they are
  /// created on the plane from the range
  std::set<size_t> slam_on_plane;

  /// RATE (experimental): angular rate [rad/s] by clone time, shared with the updaters (null: off), and the
  /// last second of gyro readings it is computed from
  std::shared_ptr<std::map<double, double>> clone_rates;
  std::deque<std::pair<double, Eigen::Vector3d>> gyro_hist;
  std::mutex gyro_mtx;

  /// Fork: SLAM feature statistics for the [SLAM] line (print_track_stats): creation time of each SLAM feature,
  /// and per log interval the features created (with their image row, 0 top to 1 bottom) and lost (with their life)
  std::map<size_t, double> slam_created;
  double slam_log_last = -1, slam_row_sum = 0.0, slam_life_sum = 0.0;
  long slam_n_new = 0, slam_n_lost = 0;

  /// Our SLAM/ARUCO feature updater
  std::shared_ptr<UpdaterSLAM> updaterSLAM;

  /// Our zero velocity tracker
  std::shared_ptr<UpdaterZeroVelocity> updaterZUPT;

  /// This is the queue of measurement times that have come in since we starting doing initialization
  /// After we initialize, we will want to prop & update to the latest timestamp quickly
  std::vector<double> camera_queue_init;
  std::mutex camera_queue_init_mtx;

  // Timing statistic file and variables
  std::ofstream of_statistics;
  boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7;

  // Track how much distance we have traveled
  double timelastupdate = -1;
  double distance = 0;

  // Startup time of the filter
  double startup_time = -1;

  // Threads and their atomics
  std::atomic<bool> thread_init_running, thread_init_success;

  // If we did a zero velocity update
  bool did_zupt_update = false;
  bool has_moved_since_zupt = false;

  // Good features that where used in the last update (used in visualization)
  std::vector<Eigen::Vector3d> good_features_MSCKF;

  // Re-triangulated features 3d positions seen from the current frame (used in visualization)
  // For each feature we have a linear system A * p_FinG = b we create and increment their costs
  double active_tracks_time = -1;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_posinG;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_uvd;
  cv::Mat active_image;
  std::map<size_t, Eigen::Matrix3d> active_feat_linsys_A;
  std::map<size_t, Eigen::Vector3d> active_feat_linsys_b;
  std::map<size_t, int> active_feat_linsys_count;
};

} // namespace ov_msckf

#endif // OV_MSCKF_VIOMANAGER_H
