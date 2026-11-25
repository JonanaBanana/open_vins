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

#include "VioManager.h"

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "feat/FeatureInitializer.h"
#include "track/TrackAruco.h"
#include "track/TrackDescriptor.h"
#include "track/TrackKLT.h"
#include "track/TrackSIM.h"
#include "types/Landmark.h"
#include "types/LandmarkRepresentation.h"
#include "utils/opencv_lambda_body.h"
#include "utils/print.h"
#include "utils/quat_ops.h"
#include "utils/sensor_data.h"

#include "init/InertialInitializer.h"

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterMSCKF.h"
#include "update/UpdaterSLAM.h"
#include "update/UpdaterZeroVelocity.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

VioManager::VioManager(VioManagerOptions &params_) : thread_init_running(false), thread_init_success(false) {

  // Nice startup message
  PRINT_DEBUG("=======================================\n");
  PRINT_DEBUG("OPENVINS ON-MANIFOLD EKF IS STARTING\n");
  PRINT_DEBUG("=======================================\n");

  // Nice debug
  this->params = params_;
  params.print_and_load_estimator();
  params.print_and_load_noise();
  params.print_and_load_state();
  params.print_and_load_trackers();

  // This will globally set the thread count we will use
  // -1 will reset to the system default threading (usually the num of cores)
  cv::setNumThreads(params.num_opencv_threads);
  cv::setRNGSeed(0);

  // Create the state!!
  state = std::make_shared<State>(params.state_options);

  // Set the IMU intrinsics
  state->_calib_imu_dw->set_value(params.vec_dw);
  state->_calib_imu_dw->set_fej(params.vec_dw);
  state->_calib_imu_da->set_value(params.vec_da);
  state->_calib_imu_da->set_fej(params.vec_da);
  state->_calib_imu_tg->set_value(params.vec_tg);
  state->_calib_imu_tg->set_fej(params.vec_tg);
  state->_calib_imu_GYROtoIMU->set_value(params.q_GYROtoIMU);
  state->_calib_imu_GYROtoIMU->set_fej(params.q_GYROtoIMU);
  state->_calib_imu_ACCtoIMU->set_value(params.q_ACCtoIMU);
  state->_calib_imu_ACCtoIMU->set_fej(params.q_ACCtoIMU);

  // Timeoffset from camera to IMU
  Eigen::VectorXd temp_camimu_dt;
  temp_camimu_dt.resize(1);
  temp_camimu_dt(0) = params.calib_camimu_dt;
  state->_calib_dt_CAMtoIMU->set_value(temp_camimu_dt);
  state->_calib_dt_CAMtoIMU->set_fej(temp_camimu_dt);

  // Loop through and load each of the cameras
  state->_cam_intrinsics_cameras = params.camera_intrinsics;
  for (int i = 0; i < state->_options.num_cameras; i++) {
    state->_cam_intrinsics.at(i)->set_value(params.camera_intrinsics.at(i)->get_value());
    state->_cam_intrinsics.at(i)->set_fej(params.camera_intrinsics.at(i)->get_value());
    state->_calib_IMUtoCAM.at(i)->set_value(params.camera_extrinsics.at(i));
    state->_calib_IMUtoCAM.at(i)->set_fej(params.camera_extrinsics.at(i));
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // If we are recording statistics, then open our file
  if (params.record_timing_information) {
    // If the file exists, then delete it
    if (boost::filesystem::exists(params.record_timing_filepath)) {
      boost::filesystem::remove(params.record_timing_filepath);
      PRINT_INFO(YELLOW "[STATS]: found old file found, deleted...\n" RESET);
    }
    // Create the directory that we will open the file in
    boost::filesystem::path p(params.record_timing_filepath);
    boost::filesystem::create_directories(p.parent_path());
    // Open our statistics file!
    of_statistics.open(params.record_timing_filepath, std::ofstream::out | std::ofstream::app);
    // Write the header information into it
    of_statistics << "# timestamp (sec),tracking,propagation,msckf update,";
    if (state->_options.max_slam_features > 0) {
      of_statistics << "slam update,slam delayed,";
    }
    of_statistics << "re-tri & marg,total" << std::endl;
  }

  //===================================================================================
  //===================================================================================
  //===================================================================================

  // Let's make a feature extractor
  // NOTE: after we initialize we will increase the total number of feature tracks
  // NOTE: we will split the total number of features over all cameras uniformly
  int init_max_features = std::floor((double)params.init_options.init_max_features / (double)params.state_options.num_cameras);
  if (params.use_klt) {
    trackFEATS = std::shared_ptr<TrackBase>(new TrackKLT(state->_cam_intrinsics_cameras, init_max_features,
                                                         state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
                                                         params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist));
  } else {
    trackFEATS = std::shared_ptr<TrackBase>(new TrackDescriptor(
        state->_cam_intrinsics_cameras, init_max_features, state->_options.max_aruco_features, params.use_stereo, params.histogram_method,
        params.fast_threshold, params.grid_x, params.grid_y, params.min_px_dist, params.knn_ratio));
  }

  // Initialize our aruco tag extractor
  if (params.use_aruco) {
    trackARUCO = std::shared_ptr<TrackBase>(new TrackAruco(state->_cam_intrinsics_cameras, state->_options.max_aruco_features,
                                                           params.use_stereo, params.histogram_method, params.downsize_aruco));
  }

  // Initialize our state propagator
  propagator = std::make_shared<Propagator>(params.imu_noises, params.gravity_mag);

  // Our state initialize
  initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS->get_feature_database());

  // Make the updater!
  updaterMSCKF = std::make_shared<UpdaterMSCKF>(params.msckf_options, params.featinit_options);
  if (params.range_options.enabled) {
    // The MSCKF range row is written for global 3D features (UpdaterMSCKF); with any other representation it
    // would never be added.
    if (params.state_options.feat_rep_msckf != LandmarkRepresentation::Representation::GLOBAL_3D) {
      PRINT_ERROR(RED "use_range requires feat_rep_msckf: GLOBAL_3D (got %s)\n" RESET,
                  LandmarkRepresentation::as_string(params.state_options.feat_rep_msckf).c_str());
      std::exit(EXIT_FAILURE);
    }
    // SLAM features from the range need a 3D position; the single inverse depth is initialised as a depth only
    // (UpdaterSLAM::delayed_init then creates them as upstream does)
    if (params.range_options.slam_init &&
        params.state_options.feat_rep_slam == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
      PRINT_WARNING(YELLOW "[RANGE] range_slam_init has no effect with feat_rep_slam: ANCHORED_INVERSE_DEPTH_SINGLE; SLAM "
                           "features are created by triangulation only\n" RESET);
    }
    // Sanity check of the extrinsics: the plane model only needs the features to lie on the ground the beam
    // ranges to, not the beam to point where a camera looks. But a camera looking away from the beam (e.g.
    // up or forward while the beam points down) sees mostly points off that plane.
    for (const auto &cam : params.camera_extrinsics) {
      const Eigen::Matrix3d R_ItoC = quat_2_Rot(cam.second.block(0, 0, 4, 1));
      const Eigen::Vector3d axis_in_imu = R_ItoC.row(2).transpose();
      const double angle_deg = std::acos(std::max(-1.0, std::min(1.0, axis_in_imu.dot(params.range_options.dir_in_imu)))) * 180.0 / M_PI;
      if (angle_deg > 90.0) {
        PRINT_WARNING(YELLOW "[RANGE] cam%zu looks %.0f deg away from the rangefinder beam: the ground plane model assumes the "
                             "camera sees the ground the beam ranges to; check T_imu_cam and T_imu_range\n" RESET,
                      cam.first, angle_deg);
      } else {
        PRINT_INFO("[RANGE] cam%zu optical axis is %.0f deg from the rangefinder beam\n", cam.first, angle_deg);
      }
    }
    range_buffer = std::make_shared<RangeBuffer>(params.range_options);
    range_updater = std::make_shared<UpdaterRange>(range_buffer, params.range_options);
    updaterMSCKF->set_range(range_updater);
  }
  updaterSLAM = std::make_shared<UpdaterSLAM>(params.slam_options, params.aruco_options, params.featinit_options);
  promote_init = std::make_shared<FeatureInitializer>(params.featinit_options);
  if (range_updater != nullptr && params.range_options.slam_init) {
    updaterSLAM->set_range(range_updater);
  }

  // If we are using zero velocity updates, then create the updater
  if (params.try_zupt) {
    updaterZUPT = std::make_shared<UpdaterZeroVelocity>(params.zupt_options, params.imu_noises, trackFEATS->get_feature_database(),
                                                        propagator, params.gravity_mag, params.zupt_max_velocity,
                                                        params.zupt_noise_multiplier, params.zupt_max_disparity);
  }
}

void VioManager::feed_measurement_imu(const ov_core::ImuData &message) {

  // The oldest time we need IMU with is the last clone
  // We shouldn't really need the whole window, but if we go backwards in time we will
  double oldest_time = state->margtimestep();
  if (oldest_time > state->_timestamp) {
    oldest_time = -1;
  }
  if (!is_initialized_vio) {
    oldest_time = message.timestamp - params.init_options.init_window_time + state->_calib_dt_CAMtoIMU->value()(0) - 0.10;
  }
  propagator->feed_imu(message, oldest_time);

  // Push back to our initializer
  if (!is_initialized_vio) {
    initializer->feed_imu(message, oldest_time);
  }

  // Push back to the zero velocity updater if it is enabled
  // No need to push back if we are just doing the zv-update at the begining and we have moved
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    updaterZUPT->feed_imu(message, oldest_time);
  }
}

void VioManager::feed_measurement_simulation(double timestamp, const std::vector<int> &camids,
                                             const std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> &feats) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Check if we actually have a simulated tracker
  // If not, recreate and re-cast the tracker to our simulation tracker
  std::shared_ptr<TrackSIM> trackSIM = std::dynamic_pointer_cast<TrackSIM>(trackFEATS);
  if (trackSIM == nullptr) {
    // Replace with the simulated tracker
    trackSIM = std::make_shared<TrackSIM>(state->_cam_intrinsics_cameras, state->_options.max_aruco_features);
    trackFEATS = trackSIM;
    // Need to also replace it in init and zv-upt since it points to the trackFEATS db pointer
    initializer = std::make_shared<ov_init::InertialInitializer>(params.init_options, trackFEATS->get_feature_database());
    if (params.try_zupt) {
      updaterZUPT = std::make_shared<UpdaterZeroVelocity>(params.zupt_options, params.imu_noises, trackFEATS->get_feature_database(),
                                                          propagator, params.gravity_mag, params.zupt_max_velocity,
                                                          params.zupt_noise_multiplier, params.zupt_max_disparity);
    }
    PRINT_WARNING(RED "[SIM]: casting our tracker to a TrackSIM object!\n" RESET);
  }

  // Feed our simulation tracker
  trackSIM->feed_measurement_simulation(timestamp, camids, feats);
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Check if we should do zero-velocity, if so update the state with it
  // Note that in the case that we only use in the beginning initialization phase
  // If we have since moved, then we should never try to do a zero velocity update!
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    // If the same state time, use the previous timestep decision
    if (state->_timestamp != timestamp) {
      did_zupt_update = updaterZUPT->try_update(state, timestamp);
    }
    if (did_zupt_update) {
      assert(state->_timestamp == timestamp);
      propagator->clean_old_imu_measurements(timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      updaterZUPT->clean_old_imu_measurements(timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      propagator->invalidate_cache();
      return;
    }
  }

  // If we do not have VIO initialization, then return an error
  if (!is_initialized_vio) {
    PRINT_ERROR(RED "[SIM]: your vio system should already be initialized before simulating features!!!\n" RESET);
    PRINT_ERROR(RED "[SIM]: initialize your system first before calling feed_measurement_simulation()!!!!\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  // Fork: with keyframes and no recent clones, only keyframes are propagated, cloned and used to update
  frame_is_keyframe = is_keyframe();
  if (!frame_is_keyframe && !use_recent_clones())
    return;

  // Call on our propagate and update function
  // Simulation is either all sync, or single camera...
  ov_core::CameraData message;
  message.timestamp = timestamp;
  for (auto const &camid : camids) {
    int width = state->_cam_intrinsics_cameras.at(camid)->w();
    int height = state->_cam_intrinsics_cameras.at(camid)->h();
    message.sensor_ids.push_back(camid);
    message.images.push_back(cv::Mat::zeros(cv::Size(width, height), CV_8UC1));
    message.masks.push_back(cv::Mat::zeros(cv::Size(width, height), CV_8UC1));
  }
  do_feature_propagate_update(message);
}

void VioManager::track_image_and_update(const ov_core::CameraData &message_const) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Assert we have valid measurement data and ids
  assert(!message_const.sensor_ids.empty());
  assert(message_const.sensor_ids.size() == message_const.images.size());
  for (size_t i = 0; i < message_const.sensor_ids.size() - 1; i++) {
    assert(message_const.sensor_ids.at(i) != message_const.sensor_ids.at(i + 1));
  }

  // Downsample if we are downsampling
  ov_core::CameraData message = message_const;
  for (size_t i = 0; i < message.sensor_ids.size() && params.downsample_cameras; i++) {
    cv::Mat img = message.images.at(i);
    cv::Mat mask = message.masks.at(i);
    cv::Mat img_temp, mask_temp;
    cv::pyrDown(img, img_temp, cv::Size(img.cols / 2.0, img.rows / 2.0));
    message.images.at(i) = img_temp;
    cv::pyrDown(mask, mask_temp, cv::Size(mask.cols / 2.0, mask.rows / 2.0));
    message.masks.at(i) = mask_temp;
  }

  // Perform our feature tracking!
  trackFEATS->feed_new_camera(message);

  // If the aruco tracker is available, the also pass to it
  // NOTE: binocular tracking for aruco doesn't make sense as we by default have the ids
  // NOTE: thus we just call the stereo tracking if we are doing binocular!
  if (is_initialized_vio && trackARUCO != nullptr) {
    trackARUCO->feed_new_camera(message);
  }
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Check if we should do zero-velocity, if so update the state with it
  // Note that in the case that we only use in the beginning initialization phase
  // If we have since moved, then we should never try to do a zero velocity update!
  if (is_initialized_vio && updaterZUPT != nullptr && (!params.zupt_only_at_beginning || !has_moved_since_zupt)) {
    // If the same state time, use the previous timestep decision
    if (state->_timestamp != message.timestamp) {
      did_zupt_update = updaterZUPT->try_update(state, message.timestamp);
    }
    if (did_zupt_update) {
      assert(state->_timestamp == message.timestamp);
      propagator->clean_old_imu_measurements(message.timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      updaterZUPT->clean_old_imu_measurements(message.timestamp + state->_calib_dt_CAMtoIMU->value()(0) - 0.10);
      propagator->invalidate_cache();
      return;
    }
  }

  // If we do not have VIO initialization, then try to initialize
  // TODO: Or if we are trying to reset the system, then do that here!
  if (!is_initialized_vio) {
    is_initialized_vio = try_to_initialize(message);
    if (!is_initialized_vio) {
      double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
      PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
      return;
    }
    frames_since_keyframe = params.keyframe_every_n_frames;  // the initialisation frame is a keyframe
  }

  // Fork: with keyframes and no recent clones, only keyframes are propagated, cloned and used to update (the
  // others were only tracked)
  frame_is_keyframe = is_keyframe();
  if (!frame_is_keyframe && !use_recent_clones())
    return;

  // Call on our propagate and update function
  do_feature_propagate_update(message);
}

namespace {
/// Fork: the parallax of a track as upstream measures it (FeatureInitializer::single_gaussnewton, fi_max_baseline):
/// the largest baseline between the anchor and the clones it is seen at, perpendicular to the ray to the feature,
/// over the feature's distance. Triangulates a copy (observations at clones only; the track is not changed); -1 if it
/// is seen at fewer than 3 clones or the triangulation fails.
double track_parallax(const std::shared_ptr<Feature> &feat, const std::vector<double> &clone_times, FeatureInitializer &init,
                      std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> &clones_cam) {
  auto f = std::make_shared<Feature>(*feat);
  f->clean_old_measurements(clone_times);
  size_t n = 0;
  for (const auto &cam : f->timestamps)
    n += cam.second.size();
  if (n < 3 || !init.single_triangulation(f, clones_cam))
    return -1.0;
  const double dist = f->p_FinA.norm();
  if (!(dist > 1e-6))
    return -1.0;
  FeatureInitializer::ClonePose &anchor = clones_cam.at(f->anchor_cam_id).at(f->anchor_clone_timestamp);
  const Eigen::Vector3d ray = f->p_FinA / dist;
  double base_max = 0.0;
  for (const auto &cam : f->timestamps) {
    for (double t : cam.second) {
      const Eigen::Vector3d p_CiinA = anchor.Rot() * (clones_cam.at(cam.first).at(t).pos() - anchor.pos());
      base_max = std::max(base_max, (p_CiinA - p_CiinA.dot(ray) * ray).norm());
    }
  }
  return base_max / dist;
}
} // namespace

bool VioManager::is_keyframe() {
  if (++frames_since_keyframe < params.keyframe_every_n_frames)
    return false;
  frames_since_keyframe = 0;
  return true;
}

void VioManager::choose_marg_clone(double &time, bool &oldest) {
  time = -1;
  oldest = false;
  if (state->_clones_IMU.empty())
    return;
  // Clones older than the recent part are kept only at keyframes: the newest non-keyframe there is the one leaving
  // the recent part now (or one left from the start-up, before updates began)
  if ((int)state->_clones_IMU.size() > params.num_recent_clones) {
    auto it = state->_clones_IMU.rbegin();
    std::advance(it, params.num_recent_clones);
    for (; it != state->_clones_IMU.rend(); ++it) {
      if (keyframe_times.find(it->first) == keyframe_times.end()) {
        time = it->first;
        break;
      }
    }
  }
  // Otherwise the window is all keyframes: drop the oldest when it is full, as upstream
  if (time < 0 && (int)state->_clones_IMU.size() > state->_options.max_clone_size)
    time = state->_clones_IMU.begin()->first;
  oldest = time >= 0 && time == state->_clones_IMU.begin()->first;
}

void VioManager::do_feature_propagate_update(const ov_core::CameraData &message) {

  //===================================================================================
  // State propagation, and clone augmentation
  //===================================================================================

  // Return if the camera measurement is out of order
  if (state->_timestamp > message.timestamp) {
    PRINT_WARNING(YELLOW "image received out of order, unable to do anything (prop dt = %3f)\n" RESET,
                  (message.timestamp - state->_timestamp));
    return;
  }

  // Propagate the state forward to the current update time
  // Also augment it with a new clone!
  // NOTE: if the state is already at the given time (can happen in sim)
  // NOTE: then no need to prop since we already are at the desired timestep
  if (state->_timestamp != message.timestamp) {
    propagator->propagate_and_clone(state, message.timestamp);
  }
  if (use_recent_clones() && frame_is_keyframe && state->_clones_IMU.find(message.timestamp) != state->_clones_IMU.end())
    keyframe_times.insert(message.timestamp);
  rT3 = boost::posix_time::microsec_clock::local_time();

  // If we have not reached max clones, we should just return...
  // This isn't super ideal, but it keeps the logic after this easier...
  // We can start processing things when we have at least 5 clones since we can start triangulating things...
  if ((int)state->_clones_IMU.size() < std::min(state->_options.max_clone_size, 5)) {
    PRINT_DEBUG("waiting for enough clone states (%d of %d)....\n", (int)state->_clones_IMU.size(),
                std::min(state->_options.max_clone_size, 5));
    return;
  }

  // Return if we where unable to propagate
  if (state->_timestamp != message.timestamp) {
    PRINT_WARNING(RED "[PROP]: Propagator unable to propagate the state forward in time!\n" RESET);
    PRINT_WARNING(RED "[PROP]: It has been %.3f since last time we propagated\n" RESET, message.timestamp - state->_timestamp);
    return;
  }
  has_moved_since_zupt = true;

  //===================================================================================
  // MSCKF features and KLT tracks that are SLAM features
  //===================================================================================

  // Now, lets get all features that should be used for an update that are lost in the newest frame
  // We explicitly request features that have not been deleted (used) in another update step
  std::vector<std::shared_ptr<Feature>> feats_lost, feats_marg, feats_slam;
  feats_lost = trackFEATS->get_feature_database()->features_not_containing_newer(state->_timestamp, false, true);

  // Don't need to get the oldest features until we reach our max number of clones
  // Fork: with recent clones, only when the oldest clone is marginalized after this update; a non-keyframe
  // clone is marginalized with its observations dropped (the features keep the others, at the keyframes)
  double marg_time = -1;
  bool marg_oldest = false;
  if (use_recent_clones()) {
    choose_marg_clone(marg_time, marg_oldest);
    if (marg_oldest) {
      feats_marg = trackFEATS->get_feature_database()->features_containing(marg_time, false, true);
      if (trackARUCO != nullptr && message.timestamp - startup_time >= params.dt_slam_delay) {
        feats_slam = trackARUCO->get_feature_database()->features_containing(marg_time, false, true);
      }
    }
  } else if ((int)state->_clones_IMU.size() > state->_options.max_clone_size || (int)state->_clones_IMU.size() > 5) {
    feats_marg = trackFEATS->get_feature_database()->features_containing(state->margtimestep(), false, true);
    if (trackARUCO != nullptr && message.timestamp - startup_time >= params.dt_slam_delay) {
      feats_slam = trackARUCO->get_feature_database()->features_containing(state->margtimestep(), false, true);
    }
  }

  // Remove any lost features that were from other image streams
  // E.g: if we are cam1 and cam0 has not processed yet, we don't want to try to use those in the update yet
  // E.g: thus we wait until cam0 process its newest image to remove features which were seen from that camera
  auto it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    bool found_current_message_camid = false;
    for (const auto &camuvpair : (*it1)->uvs) {
      if (std::find(message.sensor_ids.begin(), message.sensor_ids.end(), camuvpair.first) != message.sensor_ids.end()) {
        found_current_message_camid = true;
        break;
      }
    }
    if (found_current_message_camid) {
      it1++;
    } else {
      it1 = feats_lost.erase(it1);
    }
  }

  // We also need to make sure that the max tracks does not contain any lost features
  // This could happen if the feature was lost in the last frame, but has a measurement at the marg timestep
  it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    if (std::find(feats_marg.begin(), feats_marg.end(), (*it1)) != feats_marg.end()) {
      // PRINT_WARNING(YELLOW "FOUND FEATURE THAT WAS IN BOTH feats_lost and feats_marg!!!!!!\n" RESET);
      it1 = feats_lost.erase(it1);
    } else {
      it1++;
    }
  }

  // Find tracks that have reached max length, these can be made into SLAM features
  std::vector<std::shared_ptr<Feature>> feats_maxtracks;
  auto it2 = feats_marg.begin();
  while (it2 != feats_marg.end()) {
    // See if any of our camera's reached max track
    bool reached_max = false;
    for (const auto &cams : (*it2)->timestamps) {
      // Fork: with keyframes, only the observations at clones count (the others are dropped by the updates)
      int n_obs = (int)cams.second.size();
      if (params.keyframe_every_n_frames > 1) {
        n_obs = (int)std::count_if(cams.second.begin(), cams.second.end(),
                                   [&](double t) { return state->_clones_IMU.find(t) != state->_clones_IMU.end(); });
      }
      if (n_obs > state->_options.max_clone_size) {
        reached_max = true;
        break;
      }
    }
    // If max track, then add it to our possible slam feature list
    if (reached_max) {
      feats_maxtracks.push_back(*it2);
      it2 = feats_marg.erase(it2);
    } else {
      it2++;
    }
  }

  // Count how many aruco tags we have in our state
  int curr_aruco_tags = 0;
  auto it0 = state->_features_SLAM.begin();
  while (it0 != state->_features_SLAM.end()) {
    if ((int)(*it0).second->_featid <= 4 * state->_options.max_aruco_features)
      curr_aruco_tags++;
    it0++;
  }

  // Append a new SLAM feature if we have the room to do so
  // Also check that we have waited our delay amount (normally prevents bad first set of slam points)
  if (state->_options.max_slam_features > 0 && message.timestamp - startup_time >= params.dt_slam_delay &&
      (int)state->_features_SLAM.size() < state->_options.max_slam_features + curr_aruco_tags) {
    // Get the total amount to add, then the max amount that we can add given our marginalize feature array
    int amount_to_add = (state->_options.max_slam_features + curr_aruco_tags) - (int)state->_features_SLAM.size();
    int valid_amount = (amount_to_add > (int)feats_maxtracks.size()) ? (int)feats_maxtracks.size() : amount_to_add;
    // If we have at least 1 that we can add, lets add it!
    // Note: we remove them from the feat_marg array since we don't want to reuse information...
    if (params.slam_min_parallax <= 0.0) {
      if (valid_amount > 0) {
        feats_slam.insert(feats_slam.end(), feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
        feats_maxtracks.erase(feats_maxtracks.end() - valid_amount, feats_maxtracks.end());
      }
    } else {
      // Fork (slam_min_parallax): the candidates are the tracks seen in the whole window plus the tracks in the
      // current frame with enough parallax, youngest first (latest first observation): in forward flight they are
      // the ones that stay in view longest. The chosen ones leave the MSCKF lists, so their measurements are used once.
      std::vector<std::pair<double, std::shared_ptr<Feature>>> cand;
      auto first_seen = [](const std::shared_ptr<Feature> &feat) {
        double t = INFINITY;
        for (const auto &cam : feat->timestamps)
          if (!cam.second.empty())
            t = std::min(t, cam.second.front());
        return t;
      };
      for (const auto &feat : feats_maxtracks)
        cand.push_back({first_seen(feat), feat});
      // Camera poses of the clones, as the updaters build them for triangulation
      std::vector<double> clone_times;
      std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
      for (const auto &clone : state->_clones_IMU)
        clone_times.push_back(clone.first);
      for (const auto &calib : state->_calib_IMUtoCAM) {
        for (const auto &clone : state->_clones_IMU) {
          const Eigen::Matrix3d R_GtoCi = calib.second->Rot() * clone.second->Rot();
          clones_cam[calib.first].insert({clone.first, FeatureInitializer::ClonePose(R_GtoCi, clone.second->pos() - R_GtoCi.transpose() * calib.second->pos())});
        }
      }
      std::set<size_t> early;
      for (const auto &feat : trackFEATS->get_feature_database()->features_containing(state->_timestamp, false, true)) {
        if (state->_features_SLAM.find(feat->featid) != state->_features_SLAM.end() ||
            std::find(feats_maxtracks.begin(), feats_maxtracks.end(), feat) != feats_maxtracks.end() ||
            track_parallax(feat, clone_times, *promote_init, clones_cam) < params.slam_min_parallax)
          continue;
        cand.push_back({first_seen(feat), feat});
        early.insert(feat->featid);
      }
      std::sort(cand.begin(), cand.end(),
                [](const std::pair<double, std::shared_ptr<Feature>> &a, const std::pair<double, std::shared_ptr<Feature>> &b) {
                  return a.first > b.first;
                });
      for (int i = 0; i < std::min(amount_to_add, (int)cand.size()); i++) {
        const std::shared_ptr<Feature> &feat = cand[i].second;
        feats_slam.push_back(feat);
        if (early.count(feat->featid) > 0)
          slam_on_plane.insert(feat->featid);
        feats_marg.erase(std::remove(feats_marg.begin(), feats_marg.end(), feat), feats_marg.end());
        feats_maxtracks.erase(std::remove(feats_maxtracks.begin(), feats_maxtracks.end(), feat), feats_maxtracks.end());
      }
    }
  }

  // Loop through current SLAM features, we have tracks of them, grab them for this update!
  // NOTE: if we have a slam feature that has lost tracking, then we should marginalize it out
  // NOTE: we only enforce this if the current camera message is where the feature was seen from
  // NOTE: if you do not use FEJ, these types of slam features *degrade* the estimator performance....
  // NOTE: we will also marginalize SLAM features if they have failed their update a couple times in a row
  for (std::pair<const size_t, std::shared_ptr<Landmark>> &landmark : state->_features_SLAM) {
    if (trackARUCO != nullptr) {
      std::shared_ptr<Feature> feat1 = trackARUCO->get_feature_database()->get_feature(landmark.second->_featid);
      if (feat1 != nullptr)
        feats_slam.push_back(feat1);
    }
    std::shared_ptr<Feature> feat2 = trackFEATS->get_feature_database()->get_feature(landmark.second->_featid);
    if (feat2 != nullptr)
      feats_slam.push_back(feat2);
    assert(landmark.second->_unique_camera_id != -1);
    bool current_unique_cam =
        std::find(message.sensor_ids.begin(), message.sensor_ids.end(), landmark.second->_unique_camera_id) != message.sensor_ids.end();
    if (feat2 == nullptr && current_unique_cam)
      landmark.second->should_marg = true;
    if (landmark.second->update_fail_count > 1)
      landmark.second->should_marg = true;
  }

  // Lets marginalize out all old SLAM features here
  // These are ones that where not successfully tracked into the current frame
  // We do *NOT* marginalize out our aruco tags landmarks
  StateHelper::marginalize_slam(state);

  // Separate our SLAM features into new ones, and old ones
  std::vector<std::shared_ptr<Feature>> feats_slam_DELAYED, feats_slam_UPDATE;
  for (size_t i = 0; i < feats_slam.size(); i++) {
    if (state->_features_SLAM.find(feats_slam.at(i)->featid) != state->_features_SLAM.end()) {
      feats_slam_UPDATE.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: found old feature %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    } else {
      feats_slam_DELAYED.push_back(feats_slam.at(i));
      // PRINT_DEBUG("[UPDATE-SLAM]: new feature ready %d (%d
      // measurements)\n",(int)feats_slam.at(i)->featid,(int)feats_slam.at(i)->timestamps_left.size());
    }
  }

  // Concatenate our MSCKF feature arrays (i.e., ones not being used for slam updates)
  std::vector<std::shared_ptr<Feature>> featsup_MSCKF = feats_lost;
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_marg.begin(), feats_marg.end());
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_maxtracks.begin(), feats_maxtracks.end());

  //===================================================================================
  // Now that we have a list of features, lets do the EKF update for MSCKF and SLAM!
  //===================================================================================

  // Sort based on track length
  // TODO: we should have better selection logic here (i.e. even feature distribution in the FOV etc..)
  // TODO: right now features that are "lost" are at the front of this vector, while ones at the end are long-tracks
  auto compare_feat = [](const std::shared_ptr<Feature> &a, const std::shared_ptr<Feature> &b) -> bool {
    size_t asize = 0;
    size_t bsize = 0;
    for (const auto &pair : a->timestamps)
      asize += pair.second.size();
    for (const auto &pair : b->timestamps)
      bsize += pair.second.size();
    return asize < bsize;
  };
  std::sort(featsup_MSCKF.begin(), featsup_MSCKF.end(), compare_feat);

  // Pass them to our MSCKF updater
  // NOTE: if we have more then the max, we select the "best" ones (i.e. max tracks) for this update
  // NOTE: this should only really be used if you want to track a lot of features, or have limited computational resources
  if ((int)featsup_MSCKF.size() > state->_options.max_msckf_in_update)
    featsup_MSCKF.erase(featsup_MSCKF.begin(), featsup_MSCKF.end() - state->_options.max_msckf_in_update);
  const int msckf_offered = (int)featsup_MSCKF.size();
  updaterMSCKF->update(state, featsup_MSCKF);
  propagator->invalidate_cache();
  rT4 = boost::posix_time::microsec_clock::local_time();

  // Fork: one line per frame on how vision is doing, to see when updates thin out or go wrong
  // (fast rotation, blur): tracks in the current image, features offered to the MSCKF update and how
  // many survive triangulation and the chi2 test, SLAM features, and the speed estimate (print_track_stats).
  if (params.print_track_stats) {
    int tracked = 0;
    for (const auto &cam : trackFEATS->get_last_obs()) {
      tracked += (int)cam.second.size();
    }
    const auto &s = updaterMSCKF->last_stats;
    PRINT_INFO("[TRACK] t %.3f tracked %d | msckf offered %d valid %d triangulated %d used %d | slam %d | speed %.2f m/s\n",
               state->_timestamp, tracked, msckf_offered, s.in, s.triangulated, s.used, (int)state->_features_SLAM.size(),
               state->_imu->vel().norm());
  }

  // Range constraint enabled but no readings near the state's time (nothing on the topic, a wrong
  // msg_type, or stamps in another clock than the IMU's): warn every 10 s.
  if (range_buffer != nullptr) {
    const double t_imu = state->_timestamp + state->_calib_dt_CAMtoIMU->value()(0);
    double t_range = 0.0;
    const bool any = range_buffer->latest_time(t_range);
    if ((!any || std::abs(t_imu - t_range) > 1.0) && (range_last_warn_time < 0 || t_imu - range_last_warn_time > 10.0)) {
      if (any) {
        PRINT_WARNING(YELLOW "[RANGE] newest reading is %+.3f s from the state time (IMU clock); check time_offset in "
                             "relative_config_range and that the range topic is publishing\n" RESET,
                      t_range - t_imu);
      } else {
        PRINT_WARNING(YELLOW "[RANGE] use_range is on but no readings were received; check rostopic and msg_type in "
                             "relative_config_range (readings are only fed by the ROS 2 node and run_simulation)\n" RESET);
      }
      range_last_warn_time = t_imu;
    }
  }

  // Perform SLAM delay init and update
  // NOTE: that we provide the option here to do a *sequential* update
  // NOTE: this will be a lot faster but won't be as accurate.
  std::vector<std::shared_ptr<Feature>> feats_slam_UPDATE_TEMP;
  while (!feats_slam_UPDATE.empty()) {
    // Get sub vector of the features we will update with
    std::vector<std::shared_ptr<Feature>> featsup_TEMP;
    featsup_TEMP.insert(featsup_TEMP.begin(), feats_slam_UPDATE.begin(),
                        feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
    feats_slam_UPDATE.erase(feats_slam_UPDATE.begin(),
                            feats_slam_UPDATE.begin() + std::min(state->_options.max_slam_in_update, (int)feats_slam_UPDATE.size()));
    // Do the update
    updaterSLAM->update(state, featsup_TEMP);
    feats_slam_UPDATE_TEMP.insert(feats_slam_UPDATE_TEMP.end(), featsup_TEMP.begin(), featsup_TEMP.end());
    propagator->invalidate_cache();
  }
  feats_slam_UPDATE = feats_slam_UPDATE_TEMP;
  rT5 = boost::posix_time::microsec_clock::local_time();
  updaterSLAM->delayed_init(state, feats_slam_DELAYED, slam_on_plane);
  slam_on_plane.clear();
  rT6 = boost::posix_time::microsec_clock::local_time();

  // Fork: where in the image SLAM features are created and how long they live (print_track_stats), every 10 s
  if (params.print_track_stats) {
    for (const auto &feat : feats_slam_DELAYED) {
      if (state->_features_SLAM.find(feat->featid) == state->_features_SLAM.end() || slam_created.count(feat->featid) > 0)
        continue;
      slam_created[feat->featid] = state->_timestamp;
      double newest = -1.0, row = 0.0;
      for (const auto &cam : feat->timestamps) {
        if (!cam.second.empty() && cam.second.back() > newest) {
          newest = cam.second.back();
          row = feat->uvs.at(cam.first).back()(1) / state->_cam_intrinsics_cameras.at(cam.first)->h();
        }
      }
      slam_row_sum += row;
      slam_n_new++;
    }
    for (auto it = slam_created.begin(); it != slam_created.end();) {
      if (state->_features_SLAM.find(it->first) == state->_features_SLAM.end()) {
        slam_life_sum += state->_timestamp - it->second;
        slam_n_lost++;
        it = slam_created.erase(it);
      } else {
        it++;
      }
    }
    if (slam_log_last < 0 || state->_timestamp - slam_log_last >= 10.0) {
      if (slam_log_last >= 0) {
        PRINT_INFO("[SLAM] t %.1f: %d SLAM features; %ld created (mean image row %.2f, 0 top, 1 bottom), %ld lost (mean life %.1f s)\n",
                   state->_timestamp, (int)state->_features_SLAM.size(), slam_n_new, slam_row_sum / std::max(1L, slam_n_new), slam_n_lost,
                   slam_life_sum / std::max(1L, slam_n_lost));
      }
      slam_log_last = state->_timestamp;
      slam_row_sum = slam_life_sum = 0.0;
      slam_n_new = slam_n_lost = 0;
    }
  }

  // Fork: every 10 s, how far the SLAM features sit from the plane under the beam now. Their spread is the
  // terrain; a mean that grows means the map's scale or height drifts from the rangefinder's.
  if (range_updater != nullptr && (range_last_slam_log < 0 || state->_timestamp - range_last_slam_log > 10.0)) {
    double range = 0.0, z_plane = 0.0;
    const std::unordered_map<size_t, std::vector<double>> newest = {{0, {state->_timestamp}}};
    double clone_time = -1.0;
    const std::vector<Eigen::Vector3d> slam = get_features_SLAM();
    if (!slam.empty() && range_updater->find_reading(state, newest, clone_time, range) &&
        range_updater->plane_height(state, clone_time, range, z_plane)) {
      double sum = 0.0, sq = 0.0;
      for (const auto &p : slam) {
        sum += p(2) - z_plane;
        sq += (p(2) - z_plane) * (p(2) - z_plane);
      }
      const double mean = sum / slam.size();
      PRINT_INFO("[RANGE-SLAM] t %.1f: %zu SLAM features, height above the range plane mean %+.2f m, std %.2f m (range %.1f m)\n",
                 state->_timestamp, slam.size(), mean, std::sqrt(std::max(0.0, sq / slam.size() - mean * mean)), range);
      range_last_slam_log = state->_timestamp;
    }
  }

  //===================================================================================
  // Update our visualization feature set, and clean up the old features
  //===================================================================================

  // Re-triangulate all current tracks in the current frame
  if (message.sensor_ids.at(0) == 0) {

    // Re-triangulate features
    retriangulate_active_tracks(message);

    // Clear the MSCKF features only on the base camera
    // Thus we should be able to visualize the other unique camera stream
    // MSCKF features as they will also be appended to the vector
    good_features_MSCKF.clear();
  }

  // Save all the MSCKF features used in the update
  for (auto const &feat : featsup_MSCKF) {
    good_features_MSCKF.push_back(feat->p_FinG);
    feat->to_delete = true;
  }

  //===================================================================================
  // Cleanup, marginalize out what we don't need any more...
  //===================================================================================

  // Remove features that where used for the update from our extractors at the last timestep
  // This allows for measurements to be used in the future if they failed to be used this time
  // Note we need to do this before we feed a new image, as we want all new measurements to NOT be deleted
  trackFEATS->get_feature_database()->cleanup();
  if (trackARUCO != nullptr) {
    trackARUCO->get_feature_database()->cleanup();
  }

  if (use_recent_clones()) {
    // Fork: marginalize the clone chosen before the update (if any), as upstream does the oldest below
    if (marg_time >= 0) {
      updaterSLAM->change_anchors(state, marg_time);
      if (marg_oldest) {
        trackFEATS->get_feature_database()->cleanup_measurements(marg_time);
      } else {
        trackFEATS->get_feature_database()->cleanup_measurements_exact(marg_time);
      }
      if (trackARUCO != nullptr) {
        if (marg_oldest) {
          trackARUCO->get_feature_database()->cleanup_measurements(marg_time);
        } else {
          trackARUCO->get_feature_database()->cleanup_measurements_exact(marg_time);
        }
      }
      StateHelper::marginalize_clone(state, marg_time);
    }
    // Forget keyframes that are no longer clones
    for (auto it = keyframe_times.begin(); it != keyframe_times.end();) {
      it = state->_clones_IMU.find(*it) == state->_clones_IMU.end() ? keyframe_times.erase(it) : std::next(it);
    }
  } else {
    // First do anchor change if we are about to lose an anchor pose
    updaterSLAM->change_anchors(state);

    // Cleanup any features older than the marginalization time
    if ((int)state->_clones_IMU.size() > state->_options.max_clone_size) {
      trackFEATS->get_feature_database()->cleanup_measurements(state->margtimestep());
      if (trackARUCO != nullptr) {
        trackARUCO->get_feature_database()->cleanup_measurements(state->margtimestep());
      }
    }

    // Finally marginalize the oldest clone if needed
    StateHelper::marginalize_old_clone(state);
  }

  // Fork: range readings are only looked up at clone times; keep those from the oldest clone on (1 s margin)
  if (range_buffer != nullptr && !state->_clones_IMU.empty()) {
    range_buffer->prune_before(state->_clones_IMU.begin()->first + state->_calib_dt_CAMtoIMU->value()(0) - 1.0);
  }
  rT7 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  // Debug info, and stats tracking
  //===================================================================================

  // Get timing statitics information
  double time_track = (rT2 - rT1).total_microseconds() * 1e-6;
  double time_prop = (rT3 - rT2).total_microseconds() * 1e-6;
  double time_msckf = (rT4 - rT3).total_microseconds() * 1e-6;
  double time_slam_update = (rT5 - rT4).total_microseconds() * 1e-6;
  double time_slam_delay = (rT6 - rT5).total_microseconds() * 1e-6;
  double time_marg = (rT7 - rT6).total_microseconds() * 1e-6;
  double time_total = (rT7 - rT1).total_microseconds() * 1e-6;

  // Timing information
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for tracking\n" RESET, time_track);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for propagation\n" RESET, time_prop);
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for MSCKF update (%d feats)\n" RESET, time_msckf, (int)featsup_MSCKF.size());
  if (state->_options.max_slam_features > 0) {
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM update (%d feats)\n" RESET, time_slam_update, (int)state->_features_SLAM.size());
    PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for SLAM delayed init (%d feats)\n" RESET, time_slam_delay, (int)feats_slam_DELAYED.size());
  }
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for re-tri & marg (%d clones in state)\n" RESET, time_marg, (int)state->_clones_IMU.size());

  std::stringstream ss;
  ss << "[TIME]: " << std::setprecision(4) << time_total << " seconds for total (camera";
  for (const auto &id : message.sensor_ids) {
    ss << " " << id;
  }
  ss << ")" << std::endl;
  PRINT_DEBUG(BLUE "%s" RESET, ss.str().c_str());

  // Finally if we are saving stats to file, lets save it to file
  if (params.record_timing_information && of_statistics.is_open()) {
    // We want to publish in the IMU clock frame
    // The timestamp in the state will be the last camera time
    double t_ItoC = state->_calib_dt_CAMtoIMU->value()(0);
    double timestamp_inI = state->_timestamp + t_ItoC;
    // Append to the file
    of_statistics << std::fixed << std::setprecision(15) << timestamp_inI << "," << std::fixed << std::setprecision(5) << time_track << ","
                  << time_prop << "," << time_msckf << ",";
    if (state->_options.max_slam_features > 0) {
      of_statistics << time_slam_update << "," << time_slam_delay << ",";
    }
    of_statistics << time_marg << "," << time_total << std::endl;
    of_statistics.flush();
  }

  // Update our distance traveled
  if (timelastupdate != -1 && state->_clones_IMU.find(timelastupdate) != state->_clones_IMU.end()) {
    Eigen::Matrix<double, 3, 1> dx = state->_imu->pos() - state->_clones_IMU.at(timelastupdate)->pos();
    distance += dx.norm();
  }
  timelastupdate = message.timestamp;

  // Debug, print our current state
  PRINT_INFO("q_GtoI = %.3f,%.3f,%.3f,%.3f | p_IinG = %.3f,%.3f,%.3f | dist = %.2f (meters)\n", state->_imu->quat()(0),
             state->_imu->quat()(1), state->_imu->quat()(2), state->_imu->quat()(3), state->_imu->pos()(0), state->_imu->pos()(1),
             state->_imu->pos()(2), distance);
  PRINT_INFO("bg = %.4f,%.4f,%.4f | ba = %.4f,%.4f,%.4f\n", state->_imu->bias_g()(0), state->_imu->bias_g()(1), state->_imu->bias_g()(2),
             state->_imu->bias_a()(0), state->_imu->bias_a()(1), state->_imu->bias_a()(2));

  // Debug for camera imu offset
  if (state->_options.do_calib_camera_timeoffset) {
    PRINT_INFO("camera-imu timeoffset = %.5f\n", state->_calib_dt_CAMtoIMU->value()(0));
  }

  // Debug for camera intrinsics
  if (state->_options.do_calib_camera_intrinsics) {
    for (int i = 0; i < state->_options.num_cameras; i++) {
      std::shared_ptr<Vec> calib = state->_cam_intrinsics.at(i);
      PRINT_INFO("cam%d intrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f,%.3f\n", (int)i, calib->value()(0), calib->value()(1),
                 calib->value()(2), calib->value()(3), calib->value()(4), calib->value()(5), calib->value()(6), calib->value()(7));
    }
  }

  // Debug for camera extrinsics
  if (state->_options.do_calib_camera_pose) {
    for (int i = 0; i < state->_options.num_cameras; i++) {
      std::shared_ptr<PoseJPL> calib = state->_calib_IMUtoCAM.at(i);
      PRINT_INFO("cam%d extrinsics = %.3f,%.3f,%.3f,%.3f | %.3f,%.3f,%.3f\n", (int)i, calib->quat()(0), calib->quat()(1), calib->quat()(2),
                 calib->quat()(3), calib->pos()(0), calib->pos()(1), calib->pos()(2));
    }
  }

  // Debug for imu intrinsics
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::KALIBR) {
    PRINT_INFO("q_GYROtoI = %.3f,%.3f,%.3f,%.3f\n", state->_calib_imu_GYROtoIMU->value()(0), state->_calib_imu_GYROtoIMU->value()(1),
               state->_calib_imu_GYROtoIMU->value()(2), state->_calib_imu_GYROtoIMU->value()(3));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::RPNG) {
    PRINT_INFO("q_ACCtoI = %.3f,%.3f,%.3f,%.3f\n", state->_calib_imu_ACCtoIMU->value()(0), state->_calib_imu_ACCtoIMU->value()(1),
               state->_calib_imu_ACCtoIMU->value()(2), state->_calib_imu_ACCtoIMU->value()(3));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::KALIBR) {
    PRINT_INFO("Dw = | %.4f,%.4f,%.4f | %.4f,%.4f | %.4f |\n", state->_calib_imu_dw->value()(0), state->_calib_imu_dw->value()(1),
               state->_calib_imu_dw->value()(2), state->_calib_imu_dw->value()(3), state->_calib_imu_dw->value()(4),
               state->_calib_imu_dw->value()(5));
    PRINT_INFO("Da = | %.4f,%.4f,%.4f | %.4f,%.4f | %.4f |\n", state->_calib_imu_da->value()(0), state->_calib_imu_da->value()(1),
               state->_calib_imu_da->value()(2), state->_calib_imu_da->value()(3), state->_calib_imu_da->value()(4),
               state->_calib_imu_da->value()(5));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.imu_model == StateOptions::ImuModel::RPNG) {
    PRINT_INFO("Dw = | %.4f | %.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_calib_imu_dw->value()(0), state->_calib_imu_dw->value()(1),
               state->_calib_imu_dw->value()(2), state->_calib_imu_dw->value()(3), state->_calib_imu_dw->value()(4),
               state->_calib_imu_dw->value()(5));
    PRINT_INFO("Da = | %.4f | %.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_calib_imu_da->value()(0), state->_calib_imu_da->value()(1),
               state->_calib_imu_da->value()(2), state->_calib_imu_da->value()(3), state->_calib_imu_da->value()(4),
               state->_calib_imu_da->value()(5));
  }
  if (state->_options.do_calib_imu_intrinsics && state->_options.do_calib_imu_g_sensitivity) {
    PRINT_INFO("Tg = | %.4f,%.4f,%.4f |  %.4f,%.4f,%.4f | %.4f,%.4f,%.4f |\n", state->_calib_imu_tg->value()(0),
               state->_calib_imu_tg->value()(1), state->_calib_imu_tg->value()(2), state->_calib_imu_tg->value()(3),
               state->_calib_imu_tg->value()(4), state->_calib_imu_tg->value()(5), state->_calib_imu_tg->value()(6),
               state->_calib_imu_tg->value()(7), state->_calib_imu_tg->value()(8));
  }
}
