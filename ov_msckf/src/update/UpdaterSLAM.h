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

#ifndef OV_MSCKF_UPDATER_SLAM_H
#define OV_MSCKF_UPDATER_SLAM_H

#include <Eigen/Eigen>
#include <memory>
#include <set>

#include "feat/FeatureInitializerOptions.h"

#include "UpdaterOptions.h"
#include "UpdaterRange.h"

namespace ov_core {
class Feature;
class FeatureInitializer;
} // namespace ov_core
namespace ov_type {
class Landmark;
} // namespace ov_type

namespace ov_msckf {

class State;

/**
 * @brief Will compute the system for our sparse SLAM features and update the filter.
 *
 * This class is responsible for performing delayed feature initialization, SLAM update, and
 * SLAM anchor change for anchored feature representations.
 */
class UpdaterSLAM {

public:
  /**
   * @brief Default constructor for our SLAM updater
   *
   * Our updater has a feature initializer which we use to initialize features as needed.
   * Also the options allow for one to tune the different parameters for update.
   *
   * @param options_slam Updater options (include measurement noise value) for SLAM features
   * @param options_aruco Updater options (include measurement noise value) for ARUCO features
   * @param feat_init_options Feature initializer options
   */
  UpdaterSLAM(UpdaterOptions &options_slam, UpdaterOptions &options_aruco, ov_core::FeatureInitializerOptions &feat_init_options);

  /**
   * @brief Given tracked SLAM features, this will try to use them to update the state.
   * @param state State of the filter
   * @param feature_vec Features that can be used for update
   */
  void update(std::shared_ptr<State> state, std::vector<std::shared_ptr<ov_core::Feature>> &feature_vec);

  /**
   * @brief Given max track features, this will try to use them to initialize them in the state.
   * @param state State of the filter
   * @param feature_vec Features that can be used for update
   * @param on_plane Fork: ids of features to create on the plane from the range (tracks promoted early by
   * slam_min_parallax; with the rangefinder only)
   */
  void delayed_init(std::shared_ptr<State> state, std::vector<std::shared_ptr<ov_core::Feature>> &feature_vec,
                    const std::set<size_t> &on_plane = {});

  /**
   * @brief Fork: create SLAM features with the rangefinder (range_slam_init). A feature triangulated with enough
   * parallax is initialised from all its image observations plus a range row; otherwise (or if triangulation
   * fails) it is placed where its newest bearing meets the plane at the measured height and initialised from that
   * bearing and the range alone, without updating the rest of the state. See UpdaterRange.
   */
  void set_range(std::shared_ptr<UpdaterRange> range) { _range = range; }

  /// RATE (experimental): angular rate [rad/s] by clone time, for the rate-dependent pixel noise
  void set_rates(std::shared_ptr<std::map<double, double>> rates) { _rates = rates; }
  std::shared_ptr<std::map<double, double>> _rates;

  /// DIAG (temporary): last update's SLAM features offered and chi2-rejected
  int diag_in = 0, diag_chi2 = 0;

  /**
   * @brief Will change SLAM feature anchors if it will be marginalized
   *
   * Makes sure that if any clone is about to be marginalized, it changes anchor representation.
   * By default, this will shift the anchor into the newest IMU clone and keep the camera calibration anchor the same.
   *
   * @param state State of the filter
   */
  void change_anchors(std::shared_ptr<State> state);

  /**
   * @brief Fork: move the anchor of SLAM features anchored in the given clone (any clone, e.g. a non-keyframe
   * about to be marginalized) into the newest clone
   * @param state State of the filter
   * @param marg_timestep Time of the clone that will be marginalized
   */
  void change_anchors(std::shared_ptr<State> state, double marg_timestep);

protected:
  /**
   * @brief Shifts landmark anchor to new clone
   * @param state State of filter
   * @param landmark landmark whose anchor is being shifter
   * @param new_anchor_timestamp Clone timestamp we want to move to
   * @param new_cam_id Which camera frame we want to move to
   */
  void perform_anchor_change(std::shared_ptr<State> state, std::shared_ptr<ov_type::Landmark> landmark, double new_anchor_timestamp,
                             size_t new_cam_id);

  /// Options used during update for slam features
  UpdaterOptions _options_slam;

  /// Options used during update for aruco features
  UpdaterOptions _options_aruco;

  /// Feature initializer class object
  std::shared_ptr<ov_core::FeatureInitializer> initializer_feat;

  /// Fork: the rangefinder plane constraint for SLAM feature initialisation (null: upstream)
  std::shared_ptr<UpdaterRange> _range;

  /// Chi squared 95th percentile table (lookup would be size of residual)
  std::map<int, double> chi_squared_table;
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_SLAM_H
