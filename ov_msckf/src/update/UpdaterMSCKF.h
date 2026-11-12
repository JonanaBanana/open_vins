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

#ifndef OV_MSCKF_UPDATER_MSCKF_H
#define OV_MSCKF_UPDATER_MSCKF_H

#include <Eigen/Eigen>
#include <memory>

#include "feat/FeatureInitializerOptions.h"

#include "RangePlane.h"
#include "UpdaterOptions.h"

namespace ov_core {
class Feature;
class FeatureInitializer;
} // namespace ov_core

namespace ov_type {
class Type;
} // namespace ov_type

namespace ov_msckf {

class State;

/**
 * @brief Will compute the system for our sparse features and update the filter.
 *
 * This class is responsible for computing the entire linear system for all features that are going to be used in an update.
 * This follows the original MSCKF, where we first triangulate features, we then nullspace project the feature Jacobian.
 * After this we compress all the measurements to have an efficient update and update the state.
 */
class UpdaterMSCKF {

public:
  /**
   * @brief Default constructor for our MSCKF updater
   *
   * Our updater has a feature initializer which we use to initialize features as needed.
   * Also the options allow for one to tune the different parameters for update.
   *
   * @param options Updater options (include measurement noise value)
   * @param feat_init_options Feature initializer options
   */
  UpdaterMSCKF(UpdaterOptions &options, ov_core::FeatureInitializerOptions &feat_init_options);

  /**
   * @brief Given tracked features, this will try to use them to update the state.
   *
   * @param state State of the filter
   * @param feature_vec Features that can be used for update
   */
  void update(std::shared_ptr<State> state, std::vector<std::shared_ptr<ov_core::Feature>> &feature_vec);

  /**
   * @brief GeoTNav: enable the rangefinder plane constraint.
   *
   * Each feature that passes the (visual) chi2 test gets one more row before it is projected out: the
   * rangefinder's slant range at the newest clone it is seen from, predicted as the distance along the
   * beam to the horizontal plane through the feature, (p_I,z - p_F,z) / cos(tilt). That ties the
   * features' depths, and so the scale, to a metric height.
   */
  void set_range(std::shared_ptr<RangeBuffer> buffer, const RangeOptions &options) {
    _range_buffer = buffer;
    _range_options = options;
  }

  /// GeoTNav: how the last update's features fared (for the per-frame [TRACK] log line)
  struct Stats {
    int in = 0;            // features offered with at least two measurements
    int triangulated = 0;  // of those, triangulated and refined
    int used = 0;          // of those, passed the chi2 test and went into the update
  };
  Stats last_stats;

protected:
  /**
   * @brief GeoTNav: append the range row to a feature's (unprojected) system; false if no valid range.
   *
   * Only for GLOBAL_3D features (H_f's columns are p_FinG). The row is whitened to the pixel noise,
   * with a Huber weight beyond huber_k sigmas.
   */
  bool append_range_row(std::shared_ptr<State> state, const std::shared_ptr<ov_core::Feature> &feature, Eigen::MatrixXd &H_f,
                        Eigen::MatrixXd &H_x, Eigen::VectorXd &res, const std::vector<std::shared_ptr<ov_type::Type>> &Hx_order);

  /// GeoTNav: rangefinder readings (null: the constraint is off) and options
  std::shared_ptr<RangeBuffer> _range_buffer;
  RangeOptions _range_options;

  /// GeoTNav: statistics of the range rows, logged periodically
  long _range_rows = 0;
  long _range_huber = 0;
  long _range_missing = 0;
  double _range_res_sum = 0.0;
  double _range_res_sq = 0.0;
  double _range_bin_sum[4] = {0, 0, 0, 0};  // whole run, by tilt
  long _range_bin_n[4] = {0, 0, 0, 0};

  /// Options used during update
  UpdaterOptions _options;

  /// Feature initializer class object
  std::shared_ptr<ov_core::FeatureInitializer> initializer_feat;

  /// Chi squared 95th percentile table (lookup would be size of residual)
  std::map<int, double> chi_squared_table;
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_MSCKF_H
