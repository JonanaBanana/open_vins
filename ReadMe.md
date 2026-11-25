# OpenVINS

[![ROS 1 Workflow](https://github.com/rpng/open_vins/actions/workflows/build_ros1.yml/badge.svg)](https://github.com/rpng/open_vins/actions/workflows/build_ros1.yml)
[![ROS 2 Workflow](https://github.com/rpng/open_vins/actions/workflows/build_ros2.yml/badge.svg)](https://github.com/rpng/open_vins/actions/workflows/build_ros2.yml)
[![ROS Free Workflow](https://github.com/rpng/open_vins/actions/workflows/build.yml/badge.svg)](https://github.com/rpng/open_vins/actions/workflows/build.yml)

Welcome to the OpenVINS project!
The OpenVINS project houses some core computer vision code along with a state-of-the art filter-based visual-inertial
estimator. The core filter is an [Extended Kalman filter](https://en.wikipedia.org/wiki/Extended_Kalman_filter) which
fuses inertial information with sparse visual feature tracks. These visual feature tracks are fused leveraging
the [Multi-State Constraint Kalman Filter (MSCKF)](https://ieeexplore.ieee.org/document/4209642) sliding window
formulation which allows for 3D features to update the state estimate without directly estimating the feature states in
the filter. Inspired by graph-based optimization systems, the included filter has modularity allowing for convenient
covariance management with a proper type-based state system. Please take a look at the feature list below for full
details on what the system supports.

* Github project page - https://github.com/rpng/open_vins
* Documentation - https://docs.openvins.com/
* Getting started guide - https://docs.openvins.com/getting-started.html
* Publication reference - https://pgeneva.com/downloads/papers/Geneva2020ICRA.pdf

## About this fork

This fork adds support for monocular VIO from a downward-looking camera at altitude: a rangefinder constraint that
fixes the metric scale, a longer sliding window for distant scenes, and a few fixes. Changes are marked `Fork:` in
the source.

* **ROS 2 only.** The range readings are fed by the ROS 2 node (`run_subscribe_msckf`) and `run_simulation`. The ROS 1
  wrapper builds but does not subscribe to a rangefinder and lacks the ROS 2 fixes below.
* **Everything is opt-in.** With none of the options below set, the filter behaves as upstream (checked in simulation:
  identical estimates on `config/rpng_sim`).
* **Example:** `config/sim_range_flight` uses all of it (see [Simulation](#simulation)).

### Rangefinder plane constraint

A single-beam rangefinder pointing at the ground fixes the scale of monocular VIO. The ground is modelled as a plane
perpendicular to gravity: the range at a clone gives the plane's height below the vehicle, and each MSCKF feature,
assumed to lie on that plane, gets one more row in its update relating the measured slant range to the clone's
height, attitude and the feature's height. New SLAM features can also be created with the range.

Enable it with `use_range: true` and describe the sensor in a separate file, like the IMU and cameras:

```yaml
# estimator_config.yaml
use_range: true
relative_config_range: "range_chain.yaml"
```

```yaml
# range_chain.yaml
range0:
  T_imu_range:        # rangefinder frame in the IMU frame (required); the beam is its +x axis (ROS convention)
    - [ 0.0, 0.0, 1.0, 0.0]   # here: beam along the IMU's -z (down for an IMU with z up)
    - [ 0.0, 1.0, 0.0, 0.0]   # the translation is the lever arm [m]
    - [-1.0, 0.0, 0.0, 0.0]
    - [ 0.0, 0.0, 0.0, 1.0]
  rostopic: /rangefinder      # required
  msg_type: range             # range (sensor_msgs/Range, default) or laserscan (sensor_msgs/LaserScan, first range)
  time_offset: 0.0            # [s] t_imu = t_range + time_offset
```

Readings outside the message's own limits, or not finite, are treated as gaps. Options in `estimator_config.yaml`:

| Option | Default | Meaning |
|---|---|---|
| `range_sigma` | 3.0 | [m] each feature's own deviation from the plane (vegetation, roofs, terrain) |
| `range_sigma_shared` | 0.0 | [m] error shared by all rows using the same reading (range noise, footprint off the plane). A row's variance is `range_sigma² + N·range_sigma_shared²` with N the rows sharing the reading, so one reading never counts as many independent measurements. 0: rows independent |
| `range_huber_k` | 3.0 | residuals beyond this many sigmas get a proportionally larger sigma (bounded influence) |
| `range_min`, `range_max` | 0, ∞ | [m] valid readings are in (min, max], on top of the message's limits. Set `range_max` below a "no return" value if the sensor reports one as a range |
| `range_max_tilt_deg` | 60 | no rows when the beam is tilted further from vertical |
| `range_max_gap` | 0.1 | [s] largest gap between the two readings interpolated at a clone's time |
| `range_min_height` | 0.0 | [m] no rows below this height above the plane (take-off, landing) |
| `range_slam_init` | true | create SLAM features with the range: tracks seen in the whole window from all image observations plus a range row; tracks promoted early (`slam_min_parallax`) or failing triangulation on the plane from one bearing and the range |
| `range_outlier_max_dev` | 0.0 | [m] drop a reading whose ground height differs by more than this from the median of the window's other readings (short returns off trees, roofs). 0: off |

Requirements and limits:

* `feat_rep_msckf: GLOBAL_3D` is required (start-up error otherwise). `range_slam_init` has no effect with
  `feat_rep_slam: ANCHORED_INVERSE_DEPTH_SINGLE`.
* The camera should see the ground the beam ranges to (a start-up warning is printed if a camera looks away from the
  beam). A beam tilted beyond `range_max_tilt_deg` is ignored, so a sensor that does not point down has no effect.
* **Sloped terrain.** Range and vision cannot tell a ground slope from a tilt of the vehicle. On a slope the filter
  tilts by about the slope and the height drifts with the distance flown (in simulation, about 10 m over a 300 s
  flight on a 3 % slope). Flat or gently undulating terrain is fine. Absolute altitude (a terrain map, a global
  reference) is out of scope and belongs downstream of the odometry; such a layer should not treat the odometry's
  height and the rangefinder as independent.
* A `[RANGE]` line is logged periodically (rows used, residuals, rows per reading, rejected readings), and a warning
  every 10 s if no readings arrive near the state's time (topic, message type or clock offset).

### Keyframe window

Distant scenes need a long baseline, but tracking and initialisation need a high frame rate. With
`keyframe_every_n_frames: n` (> 1), the window of `max_clones` keeps clones older than the newest
`num_recent_clones` only at keyframes (every n-th tracked frame), so it spans a longer time:

| Option | Default | Meaning |
|---|---|---|
| `keyframe_every_n_frames` | 1 | every n-th tracked frame is a keyframe. 1: every frame (upstream) |
| `num_recent_clones` | 0 | the newest frames kept as clones whether or not they are keyframes; every frame is then updated, and a non-keyframe leaving this part is marginalized with its observations. 0: only keyframes are propagated, cloned and updated. At most `max_clones - 2` |

For example `max_clones: 15`, `keyframe_every_n_frames: 6`, `num_recent_clones: 5` at 15 Hz gives an update every
frame and a window of about 4.3 s. For distant features, also raise the triangulation limits (`fi_max_dist`,
`fi_max_cond_number`): the defaults reject features about 100 m away.

### Other options

* `init_heading_axis_in_imu: [x, y, z]` (unset: upstream): the global frame's x axis is this IMU axis (e.g. the
  vehicle's forward) made horizontal at initialisation, static or dynamic.
* `slam_min_parallax` (default 0: upstream): upstream only makes SLAM features of tracks seen in the whole window,
  which in forward flight have already crossed much of the image and soon leave it. With this set, any track with at
  least this parallax can become one (youngest first), measured as upstream measures it for `fi_max_baseline`:
  perpendicular baseline / distance (`fi_max_baseline: 40` corresponds to 0.025). With the rangefinder they are created
  on the plane from the range, and vision refines them from then on. 0.03 worked well in simulation.
* `print_track_stats: true`: one `[TRACK]` line per update (tracks, features used, SLAM features, speed) and a
  `[SLAM]` line every 10 s (where SLAM features are created in the image, how long they live).
* ROS 2 parameters: `frame_global`, `frame_imu`, `frame_cam` (frame names; defaults as upstream) and `odom_rate_hz`
  (limit the odometry and TF rate; 0: every IMU sample).

### Fixes

* Published odometry: the first IMU-rate pose after each update was `calib_dt_CAMtoIMU` old
  (`Propagator::fast_state_propagate` mixed the camera and IMU clocks). The filter's estimate is unchanged.
* ROS 2 wrapper: the IMU thread read a dangling reference to the callback's message; shutdown order and thread joins
  (crashes at exit); a deep IMU queue (1000) for bursty drivers; out-of-order IMU samples dropped; odometry, TF,
  images and point clouds stamped with the state's time instead of `now()`.
* Simulator: points beyond the lens model's valid angle (where a wide-angle distortion model folds back) were
  projected into the image at wrong positions. This changes simulated measurements for strongly distorted cameras.

### Simulation

`run_simulation` can test the rangefinder against ground truth:

* `sim_ground_plane: true`: features are generated on a ground plane (`sim_ground_height`, `sim_ground_gradient`,
  `sim_ground_roughness`, and hills `sim_ground_hills: [amplitude, wavelength]`) instead of at random depths.
* With `use_range: true`, a simulated rangefinder along `T_imu_range` (`sim_range_freq`, `sim_range_sigma`, and
  short returns with `sim_range_outlier_prob`, `sim_range_outlier_max`).
* `ov_data/sim/generate_flight.py` generates a fixed-altitude figure-eight flight (with an optional hover);
  `flight_100m.txt` and `flight_100m_hover.txt` are included.
* `config/sim_range_flight` flies `flight_100m.txt` with a downward fisheye camera and rangefinder. Run it from the
  workspace root (`sim_traj_path` is relative).

## News / Events

* **May 11, 2023** - Inertial intrinsic support released as part of v2.7 along with a few bug fixes and improvements to stereo KLT tracking. Please check out the [release page](https://github.com/rpng/open_vins/releases/tag/v2.7) for details.
* **April 15, 2023** - Minor update to v2.6.3 to support incremental feature triangulation of active features for downstream applications, faster zero-velocity update, small bug fixes, some example realsense configurations, and cached fast state prediction. Please check out the [release page](https://github.com/rpng/open_vins/releases/tag/v2.6.3) for details.
* **April 3, 2023** - We have released a monocular plane-aided VINS, termed [ov_plane](https://github.com/rpng/ov_plane), which leverages the OpenVINS project. Both now support the released [Indoor AR Table](https://github.com/rpng/ar_table_dataset) dataset.
* **July 14, 2022** - Improved feature extraction logic for >100hz tracking, some bug fixes and updated scripts. See v2.6.1 [PR#259](https://github.com/rpng/open_vins/pull/259) and v2.6.2 [PR#264](https://github.com/rpng/open_vins/pull/264).
* **March 14, 2022** - Initial dynamic initialization open sourcing, asynchronous subscription to inertial readings and publishing of odometry, support for lower frequency feature tracking. See v2.6 [PR#232](https://github.com/rpng/open_vins/pull/232) for details.
* **December 13, 2021** - New YAML configuration system, ROS2 support, Docker images, robust static initialization based on disparity, internal logging system to reduce verbosity, image transport publishers, dynamic number of features support, and other small fixes. See
  v2.5 [PR#209](https://github.com/rpng/open_vins/pull/209) for details.
* **July 19, 2021** - Camera classes, masking support, alignment utility, and other small fixes. See
  v2.4 [PR#117](https://github.com/rpng/open_vins/pull/186) for details.
* **December 1, 2020** - Released improved memory management, active feature pointcloud publishing, limiting number of
  features in update to bound compute, and other small fixes. See
  v2.3 [PR#117](https://github.com/rpng/open_vins/pull/117) for details.
* **November 18, 2020** - Released groundtruth generation utility package, [vicon2gt](https://github.com/rpng/vicon2gt)
  to enable creation of groundtruth trajectories in a motion capture room for evaulating VIO methods.
* **July 7, 2020** - Released zero velocity update for vehicle applications and direct initialization when standing
  still. See [PR#79](https://github.com/rpng/open_vins/pull/79) for details.
* **May 18, 2020** - Released secondary pose graph example
  repository [ov_secondary](https://github.com/rpng/ov_secondary) based
  on [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion). OpenVINS now publishes marginalized feature
  track, feature 3d position, and first camera intrinsics and extrinsics.
  See [PR#66](https://github.com/rpng/open_vins/pull/66) for details and discussion.
* **April 3, 2020** - Released [v2.0](https://github.com/rpng/open_vins/releases/tag/v2.0) update to the codebase with
  some key refactoring, ros-free building, improved dataset support, and single inverse depth feature representation.
  Please check out the [release page](https://github.com/rpng/open_vins/releases/tag/v2.0) for details.
* **January 21, 2020** - Our paper has been accepted for presentation in [ICRA 2020](https://www.icra2020.org/). We look
  forward to seeing everybody there! We have also added links to a few videos of the system running on different
  datasets.
* **October 23, 2019** - OpenVINS placed first in the [IROS 2019 FPV Drone Racing VIO Competition
  ](http://rpg.ifi.uzh.ch/uzh-fpv.html). We will be giving a short presentation at
  the [workshop](https://wp.nyu.edu/workshopiros2019mav/) at 12:45pm in Macau on November 8th.
* **October 1, 2019** - We will be presenting at the [Visual-Inertial Navigation: Challenges and Applications
  ](http://udel.edu/~ghuang/iros19-vins-workshop/index.html) workshop at [IROS 2019](https://www.iros2019.org/). The
  submitted workshop paper can be found at [this](http://udel.edu/~ghuang/iros19-vins-workshop/papers/06.pdf) link.
* **August 21, 2019** - Open sourced [ov_maplab](https://github.com/rpng/ov_maplab) for interfacing OpenVINS with
  the [maplab](https://github.com/ethz-asl/maplab) library.
* **August 15, 2019** - Initial release of OpenVINS repository and documentation website!

## Project Features

* Sliding window visual-inertial MSCKF
* Modular covariance type system
* Comprehensive documentation and derivations
* Extendable visual-inertial simulator
    * On manifold SE(3) b-spline
    * Arbitrary number of cameras
    * Arbitrary sensor rate
    * Automatic feature generation
* Five different feature representations
    1. Global XYZ
    2. Global inverse depth
    3. Anchored XYZ
    4. Anchored inverse depth
    5. Anchored MSCKF inverse depth
    6. Anchored single inverse depth
* Calibration of sensor intrinsics and extrinsics
    * Camera to IMU transform
    * Camera to IMU time offset
    * Camera intrinsics
    * Inertial intrinsics (including g-sensitivity)
* Environmental SLAM feature
    * OpenCV ARUCO tag SLAM features
    * Sparse feature SLAM features
* Visual tracking support
    * Monocular camera
    * Stereo camera (synchronized)
    * Binocular cameras (synchronized)
    * KLT or descriptor based
    * Masked tracking
* Static and dynamic state initialization
* Zero velocity detection and updates
* Out of the box evaluation on EuRocMav, TUM-VI, UZH-FPV, KAIST Urban and other VIO datasets
* Extensive evaluation suite (ATE, RPE, NEES, RMSE, etc..)

## Codebase Extensions

* **[ov_plane](https://github.com/rpng/ov_plane)** - A real-time monocular visual-inertial odometry (VIO) system which leverages
  environmental planes. At the core it presents an efficient robust monocular-based plane detection algorithm which does
  not require additional sensing modalities such as a stereo, depth camera or neural network. The plane detection and tracking
  algorithm enables real-time regularization of point features to environmental planes which are either maintained in the state
  vector as long-lived planes, or marginalized for efficiency. Planar regularities are applied to both in-state SLAM and
  out-of-state MSCKF point features, enabling long-term point-to-plane loop-closures due to the large spacial volume of planes.

* **[vicon2gt](https://github.com/rpng/vicon2gt)** - This utility was created to generate groundtruth trajectories using
  a motion capture system (e.g. Vicon or OptiTrack) for use in evaluating visual-inertial estimation systems.
  Specifically we calculate the inertial IMU state (full 15 dof) at camera frequency rate and generate a groundtruth
  trajectory similar to those provided by the EurocMav datasets. Performs fusion of inertial and motion capture
  information and estimates all unknown spacial-temporal calibrations between the two sensors.

* **[ov_maplab](https://github.com/rpng/ov_maplab)** - This codebase contains the interface wrapper for exporting
  visual-inertial runs from [OpenVINS](https://github.com/rpng/open_vins) into the ViMap structure taken
  by [maplab](https://github.com/ethz-asl/maplab). The state estimates and raw images are appended to the ViMap as
  OpenVINS runs through a dataset. After completion of the dataset, features are re-extract and triangulate with
  maplab's feature system. This can be used to merge multi-session maps, or to perform a batch optimization after first
  running the data through OpenVINS. Some example have been provided along with a helper script to export trajectories
  into the standard groundtruth format.

* **[ov_secondary](https://github.com/rpng/ov_secondary)** - This is an example secondary thread which provides loop
  closure in a loosely coupled manner for [OpenVINS](https://github.com/rpng/open_vins). This is a modification of the
  code originally developed by the HKUST aerial robotics group and can be found in
  their [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) repository. Here we stress that this is a
  loosely coupled method, thus no information is returned to the estimator to improve the underlying OpenVINS odometry.
  This codebase has been modified in a few key areas including: exposing more loop closure parameters, subscribing to
  camera intrinsics, simplifying configuration such that only topics need to be supplied, and some tweaks to the loop
  closure detection to improve frequency.


## Demo Videos

<a href="http://www.youtube.com/watch?v=KCX51GvYGss">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/KCX51GvYGss.jpg" width="120" height="90" />
</a>
<a href="http://www.youtube.com/watch?v=Lc7VQHngSuQ">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/Lc7VQHngSuQ.jpg" width="120" height="90" />
</a>
<a href="http://www.youtube.com/watch?v=vaia7iPaRW8">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/vaia7iPaRW8.jpg" width="120" height="90" />
</a>
<a href="http://www.youtube.com/watch?v=MCzTF9ye2zw">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/MCzTF9ye2zw.jpg"  width="120" height="90"/>
</a>
<a href="http://www.youtube.com/watch?v=eSQLWcNrx_I">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/eSQLWcNrx_I.jpg" width="120" height="90" />
</a>
<br/>

<a href="http://www.youtube.com/watch?v=187AXuuGNNw">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/187AXuuGNNw.jpg" width="120" height="90" />
</a>
<a href="http://www.youtube.com/watch?v=oUoLlrFryk0">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/oUoLlrFryk0.jpg" width="120" height="90" />
</a>
<a href="http://www.youtube.com/watch?v=ExPIGwORm4E">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/ExPIGwORm4E.jpg" width="120" height="90" />
</a>
<a href="http://www.youtube.com/watch?v=lXHl-qgLGl8">
   <img src="https://raw.githubusercontent.com/rpng/open_vins/master/docs/youtube/lXHl-qgLGl8.jpg" width="120" height="90" />
</a>



## Credit / Licensing

This code was written by the [Robot Perception and Navigation Group (RPNG)](https://sites.udel.edu/robot/) at the
University of Delaware. If you have any issues with the code please open an issue on our github page with relevant
implementation details and references. For researchers that have leveraged or compared to this work, please cite the
following:

```txt
@Conference{Geneva2020ICRA,
  Title      = {{OpenVINS}: A Research Platform for Visual-Inertial Estimation},
  Author     = {Patrick Geneva and Kevin Eckenhoff and Woosik Lee and Yulin Yang and Guoquan Huang},
  Booktitle  = {Proc. of the IEEE International Conference on Robotics and Automation},
  Year       = {2020},
  Address    = {Paris, France},
  Url        = {\url{https://github.com/rpng/open_vins}}
}
```

The codebase and documentation is licensed under the [GNU General Public License v3 (GPL-3)](https://www.gnu.org/licenses/gpl-3.0.txt).
You must preserve the copyright and license notices in your derivative work and make available the complete source code with modifications under the same license ([see this](https://choosealicense.com/licenses/gpl-3.0/); this is not legal advice).


