#!/bin/bash
# Replays a VOXL bag through the ircam_wide decoder and OpenVINS, recording OpenVINS's odometry and
# PX4's (the reference) with vio_record.py.
#
#   run_replay.sh <config dir> <out prefix> <bag> [seconds] [ROS domain id]
#
# Inputs to OpenVINS: /ircam_wide/decoded (decoded from /ircam_wide/h264), /imu_apps and, with
# use_range, /alt_raw. PX4's odometry (/fmu/out/vehicle_odometry) is played for the recorder: the
# reference, never an input.
# Separate domain ids let several replays run at once. Then: ov_eval_init.py <out prefix>.
# OV_ARGS adds ROS parameters to OpenVINS, e.g.
#   OV_ARGS="-p frame_global:=odom_vio -p frame_imu:=base_link_vio -p odom_rate_hz:=50"
# DECODER_CMD is the command that decodes /ircam_wide/h264 into /ircam_wide/decoded
# (sensor_msgs/Image); it is required, as the bag holds only the h264 stream.
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
set -u  # after the setup scripts, which use unset variables
CFG=$1; OUT=$2; BAG=$3; DUR=${4:-100000}; export ROS_DOMAIN_ID=${5:-70}
D=$(dirname "$0")
if [ -z "${DECODER_CMD:-}" ]; then echo "set DECODER_CMD (see the top of $0)" >&2; exit 1; fi

$DECODER_CMD > "$OUT.decoder.log" 2>&1 &
ros2 run ov_msckf run_subscribe_msckf --ros-args -r __ns:=/ov_msckf \
  -p config_path:="$CFG/estimator_config.yaml" -p verbosity:=INFO -p use_stereo:=false \
  -p max_cameras:=1 ${OV_ARGS:-} > "$OUT.ov.log" 2>&1 &
python3 "$D/vio_record.py" /ov_msckf/odomimu "$OUT" > "$OUT.rec.log" 2>&1 &
sleep 5
timeout -s INT "$DUR" ros2 bag play "$BAG" --disable-keyboard-controls \
  --topics /ircam_wide/h264 /imu_apps /alt_raw /fmu/out/vehicle_odometry > "$OUT.play.log" 2>&1
sleep 2
# Stop the ros2 run wrappers and the binaries they started (these outlive a killed wrapper).
for p in $(jobs -p); do pkill -INT -P "$p" 2>/dev/null; kill -INT "$p" 2>/dev/null; done
sleep 3
for p in $(jobs -p); do pkill -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; done
echo "done $OUT"
