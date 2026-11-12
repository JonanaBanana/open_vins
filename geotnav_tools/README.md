# GeoTNav tools for this fork

Replaying a VOXL flight bag through OpenVINS and evaluating the result GPS-free.

- `run_replay.sh <config dir> <out prefix> <bag> [seconds] [ROS domain id]`: the ircam_wide decoder
  (`DECODER_CMD`, any node decoding `/ircam_wide/h264` into `/ircam_wide/decoded`), OpenVINS
  (`run_subscribe_msckf`) and `vio_record.py`, then `ros2 bag play`. Inputs to OpenVINS are the
  camera and `/imu_apps` only. Example:

      DECODER_CMD="ros2 run <pkg> <decoder> ..." geotnav_tools/run_replay.sh config/geotnav_ircam_wide /tmp/ov/base \
        ~/dataset/oct23/voxlbag_oct23_mission_1_timefixed

- `vio_record.py <odometry topic> <out prefix>`: records OpenVINS's odometry (`<prefix>_vio.csv`) and
  PX4's (`<prefix>_ref.csv`, ENU position; `qx..qw` hold PX4's FRD -> NED quaternion as x, y, z, w),
  both stamped with the receive time (PX4's own stamps are its boot time).
- `ov_eval_init.py <out prefix> [plot.png]`: places OpenVINS's track at PX4's position and heading
  when OpenVINS starts publishing (the initial pose an external source would give) and
  dead-reckons from there. Reports the end error, the error per 100 m and 500 m flown, and per minute
  of cruise the speed ratio (scale) and the heading error. PX4 is used for nothing else.

PX4's odometry fuses GPS: it is the reference and the initial pose only, never an input.
