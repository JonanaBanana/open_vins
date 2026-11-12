"""
Evaluate a recorded OpenVINS track (vio_record.py: <prefix>_vio.csv, <prefix>_ref.csv) GPS-free:
PX4 (GPS-aided) is used only for the initial pose (position and heading when OpenVINS starts
publishing, as an external initial pose would give) and as the reference.

    python3 ov_eval_init.py <prefix> [plot.png]
"""
import sys

import numpy as np

T_ENU_NED = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float)
R_FRD_FLU = np.diag([1.0, -1.0, -1.0])


def rot(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


a = np.genfromtxt(sys.argv[1] + '_vio.csv', delimiter=',', names=True)
r = np.genfromtxt(sys.argv[1] + '_ref.csv', delimiter=',', names=True)
a, r = a[np.argsort(a['t'])], r[np.argsort(r['t'])]
a = a[(a['t'] > r['t'][0]) & (a['t'] < r['t'][-1])]
t0 = a['t'][0]
# Initial pose: VIO yaw (IMU = base_link FLU in its global frame), PX4 yaw of FLU in ENU.
yaw_v = np.arctan2(*rot(a['qx'][0], a['qy'][0], a['qz'][0], a['qw'][0])[[1, 0], 0])
j = np.searchsorted(r['t'], t0)
Rp = T_ENU_NED @ rot(r['qx'][j], r['qy'][j], r['qz'][j], r['qw'][j]) @ R_FRD_FLU  # PX4 q as x, y, z, w
yaw_p = np.arctan2(Rp[1, 0], Rp[0, 0])
d = yaw_p - yaw_v
Rz = np.array([[np.cos(d), -np.sin(d)], [np.sin(d), np.cos(d)]])
ref0 = np.array([np.interp(t0, r['t'], r['x']), np.interp(t0, r['t'], r['y'])])
t = np.arange(t0, a['t'][-1], 0.1)
pv = np.stack([np.interp(t, a['t'], a[k]) for k in ('x', 'y')], 1)
pos = (pv - pv[0]) @ Rz.T + ref0
ref = np.stack([np.interp(t, r['t'], r[k]) for k in ('x', 'y')], 1)
refz = np.interp(t, r['t'], r['z'])
dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(ref, axis=0), axis=1))])
e = np.linalg.norm(pos - ref, axis=1)
print(f'{t[-1] - t[0]:.0f} s from OpenVINS start, {dist[-1]:.0f} m flown; initial heading offset '
      f'{np.degrees(d):+.1f} deg')
print(f'end xy error {e[-1]:.0f} m ({100 * e[-1] / dist[-1]:.1f} %), max {e.max():.0f} m')
for D in (100, 500):
    errs = []
    for i in range(0, len(dist), 5):
        k = np.searchsorted(dist, dist[i] + D)
        if k >= len(dist):
            break
        errs.append(np.linalg.norm((pos[k] - pos[i]) - (ref[k] - ref[i])) / D)
    print(f'relative xy error over {D} m: median {100 * np.median(errs):.1f} %, '
          f'90th percentile {100 * np.percentile(errs, 90):.1f} %')
# Per minute: speed ratio and heading error (from 1 s position differences, cruise only).
vv, vr = (pos[10:] - pos[:-10]), (ref[10:] - ref[:-10])
tm = t[5:-5] - t0
cr = (np.linalg.norm(vr, axis=1) > 5) & (refz[5:-5] > 20)
sr = np.linalg.norm(vv, axis=1) / np.maximum(np.linalg.norm(vr, axis=1), 1e-6)
he = np.degrees(np.angle(np.exp(1j * (np.arctan2(vv[:, 1], vv[:, 0]) - np.arctan2(vr[:, 1], vr[:, 0])))))
rows_s, rows_h = [], []
for m in range(int(tm[-1] // 60) + 1):
    k = cr & (tm >= 60 * m) & (tm < 60 * m + 60)
    rows_s.append(f'{np.median(sr[k]):.2f}' if k.sum() > 50 else ' -- ')
    rows_h.append(f'{np.median(he[k]):+.1f}' if k.sum() > 50 else ' -- ')
print('speed ratio per minute:   ', ' '.join(rows_s))
print('heading error per minute: ', ' '.join(rows_h), 'deg')
print(f'heading |error| median {np.median(np.abs(he[cr])):.1f} deg, p90 {np.percentile(np.abs(he[cr]), 90):.1f}')
if len(sys.argv) > 2:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    ax[0].plot(ref[:, 0], ref[:, 1], 'k', label='PX4 (GPS, reference)')
    ax[0].plot(pos[:, 0], pos[:, 1], 'r', label='OpenVINS from the initial pose')
    ax[0].axis('equal')
    ax[0].legend()
    ax[1].plot(tm[cr], sr[cr], ',')
    ax[1].set_ylim(0, 2)
    ax[1].set_ylabel('speed ratio (1 s)')
    ax[2].plot(t - t0, e)
    ax[2].set_ylabel('xy error [m]')
    fig.savefig(sys.argv[2], dpi=80)
