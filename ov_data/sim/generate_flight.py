#!/usr/bin/env python3
"""
Generate a fixed-altitude flight trajectory for the OpenVINS simulator (sim_traj_path).

The path is a horizontal Lissajous figure-eight at a given height above z = 0, with a slow height
oscillation. The IMU frame is x forward, y left, z up (FLU) in coordinated flight: z along the specific
force (acceleration minus gravity), x along the velocity made perpendicular to it. So the vehicle banks in
turns and pitches when climbing, as a multirotor or airplane in steady flight would.

Output format (as the other files here): timestamp tx ty tz qx qy qz qw, with p_IinG and the JPL quaternion
q_GtoI (numerically the Hamilton q_ItoG).

    python3 generate_flight.py out.txt [--height 100] [--size 600] [--period 300] [--duration 300]
                                       [--hover-at 100 --hover-for 20 --hover-ramp 5]

With --hover-for, the vehicle slows smoothly to a stop at --hover-at (over --hover-ramp seconds), holds its
position and heading for --hover-for seconds, and speeds up again along the same path.
"""
import argparse

import numpy as np

G = 9.81


def rot_to_quat_xyzw(R):
    """Hamilton quaternion (x, y, z, w) of a rotation matrix."""
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.zeros(4)
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
        x, y, z, w = q
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('--height', type=float, default=100.0, help='mean height above z = 0 [m]')
    ap.add_argument('--height-amp', type=float, default=5.0, help='height oscillation amplitude [m]')
    ap.add_argument('--height-period', type=float, default=60.0, help='height oscillation period [s]')
    ap.add_argument('--size', type=float, default=600.0, help='half-width of the figure-eight along x [m]')
    ap.add_argument('--period', type=float, default=300.0, help='time for one figure-eight [s]')
    ap.add_argument('--duration', type=float, default=300.0, help='length of the trajectory [s]')
    ap.add_argument('--rate', type=float, default=20.0, help='poses per second')
    ap.add_argument('--t0', type=float, default=1000.0, help='first timestamp [s]')
    ap.add_argument('--hover-at', type=float, default=100.0, help='time the hover starts [s]')
    ap.add_argument('--hover-for', type=float, default=0.0, help='hover duration [s] (0: no hover)')
    ap.add_argument('--hover-ramp', type=float, default=5.0, help='slow-down and speed-up time [s]')
    a = ap.parse_args()

    w = 2 * np.pi / a.period
    wh = 2 * np.pi / a.height_period
    A, B = a.size, a.size / 2
    t = np.arange(0.0, a.duration + 1e-9, 1.0 / a.rate)

    # Progress along the path: tau(t) = integral of a speed factor f in [0, 1], which ramps smoothly (cosine) to 0
    # for the hover and back to 1. Velocity and acceleration follow by the chain rule.
    def factor(tt):
        f = np.ones_like(tt)
        if a.hover_for > 0:
            r, h0, h1 = a.hover_ramp, a.hover_at, a.hover_at + a.hover_for
            down = (tt > h0 - r) & (tt < h0)
            f[down] = 0.5 * (1 + np.cos(np.pi * (tt[down] - (h0 - r)) / r))
            f[(tt >= h0) & (tt <= h1)] = 0.0
            up = (tt > h1) & (tt < h1 + r)
            f[up] = 0.5 * (1 - np.cos(np.pi * (tt[up] - h1) / r))
        return f
    fine = np.arange(0.0, a.duration + 1.0, 1e-3)
    tau = np.interp(t, fine, np.concatenate([[0], np.cumsum(factor(fine)[:-1]) * 1e-3]))
    f = factor(t)
    df = np.gradient(factor(fine), 1e-3)
    df = np.interp(t, fine, df)

    # Path (as a function of tau) and its first two derivatives
    P = np.stack([A * np.sin(w * tau), B * np.sin(2 * w * tau), a.height + a.height_amp * np.sin(wh * tau)], 1)
    dP = np.stack([A * w * np.cos(w * tau), 2 * B * w * np.cos(2 * w * tau), a.height_amp * wh * np.cos(wh * tau)], 1)
    ddP = np.stack([-A * w**2 * np.sin(w * tau), -4 * B * w**2 * np.sin(2 * w * tau), -a.height_amp * wh**2 * np.sin(wh * tau)], 1)
    p = P
    v = dP * f[:, None]
    acc = ddP * (f**2)[:, None] + dP * df[:, None]

    with open(a.out, 'w') as f:
        f.write('# format: timestamp tx ty tz qx qy qz qw\n')
        for i in range(len(t)):
            z_b = acc[i] + np.array([0.0, 0.0, G])
            z_b /= np.linalg.norm(z_b)
            heading = v[i] if np.linalg.norm(v[i]) > 0.5 else dP[i]  # keep the path's heading while hovering
            x_b = heading - heading.dot(z_b) * z_b
            x_b /= np.linalg.norm(x_b)
            R_ItoG = np.stack([x_b, np.cross(z_b, x_b), z_b], 1)
            q = rot_to_quat_xyzw(R_ItoG)
            f.write(f'{a.t0 + t[i]:.6f} {p[i, 0]:.6f} {p[i, 1]:.6f} {p[i, 2]:.6f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n')

    speed = np.linalg.norm(v, axis=1)
    bank = np.degrees(np.arccos(np.clip([(acc[i] + [0, 0, G])[2] / np.linalg.norm(acc[i] + [0, 0, G]) for i in range(len(t))], -1, 1)))
    print(f'{a.out}: {len(t)} poses, {a.duration:.0f} s, speed {speed.min():.1f}-{speed.max():.1f} m/s, '
          f'height {p[:, 2].min():.0f}-{p[:, 2].max():.0f} m, tilt up to {bank.max():.1f} deg')


if __name__ == '__main__':
    main()
