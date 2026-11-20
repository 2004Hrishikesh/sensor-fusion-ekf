"""Extended Kalman filter that fuses GPS, IMU and wheel-speed data to track a car in 2D.

State x = [px, py, heading, speed].
Inputs from the IMU: yaw rate (rad/s) and forward acceleration (m/s^2).
Measurements: GPS position (1 Hz) and wheel speed (10 Hz).
"""
import numpy as np

DT = 0.1  # 10 Hz


def wrap(a):
    """Keep an angle between -pi and pi."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def simulate(seconds=120, seed=0, gps_std=3.0, dropout=(60.0, 80.0)):
    """Make a ground-truth drive plus noisy sensor readings."""
    rng = np.random.default_rng(seed)
    n = int(seconds / DT)
    t = np.arange(n) * DT
    yaw_rate = 0.15 * np.sin(0.05 * t)        # gentle S-curves
    accel = 0.3 * np.sin(0.1 * t)
    truth = np.zeros((n, 4))
    truth[0] = [0.0, 0.0, 0.0, 10.0]
    for k in range(1, n):
        px, py, th, v = truth[k - 1]
        truth[k] = [px + v * np.cos(th) * DT, py + v * np.sin(th) * DT,
                    wrap(th + yaw_rate[k - 1] * DT), max(v + accel[k - 1] * DT, 0.0)]

    imu = np.column_stack([yaw_rate + rng.normal(0, 0.02, n) + 0.01,  # noise plus a small gyro bias
                           accel + rng.normal(0, 0.2, n)])
    wheel = truth[:, 3] + rng.normal(0, 0.3, n)
    gps = np.full((n, 2), np.nan)
    every = int(1.0 / DT)
    for k in range(0, n, every):
        if not (dropout and dropout[0] <= t[k] < dropout[1]):
            gps[k] = truth[k, :2] + rng.normal(0, gps_std, 2)
    return dict(t=t, truth=truth, imu=imu, wheel=wheel, gps=gps, gps_std=gps_std)


class EKF:
    def __init__(self, x0, P0, gps_std=3.0, wheel_std=0.3):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.Q = np.diag([0.05, 0.05, 0.02, 0.2]) ** 2 * DT  # process noise
        self.R_gps = np.eye(2) * gps_std ** 2
        self.R_wheel = np.array([[wheel_std ** 2]])

    def predict(self, yaw_rate, accel):
        px, py, th, v = self.x
        self.x = np.array([px + v * np.cos(th) * DT, py + v * np.sin(th) * DT,
                           wrap(th + yaw_rate * DT), v + accel * DT])
        # Jacobian of the motion model with respect to the state
        F = np.array([[1, 0, -v * np.sin(th) * DT, np.cos(th) * DT],
                      [0, 1, v * np.cos(th) * DT, np.sin(th) * DT],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        self.P = F @ self.P @ F.T + self.Q

    def _update(self, z, H, R):
        y = z - H @ self.x                      # innovation
        S = H @ self.P @ H.T + R                # innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)     # Kalman gain
        self.x = self.x + K @ y
        self.x[2] = wrap(self.x[2])
        self.P = (np.eye(4) - K @ H) @ self.P

    def update_gps(self, z):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self._update(np.asarray(z), H, self.R_gps)

    def update_wheel(self, speed):
        H = np.array([[0, 0, 0, 1]])
        self._update(np.array([speed]), H, self.R_wheel)


def run(data, use_imu=True, use_wheel=True):
    """Run the filter over a drive. Returns estimates and covariances."""
    n = len(data["t"])
    ekf = EKF([0, 0, 0, 10], np.diag([5.0, 5.0, 0.1, 1.0]), gps_std=data["gps_std"])
    est, covs = np.zeros((n, 4)), np.zeros((n, 4, 4))
    for k in range(n):
        if k > 0:
            yr, a = data["imu"][k - 1] if use_imu else (0.0, 0.0)
            ekf.predict(yr, a)
        if use_wheel:
            ekf.update_wheel(data["wheel"][k])
        if not np.isnan(data["gps"][k, 0]):
            ekf.update_gps(data["gps"][k])
        est[k], covs[k] = ekf.x, ekf.P
    return est, covs


def position_rmse(est, truth, mask=None):
    err = np.linalg.norm(est[:, :2] - truth[:, :2], axis=1)
    if mask is not None:
        err = err[mask]
    return float(np.sqrt(np.mean(err ** 2)))


def gps_only_track(data):
    """Baseline: hold the last GPS fix until the next one arrives."""
    track = np.zeros((len(data["t"]), 2))
    last = data["truth"][0, :2]
    for k, z in enumerate(data["gps"]):
        if not np.isnan(z[0]):
            last = z
        track[k] = last
    return track


def nees(est, covs, truth):
    """Average normalized estimation error squared for position (about 2 if the filter is consistent)."""
    vals = []
    for e, P, x in zip(est, covs, truth):
        d = e[:2] - x[:2]
        vals.append(d @ np.linalg.inv(P[:2, :2]) @ d)
    return float(np.mean(vals))
