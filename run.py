"""Run the simulation, print metrics and save a plot."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ekf import simulate, run, position_rmse, nees

data = simulate()
truth, t = data["truth"], data["t"]
est, covs = run(data)
drop = (t >= 60) & (t < 80)

fix = ~np.isnan(data["gps"][:, 0])
dr_data = dict(data, gps=np.full_like(data["gps"], np.nan))
dead_reckoning, _ = run(dr_data)


def err_at(track, mask):
    return float(np.sqrt(np.mean(np.sum((track[mask] - truth[mask, :2]) ** 2, axis=1))))


print(f"Error at GPS fix times, raw GPS:          {err_at(data['gps'], fix):.2f} m")
print(f"Error at GPS fix times, EKF:              {err_at(est[:, :2], fix):.2f} m")
print(f"Error during 20 s GPS dropout, EKF:       {position_rmse(est, truth, drop):.2f} m")
print(f"Error over full drive, IMU + wheel only:  {position_rmse(dead_reckoning, truth):.1f} m")
print(f"Error over full drive, EKF:               {position_rmse(est, truth):.2f} m")
print(f"Average position NEES (ideal about 2):    {nees(est[50:], covs[50:], truth[50:]):.2f}")

# Average over 20 random drives so the numbers do not depend on one seed
g, e, dd = [], [], []
for seed in range(20):
    d = simulate(seed=seed)
    es, _ = run(d)
    m = ~np.isnan(d["gps"][:, 0])
    g.append(np.sqrt(np.mean(np.sum((d["gps"][m] - d["truth"][m, :2]) ** 2, axis=1))))
    e.append(np.sqrt(np.mean(np.sum((es[m, :2] - d["truth"][m, :2]) ** 2, axis=1))))
    dmask = (d["t"] >= 60) & (d["t"] < 80)
    dd.append(position_rmse(es, d["truth"], dmask))
print(f"20 drives: raw GPS {np.mean(g):.2f} m, EKF {np.mean(e):.2f} m at fix times; "
      f"EKF during dropout {np.mean(dd):.1f} m")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(truth[:, 0], truth[:, 1], "k", label="truth")
ax[0].scatter(data["gps"][:, 0], data["gps"][:, 1], s=8, c="tab:red", label="GPS")
ax[0].plot(est[:, 0], est[:, 1], "tab:blue", label="EKF")
ax[0].set_title("Trajectory"); ax[0].axis("equal"); ax[0].legend()
err = np.linalg.norm(est[:, :2] - truth[:, :2], axis=1)
sig = 3 * np.sqrt(covs[:, 0, 0] + covs[:, 1, 1])
ax[1].plot(t, err, label="EKF position error")
ax[1].plot(t, sig, "--", label="3-sigma bound")
ax[1].axvspan(60, 80, color="grey", alpha=0.2, label="GPS dropout")
ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("m"); ax[1].set_title("Error over time"); ax[1].legend()
plt.tight_layout()
plt.savefig("results.png", dpi=120)
