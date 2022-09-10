from json import load
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from numpy import loadtxt
import sys

"""Plot the results of the double integrator with viscosity example.
"""

file = sys.argv[1]

traj = loadtxt(file, dtype=float, delimiter=",")

fig = plt.figure(constrained_layout=True)
gs = GridSpec(3, 2, figure=fig)

ax_pos = fig.add_subplot(gs[0, :])
ax_vel = fig.add_subplot(gs[1, :])
ax_imu = fig.add_subplot(gs[2, 0])
ax_gps = fig.add_subplot(gs[2, 1])

fig.suptitle("Double Integrator w/ Viscosity")
ax_pos.plot(traj[:, 0], traj[:, 1])
ax_pos.plot(traj[:, 0], traj[:, 5])
ax_pos.legend(["real", "filtered"])
ax_pos.set_xlabel("t")
ax_pos.set_ylabel("x")
ax_pos.set_title("position")

ax_vel.plot(traj[:, 0], traj[:, 2])
ax_vel.plot(traj[:, 0], traj[:, 6])
ax_vel.legend(["real", "filtered"])
ax_vel.set_xlabel("t")
ax_vel.set_ylabel("xdot")
ax_vel.set_title("velocity")

ax_imu.plot(traj[:, 0], traj[:, 3])
ax_imu.set_xlabel("t")
ax_imu.set_ylabel("a")
ax_imu.set_title("imu measurements")

ax_gps.plot(traj[:, 0], traj[:, 4])
ax_gps.set_xlabel("t")
ax_gps.set_ylabel("x")
ax_gps.set_title("gps measurements")

plt.show()
