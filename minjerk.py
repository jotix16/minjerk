#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   MinJerk.py
@Time    :   2022/02/14
@Author  :   Mikel Zhobro
@Version :   1.0
@Contact :   zhobromikel@gmail.com
@License :   (C)Copyright 2021-2022, Mikel Zhobro
@Desc    :   Used to do min-jerk trajectory computation.
             Since any free start- or end-state puts a constraint on the constate
             the equations stay the same and only the coefficients change.
             This allows us to call _get_trajectories() to create paths of different constraints.
"""
from typing import Union, Tuple
import numpy as np


##### Helper Functions for multi-interval multi-dimensional minjerk #####
# ---------------------------------------------------------------------- #
def _steps_from_time(T, dt):
    """Method to find nr of timesteps

    Args:
        T (double): Time in seconds
        dt (double): Timestep in seconds

    Returns:
        int: Number of timesteps in T
    """
    # assert T % dt < 1e-8
    return int(T / dt) + 1  # np.arange(0,T+dt,dt)

def _curate(tt, xx, uu):
    assert len(tt) == len(xx) == len(uu)
    tt_ = []
    xx_ = []
    uu_ = []
    t_prv = -1e10
    for t,x,u in zip(tt, xx, uu):
        if not((t-t_prv) < 1e-6):
            tt_.append(t)
            xx_.append(x)
            uu_.append(u)
        t_prv = t
    assert len(tt_) > 2, "Timepoints must be at least 1e-6 appart."
    return np.stack(tt_), np.stack(xx_), np.stack(uu_)

# ---------------------------------------------------------------------- #

##### 1.Straight forward approach #####
# ---------------------------------------------------------------------- #
def get_multi_interval_minjerk_xyz(
    dt, tt, xx, uu, smooth_acc=False, smooth_start=False, i_a_end=None
):
    """Generates a multi-interval multi-dimensional minjerk trajectory

    Args:
        dt ([np.array]): (nr_intervals, Dim)
        tt ([np.array]): (nr_intervals, Dim)
        xx ([np.array]): (nr_intervals, Dim)
        uu ([np.array]): (nr_intervals, Dim)
        smooth_acc (bool, optional): Whether the accelertion between intervals should be smooth.
        i_a_end ([type], optional): If not None shows the number of the interval, whose end-acceleration should be used for the last interval.

    Returns:
        xxx, vvv, aaa, jjj: np.arrays of size (nr_intervals, Dim)
    """
    dim_ = xx.shape[1]
    xxx = [None] * dim_
    vvv = [None] * dim_
    aaa = [None] * dim_
    jjj = [None] * dim_
    for i in range(dim_):
        xxx[i], vvv[i], aaa[i], jjj[i] = _get_multi_interval_minjerk_1D(
            dt,
            tt,
            xx[:, i],
            uu[:, i],
            smooth_acc=smooth_acc,
            smooth_start=smooth_start,
            i_a_end=i_a_end,
        )
    return (
        np.stack(xxx).T,
        np.stack(vvv).T,
        np.stack(aaa).T,
        np.stack(jjj).T,
    )

def _get_multi_interval_minjerk_1D(
    dt: float,
    tt: np.ndarray,
    xx: np.ndarray,
    uu: np.ndarray,
    smooth_acc=False,
    smooth_start=False,
    i_a_end=None,
    extra_at_end=None,
):
    """Generates a multi-interval minjerk trajectory in 1 dimension

    Args:
        dt ([float]):
        tt ([np.array]): (nr_intervals, )
        xx ([np.array]): (nr_intervals, )
        uu ([np.array]): (nr_intervals, )
        smooth_acc (bool, optional): Whether the acceleartion between intervals should be smooth.
        i_a_end ([type], optional): If not None shows the number of the interval, whose start-acceleration should be used for the last interval.
        extra_at_end([type], optional): If not None shows the number of times the last value of position should be repeated

    Returns:
        [lists]: x, v, a, j
    """
    # Initialization
    T_whole = tt[-1] - tt[0]
    N_Whole = _steps_from_time(T_whole, dt)
    x_ret = np.zeros(N_Whole, dtype="double")
    v_ret = np.zeros(N_Whole, dtype="double")
    a_ret = np.zeros(N_Whole, dtype="double")
    j_ret = np.zeros(N_Whole, dtype="double")

    N = len(tt)

    t_last = tt[0]  # last end-time
    x_last = xx[0]
    u_last = uu[0]
    a_last = None if not smooth_start else 0.0
    a_ende = None
    a_end = None
    n_last = 0  # last end-index

    i = 0
    while True:
        t0 = t_last
        t1 = tt[i + 1]
        x0 = x_last
        x1 = xx[i + 1]
        u0 = u_last
        u1 = uu[i + 1]
        x,v,a,j = get_min_jerk_trajectory_samples(dt, t0, t1, x0, x1, u0, u1, a_ta=a_last, a_tb=a_ende)

        len_x = len(x)
        x_ret[n_last : n_last + len_x] = x
        v_ret[n_last : n_last + len_x] = v
        a_ret[n_last : n_last + len_x] = a
        j_ret[n_last : n_last + len_x] = j

        t_last = t0 + (len_x - 1) * dt
        n_last += (
            len_x - 1
        )  # (since last end-value == new first-value we overlay them and take only 1)
        x_last = x[-1]
        u_last = v[-1]
        a_last = a[-1] if smooth_acc else None
        if i_a_end is not None and i == i_a_end:
            a_end = a[0]
        if i == N - 3:
            a_ende = a_end
        i += 1
        if i == N - 1: break
    x_ret[n_last:] = x[-1]
    v_ret[n_last:] = v[-1]
    a_ret[n_last:] = a[-1]
    j_ret[n_last:] = j[-1]

    if extra_at_end is not None:
        repeat = [1] * N_Whole
        repeat[-1] = extra_at_end
        x_ret = np.hstack((x_ret, x_ret[1:extra_at_end]))
        v_ret = np.hstack((v_ret, v_ret[1:extra_at_end]))
        a_ret = np.hstack((a_ret, a_ret[1:extra_at_end]))
        j_ret = np.hstack((j_ret, j_ret[1:extra_at_end]))

    return x_ret, v_ret, a_ret, j_ret

# ---------------------------------------------------------------------- #

##### 2.Functional approach #####
# ---------------------------------------------------------------------- #
def get_multi_interval_multi_dim_minjerk_samples(
    dt: Union[float, int],
    tt: Union[list[int], list[float]],
    xx: np.ndarray,
    uu: np.ndarray,
    smooth_acc=False,
    smooth_start=False,
    i_a_end: Union[int, None]=None,
):
    """Generates a multi-interval multi-dimensional minjerk trajectory
      The advantage of this function is that it is able to return conditional functions which can then be evaluated
      at arbitrary times or time intervals.

    Args:
      dt ([list]): float
      tt ([np.array]): (nr_intervals,)
      xx ([np.array]): (nr_intervals, Dim)
      uu ([np.array]): (nr_intervals, Dim)
      lambdas (bool, optional): Whether to return lambdas . Defaults to False.
      i_a_end ([type], optional): If not None shows the number of the interval, whose end-acceleration should be used for the last interval.

    Returns:
      [np.array]: (4, N_times, Dims) where 4 are position, velocity, acceleration and jerk
      """
    ttt = np.linspace(tt[0], tt[-1], 1 + int((tt[-1] - tt[0]) / dt))
    tmp = get_multi_interval_multi_dim_minjerk_lambda(tt, xx, uu, smooth_acc, smooth_start, i_a_end)
    return tmp(ttt)

def get_multi_interval_multi_dim_minjerk_lambda(
    tt: Union[list[int], list[float]],
    xx: np.ndarray,
    uu: np.ndarray,
    smooth_acc=False,
    smooth_start=False,
    i_a_end: Union[int, None]=None,
):
    """Generates a multi-interval multi-dimensional minjerk trajectory
      The advantage of this function is that it is able to return conditional functions which can then be evaluated
      at arbitrary times or time intervals.

    Args:
      dt ([list]): float
      tt ([np.array]): (nr_intervals,)
      xx ([np.array]): (nr_intervals, Dim)
      uu ([np.array]): (nr_intervals, Dim)
      lambdas (bool, optional): Whether to return lambdas . Defaults to False.
      i_a_end ([type], optional): If not None shows the number of the interval, whose end-acceleration should be used for the last interval.

    Returns:
      [np.array]: (4, N_times, Dims) where 4 are position, velocity, acceleration and jerk
      or
      [lambda]:  tmp(t) -> np.array(4, N_times, Dims) where 4 are x(t), v(t), a(t) and j(t)

    """
    tt_, xx_, uu_= _curate(tt, xx, uu)
    D = xx_.shape[1]
    N = len(tt_)  # nr of minjerk intervals
    xxs = np.array([None] * N)  # (N, )

    a_last = [None if not smooth_start else 0.0] * D
    a_ende = [None] * D
    a_end = None
    for i, (tb, xb, ub) in enumerate(zip(tt_[1:], xx_[1:], uu_[1:])):
        xxs[i] = _get_multi_dim_minjerk(
            tt_[i], tb, xx_[i], xb, uu_[i], ub, a_ta=a_last, a_tb=a_ende
        )
        # smooth_acc
        a_last = xxs[i](tb)[2].squeeze() if smooth_acc else [None] * D
        # i_a_end
        if i_a_end is not None and i == i_a_end:
            a_end = xxs[i](tt_[i])[2].squeeze()
        if i == N - 3:
            a_ende = a_end

    def tmp(t):
        t = np.asarray(t).reshape(-1, 1)
        ret = np.zeros((4, len(t), D))

        for i, _t in enumerate(tt_[1:]):
            mask = (tt_[i] <= t) & ((t <= _t) if i == len(tt_) - 2 else (t < _t))
            ret[:, mask.nonzero()[0], :] = xxs[i](t[mask])
        return ret  # [:,:,:,np.newaxis]

    return tmp

def _get_multi_dim_minjerk(
    ta, tb, x_ta, x_tb, u_ta, u_tb, a_ta=None, a_tb=None
):
    """Generates a multi-dimensional minjerk trajectory

    Args:
        dt ([float]):
        ta, tb ([float]): start and end time
        x_ta, x_tb ([np.array]): (Dim, ), start and end position
        u_ta, u_tb ([np.array]): (Dim, ), start and end velocity
        a_ta, a_tb ([np.array]): (Dim, ), start and end acceleration. Defaults to None.
        lambdas (bool, optional): Whether to return lambdas . Defaults to False.

    Returns:
        [np.array]: (4, N_times, Dims) where 4 are position, velocity, acceleration and jerk
        or
        [lambda]:  tmp(t) -> np.array(4, N_times, Dims) where 4 are x(t), v(t), a(t) and j(t)
    """
    a_ta = [None] * len(x_ta) if a_ta is None else a_ta
    a_tb = [None] * len(x_ta) if a_tb is None else a_tb
    xxs = [
        get_min_jerk_trajectory_lambdas(
            ta,
            tb,
            x_ta[i],
            x_tb[i],
            u_ta[i],
            u_tb[i],
            a_ta=a_ta[i],
            a_tb=a_tb[i],
        )
        for i in range(len(x_ta))
    ]

    def tmp(t):
        t = np.asarray(t).reshape(-1, 1)
        ret = np.zeros((4, len(t), len(xxs)))
        for i, xx in enumerate(xxs):
            ret[:, :, i : i + 1] = xx(t)
        return ret

    return tmp

# ---------------------------------------------------------------------- #


##### Implementation for 1D case #####
# ---------------------------------------------------------------------- #
def get_min_jerk_trajectory_lambdas(
    ta, tb, x_ta, x_tb, u_ta, u_tb, a_ta=None, a_tb=None
):
    c1, c2, c3, c4, c5, c6 = _get_min_jerk_coefs(ta, tb, x_ta, x_tb, u_ta, u_tb, a_ta, a_tb)
    return lambda t: _get_trajectories(t - ta, c1, c2, c3, c4, c5, c6)

def get_min_jerk_trajectory_samples(
    dt: float, ta: float, tb: float, x_ta: float, x_tb: float, u_ta: float, u_tb: float, a_ta=None, a_tb=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c1, c2, c3, c4, c5, c6 = _get_min_jerk_coefs(ta, tb, x_ta, x_tb, u_ta, u_tb, a_ta, a_tb)
    # Trajectory values ta->tb
    N_Whole = _steps_from_time(tb-ta, dt)
    t = np.linspace(ta, tb, N_Whole) - ta
    x, v, a, j = _get_trajectories(t, c1, c2, c3, c4, c5, c6)
    return x, v, a, j

def _get_min_jerk_coefs(
    ta: float, tb: float, x_ta: float, x_tb: float, u_ta: float, u_tb: float, a_ta=None, a_tb=None, lambdas=False
):
    """Generates a minjerk trajectory with set or free start and end conditions.

    Args:
        dt ([float]): timestep
        ta ([float]): start time of the interval
        tb ([float]): end time of the interval
        a: is set to [] if start and end acceleration are free
        x_ta, u_ta, (optional: a_ta): conditions at t=ta
        x_tb, u_tb, (optional: a_tb): conditions at t=tb
    Returns:
        xp_des(t) = [x(t)       u(t)         a(t)            u(t)]
                  = [position   velocity     acceleration    jerk]
    """
    # Get polynom parameters for different conditions
    T = tb - ta
    if a_ta is not None:
        # 1. set start acceleration
        if a_tb is not None:
            # a. set end acceleration
            c1, c2, c3, c4, c5, c6 = _set_start_acceleration(
                T, x_ta, x_tb, u_ta, u_tb, a_ta, a_tb
            )
        else:
            # b.free end acceleration
            c1, c2, c3, c4, c5, c6 = _set_start_acceleration(
                T, x_ta, x_tb, u_ta, u_tb, a_ta
            )
    else:
        # 2. free start acceleration
        if a_tb is not None:
            # a. set end acceleration
            c1, c2, c3, c4, c5, c6 = _free_start_acceleration(
                T, x_ta, x_tb, u_ta, u_tb, a_tb
            )
        else:
            # b.free end acceleration
            c1, c2, c3, c4, c5, c6 = _free_start_acceleration(T, x_ta, x_tb, u_ta, u_tb)

    return c1, c2, c3, c4, c5, c6


# Get values from polynom parameters
def _get_trajectories(t, c1, c2, c3, c4, c5, c6):
    """Given 5th order polynomial coeficients it returns values corresponing to timesteps t."""
    t_5 = t**5
    t_4 = t**4
    t_3 = t**3
    t_2 = t**2
    j = c1 * t_2 / 2 - c2 * t + c3  # jerk
    a = c1 * t_3 / 6 - c2 * t_2 / 2 + c3 * t + c4  # acceleration
    v = c1 * t_4 / 24 - c2 * t_3 / 6 + c3 * t_2 / 2 + c4 * t + c5  # velocity
    x = (
        c1 * t_5 / 120 - c2 * t_4 / 24 + c3 * t_3 / 6 + c4 * t_2 / 2 + c5 * t + c6
    )  # position
    return x, v, a, j


# 1) Acceleration is set at t=0 (a(0)=a0 => c4=a0)
def _set_start_acceleration(T: Union[float, int], x0: float, xT: float, u0: float, uT: float, a0: float, aT=None):
    M: np.ndarray
    T_5 = T**5
    T_4 = T**4
    T_3 = T**3
    T_2 = T**2
    if aT is None:
        # free end acceleration u(T)=0
        M = np.array(
            [
                [320 / T_5, -120 / T_4, -20 / (3 * T_2)],
                [200 / T_4, -72 / T_3, -8 / (3 * T)],
                [40 / T_3, -12 / T_2, -1.0 / 3.0],
            ]
        )
        c = np.array([-(a0 * T_2) / 2 - u0 * T - x0 + xT, uT - u0 - T * a0, 0])
    else:
        # set end acceleration a(T)=aT
        M = np.array(
            [
                [720 / T_5, -360 / T_4, 60 / T_3],
                [360 / T_4, -168 / T_3, 24 / T_2],
                [60 / T_3, -24 / T_2, 3 / T],
            ]
        )
        c = np.array([xT - x0 - T * u0 - (a0 * T_2) / 2, uT - u0 - T * a0, aT - a0])

    c123 = M.dot(c.T)
    c1: float = c123[0]
    c2: float = c123[1]
    c3: float = c123[2]
    c4: float = a0
    c5: float = u0
    c6: float = x0
    return c1, c2, c3, c4, c5, c6


# 2) Acceleration is free at t=0 (u(0)=0 => c3=0)
def _free_start_acceleration(T: Union[int, float], x0: float, xT: float, u0: float, uT: float, aT=None):
    T_5 = T**5
    T_4 = T**4
    T_3 = T**3
    T_2 = T**2
    if aT is None:
        # free end acceleration u(T)=0
        M = np.array(
            [
                [120 / T_5, -60 / T_4, -5 / T_2],
                [60 / T_4, -30 / T_3, -3 / (2 * T)],
                [5 / T_2, -3 / (2 * T), -T / 24],
            ]
        )
        c = np.array([xT - x0 - T * u0, uT - u0, 0])
    else:
        # set end acceleration a(T)=aT
        M = np.array(
            [
                [320 / T_5, -200 / T_4, 40 / T_3],
                [120 / T_4, -72 / T_3, 12 / T_2],
                [20 / (3 * T_2), -8 / (3 * T), 1.0 / 3.0],
            ]
        )
        c = np.array([xT - x0 - T * u0, uT - u0, aT])

    c123 = M.dot(c.T)
    c1: float = c123[0]
    c2: float = c123[1]
    c4: float = c123[2]
    c3: float = 0.
    c5: float = u0
    c6: float = x0
    return c1, c2, c3, c4, c5, c6

# ---------------------------------------------------------------------- #

##### Plotting functions #####
# ---------------------------------------------------------------------- #
def plotMinJerkTraj(
    x,
    v,
    a,
    j,
    dt,
    title,
    intervals=None,
    colors=None,
    tt=None,
    xx=None,
    uu=None,
    block=True,
):
    """Plots the x,v,a,j trajectories together with possible intervals and colors

    Args:
        x ([List(double)]): position vector
        v ([List(double)]): velocity vector
        a ([List(double)]): acceleration vector
        j ([List(double)]): jerk vector
        dt ([double]): time step
        title ([String]): tittle of the plot
        intervals ([set((a,b))], optional): {(0.1, 0.2), (0.42,0.55), ..}
        colors ([tuple], optional): ('gray', 'blue', ..)
    """
    import matplotlib.pyplot as plt

    fsize = 12
    if colors is None:
        colors = []
    fig, axs = plt.subplots(4, 1, figsize=(16, 11))
    timesteps = np.arange(0, x.shape[0]) * dt  # (1:length(x))*dt
    for ax in axs:
        ax.grid(True)
        ax.set_xlim(xmin=0, xmax=timesteps[-1])
    axs[0].plot(timesteps, x, label="Position")
    axs[0].legend(loc=1, fontsize=fsize)
    axs[1].plot(timesteps, v, label="Velocity")
    axs[1].legend(loc=1, fontsize=fsize)
    axs[2].plot(timesteps, a, label="Acceleration")
    axs[2].legend(loc=1, fontsize=fsize)
    axs[3].plot(timesteps, j, label="Jerk")
    axs[3].legend(loc=1, fontsize=fsize)
    if intervals is not None:
        for ax in axs:
            ax = _plot_intervals(ax, intervals, dt, colors)

    if tt is not None:
        if xx is not None:
            _plot_lines_coord(axs[0], tt, xx, typ=None)
        else:
            for t in tt:
                axs[0].axvline(t, linestyle="--")
        if uu is not None:
            _plot_lines_coord(axs[1], tt, uu, typ=None)
        else:
            for t in tt:
                axs[1].axvline(t, linestyle="--")
        for ax in axs[2:]:
            for t in tt:
                ax.axvline(t, linestyle="--")
    fig.suptitle(title)
    plt.show(block=block)

def _plot_lines_coord(ax, tt, xx, typ=None):
    einheit = ["m", r"$\frac{m}{s}$", r"$\frac{m}{s^2}$"]
    ei = einheit[typ] if typ is not None else ""
    assert len(tt) == len(xx)
    # Draw lines connecting points to axes
    ax.scatter(tt, xx)
    for t, x in zip(tt, xx):
        txt = r"({}s, {} {})".format(t, x, ei)
        ax.text(t, x, txt, size=9, color="k", zorder=10, weight="normal")


def _plot_intervals(ax, intervals, dt, colors=None):
    if colors is None or len(colors) != len(intervals):
        colors = np.repeat("gray", len(intervals))  # #2ca02c
    for i, col in zip(intervals, colors):
        ax.axvspan(dt * i[0], dt * i[1], facecolor=col, alpha=0.3)
    return ax

# ---------------------------------------------------------------------- #
