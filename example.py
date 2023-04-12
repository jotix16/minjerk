import numpy as np
import matplotlib.pyplot as plt
from minjerk import (
    plotMinJerkTraj,
    get_multi_interval_minjerk_xyz,
    get_multi_interval_multi_dim_minjerk_samples,
    get_multi_interval_multi_dim_minjerk_lambda
)

if __name__ == "__main__":

    dt = 0.004
    smooth_acc = False
    smooth_start = False
    i_a_end = -1
    tt = [0.0, 0.5, 0.6, 0.8, 1.0]
    xx = np.array([[0.0, 0.0, -0.1, 0.3, 1.0], [0.0, 0.6, 0.0, 0.2, 0.0]]).T

    uu = np.array([[0.0, 3.4, -0.2, 1.1, 0.0], [0.0, 2.2, 0.0, 2.0, 0.0]]).T

    # 1. straight forward
    xxx, vvv, aaa, jjj = get_multi_interval_minjerk_xyz(
        dt, tt, xx, uu, smooth_acc, smooth_start, i_a_end=i_a_end
    )

    # 2. functional approach values
    xxx1, vvv1, aaa1, jjj1 = get_multi_interval_multi_dim_minjerk_samples(
        dt, tt, xx, uu, smooth_acc, smooth_start, i_a_end
    )

    # 3. functional approach lambdas
    ttt = np.linspace(tt[0], tt[-1], 1 + int((tt[-1] - tt[0]) / dt))
    func = get_multi_interval_multi_dim_minjerk_lambda(
        tt, xx, uu, smooth_acc, smooth_start, i_a_end
    )
    xxx2, vvv2, aaa2, jjj2 = func(ttt)

    print(np.linalg.norm(xxx2 - xxx1) / xxx1.size)
    print(np.linalg.norm(xxx - xxx1) / xxx.size)
    print(np.linalg.norm(vvv - vvv1) / vvv.size)
    print(np.linalg.norm(aaa - aaa1) / vvv.size)
    print(np.linalg.norm(jjj - jjj1) / vvv.size)

    plt.plot(ttt, aaa[:, 0], "b")
    plt.plot(ttt, aaa1[:, 0], "r")
    plt.title("Differences: straight forward vs functional approach")
    plotMinJerkTraj(
        xxx[:, 0],
        vvv[:, 0],
        aaa[:, 0],
        jjj[:, 0],
        dt,
        "straight forward approach",
        block=False,
        tt=tt,
        xx=xx[:, 0],
        uu=uu[:, 0],
    )
    plotMinJerkTraj(
        xxx1[:, 0],
        vvv1[:, 0],
        aaa1[:, 0],
        jjj1[:, 0],
        dt,
        "functional approach values",
        block=False,
        tt=tt,
        xx=xx[:, 0],
        uu=uu[:, 0],
    )
    plotMinJerkTraj(
        xxx1[:, 0],
        vvv2[:, 0],
        aaa2[:, 0],
        jjj2[:, 0],
        dt,
        "functional approach lambdas",
        block=False,
        tt=tt,
        xx=xx[:, 0],
        uu=uu[:, 0],
    )
    plt.show()
