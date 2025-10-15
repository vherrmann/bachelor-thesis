#! /usr/bin/env nix
#! nix shell /home/vherrmann/repos/-#python312With.numpy.matplotlib --command python


print(
    "In case of import errors, you probably need to build the cpp module"
    " in code/dt-lif-snn-compute-regions-depthsearch with `python setup.py build_ext --inplace`"
)

import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "../../code/dt-lif-snn-compute-regions-depthsearch/src",
    )
)

import compute_regions_cpp
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures


V = np.array([[0.0, 0.0], [0.0, 0.0]])
T = 20
i0 = np.array([0.0, 0.0])
u0 = np.array([0.0, 0.0])
b = np.array([0.0, 0.0])
theta = 1.0
alpha = 0.0
beta = 1.0
neurons_n = 2


fontsize = 17
plt.rcParams.update({"font.size": fontsize})

# Φ_{\max,T,n_0}
u0 = np.array([0.012345, 0.012345])


def run_snn(
    i0=i0,
    u0=u0,
    V=V,
    b=b,
    theta=theta,
    alpha=alpha,
    beta=beta,
    T=T,
    neurons_n=neurons_n,
):
    return compute_regions_cpp.compute_regions(
        compute_regions_cpp.SNNConfig(
            i0,
            u0,
            V,
            b,
            theta,
            alpha,
            beta,
            T,
            neurons_n,
        )
    )


def run_snnDic(args):
    return run_snn(**args)


def run_snn_parallel(argsArray):
    run_snnDicVect = np.vectorize(run_snnDic)
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:  # CPU-bound tasks
        results = list(
            executor.map(
                run_snnDicVect,
                argsArray,
            )
        )
    return np.array(results)


def change_T():
    TR = np.arange(1, 50 if prodP else 20, 1)
    data = run_snn_parallel([{"T": T} for T in TR])

    plt.figure()
    plt.plot(TR, data, label="Regions of SNN")
    run_plt(name="change_T", xlabel="T")

    plt.figure()
    plt.xlabel("T")
    draw_limit(TR=TR)
    run_plt(name="change_T_limit", xlabel="T")


def change_beta():
    betaR = np.linspace(0.0, 1.0, 1000 if prodP else 20)
    T = 20
    data = [run_snn(T=T, u0=u0, beta=beta) for beta in betaR]

    plt.figure()
    plt.xlabel("β")
    plt.plot(betaR, data, label="Regions of SNN")
    draw_limit(T=T)
    run_plt(name="change_beta", xlabel="β")


def change_alpha():
    alphaR = np.linspace(0.0, 1.0, 1000 if prodP else 20)
    T = 20
    data = run_snn_parallel([{"T": T, "u0": u0, "alpha": alpha} for alpha in alphaR])

    plt.figure()
    plt.plot(alphaR, data, label="Regions of SNN")
    draw_limit(T=T)
    run_plt(name="change_alpha", xlabel="α")


def change_beta_past_limit():
    betaR = np.linspace(1.0, 200.0, 10000 if prodP else 100)
    T = 10
    data = [run_snn(T=T, u0=u0, beta=beta) for beta in betaR]

    plt.figure()
    plt.xlabel("β")
    plt.plot(betaR, data, label="Regions of SNN")
    draw_limit(T=T)
    run_plt(name="change_beta_past_limit", xlabel="β")


def change_V3D():
    lfontsize = 14
    T = 20 if prodP else 10

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    f = lambda v1, v2: run_snn(T=T, V=np.array([[0.0, v1], [-v2, 0.0]]))
    f_vec = np.vectorize(f)

    # Create a grid
    size = 71 if prodP else 21
    xS = np.linspace(-1, 1, size)
    yS = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(xS, yS)
    # Z = np.array([[f(x, y) for x in xS] for y in yS])
    # Z = f_vec(X, Y)

    Z = run_snn_parallel(
        [
            np.array([{"T": T, "V": np.array([[0.0, x], [y, 0.0]])} for x in xS])
            for y in yS
        ]
    )

    TZ = np.full_like(X, (((T**2 + T + 2) / 2) ** 2))

    # apply coloring uniformly
    zmin = min(Z.min(), TZ.min())
    zmax = max(Z.max(), TZ.max())

    ax.plot_surface(
        X, Y, Z, cmap="plasma", vmin=zmin, vmax=zmax, label="Regions of SNN"
    )
    ax.plot_wireframe(
        X,
        Y,
        TZ,
        alpha=0.3,
        # cmap="plasma",
        # vmin=zmin,
        # vmax=zmax,
        # linestyle="--",
        # linewidth=1.0,
        label="((T² + T + 2) / 2)²",
    )

    ax.set_xlabel("$V_{1,2}$", fontsize=lfontsize)
    ax.set_ylabel("$V_{2,1}$", fontsize=lfontsize, labelpad=9)
    ax.set_zlabel("#Regions", fontsize=lfontsize, labelpad=14)
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.label.set_rotation(90)
    ax.tick_params(axis="x", labelsize=lfontsize, pad=0)
    ax.tick_params(axis="y", labelsize=lfontsize, pad=0)
    ax.tick_params(axis="z", labelsize=lfontsize, pad=7)
    ax.legend(fontsize=lfontsize)
    save_or_show_plt("change_V3D", extraArgs={"pad_inches": 0.25})

    plt.rcParams.update({"font.size": 23})

    plt.figure(figsize=(6, 5))
    plt.imshow(
        Z,
        extent=[xS.min(), xS.max(), yS.min(), yS.max()],
        vmin=zmin,
        vmax=zmax,
        origin="lower",
        cmap="plasma",
        aspect="auto",
    )
    plt.colorbar(label="#Regions")
    plt.xlabel("$V_{1,2}$")
    plt.ylabel("$V_{2,1}$")
    plt.title("")
    save_or_show_plt("change_V3D_flat")

    plt.rcParams.update({"font.size": fontsize})


def draw_limit(T=None, TR=None, alpha=1):
    if not T is None:
        plt.axhline(
            y=(((T**2 + T + 2) / 2) ** 2),
            alpha=alpha,
            color="orange",
            linestyle="--",
            label=f"((T² + T + 2) / 2)²",
        )
    if not TR is None:
        plt.plot(
            TR,
            (((TR**2 + TR + 2) / 2) ** 2),
            alpha=alpha,
            color="orange",
            linestyle="--",
            label=f"((T² + T + 2) / 2)²",
        )


showPlt = "--show" in sys.argv[1:]
prodP = "--production" in sys.argv[1:]


def save_or_show_plt(name, extraArgs={}):
    if showPlt:
        plt.show()
    else:
        plt.savefig(
            f"ch05-{name}.png", **({"bbox_inches": "tight", "dpi": 300} | extraArgs)
        )


def run_plt(name, xlabel):
    plt.title("")
    plt.xlabel(xlabel)
    plt.ylabel("#Regions")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    save_or_show_plt(name)


allP = "--all" in sys.argv[1:]

if allP or "--alpha" in sys.argv[1:]:
    print("change_alpha")
    change_alpha()

if allP or "--beta" in sys.argv[1:]:
    print("change_beta")
    change_beta()

if allP or "--V" in sys.argv[1:]:
    print("change_V")
    change_V3D()

if allP or "--T" in sys.argv[1:]:
    print("change_T")
    change_T()

if allP or "--beta_past_limit" in sys.argv[1:]:
    print("change_beta_past_limit")
    change_beta_past_limit()
