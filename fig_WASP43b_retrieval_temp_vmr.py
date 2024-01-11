import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import mc3

import legend_handler as lh


def main(retrieval='free'):
    """
    This Python script reproduces Figure 4 and Extended-data Figure 8.

    Requirements
    ------------
    For better results use these versions (older version might still work):
        Python >= 3.6
        numpy >= 1.25
        matplotlib >= 3.7.2
        mc3 >= 3.1.3

    Usage
    -----
    Run from the command line from the folder containing this file:

    For Figure 4, run:
    $ python fig_WASP43b_retrieval_temp_vmr.py free
    For ED Figure 8, run:
    $ python fig_WASP43b_retrieval_temp_vmr.py equilibrium
    """
    if retrieval == 'free':
        model_mask = [True, False, True, False, True]
    elif retrieval == 'equilibrium':
        model_mask = [False, True, False, True, False]
    else:
        raise ValueError(
            "Must select a valid retrieval type, either 'free' or 'equilibrium'"
        )
    figname = f'plots/MIRI_WASP43b_{retrieval}_retrieval_temperature_vmr.pdf'

    molecs = 'H2O CH4'.split()
    nmol = len(molecs)

    models = np.array([
        'hydra',
        'scarlet',
        'pyratbay',
        'platon',
        'nemesis',
    ])
    nmodels = len(models)

    obs_phase = np.array([0.0, 0.25, 0.5, 0.75])
    nphase = len(obs_phase)

    # Read VMR posterior data:
    vmr_posteriors = np.zeros((nmodels, nphase), dtype=object)
    for i,model in enumerate(models):
        print(f'Reading {model} posterior')
        data = np.load(f'data/vmr_posterior_{model}.npz')
        for j,phase in enumerate(obs_phase):
            vmr_posterior = data[f'vmr_posterior_phase_{phase:.2f}']
            vmr_posteriors[i,j] = mc3.plots.Posterior(
                vmr_posterior, pnames=molecs, quantile=0.68,
            )

    # Read temperature posterior profiles:
    temp_posteriors = []
    pressures = []
    for i,model in enumerate(models):
        data = np.load(f'data/temperature_posterior_{model}.npz')
        temp_posteriors.append(data['temp_posteriors'])
        pressures.append(data['pressure'])


    # Read GCM models data:
    with np.load('data/gcm_data.npz') as d:
        gcm_pressure = d['gcm_pressure']
        gcm_temperature = d['gcm_temperature']
        gcm_vmr_CH4_equil = d['gcm_vmr_CH4_equil']
        gcm_vmr_H2O_equil = d['gcm_vmr_H2O_equil']
        gcm_vmr_CH4_diseq = d['gcm_vmr_CH4_diseq']
        gcm_vmr_H2O_diseq = d['gcm_vmr_H2O_diseq']

    # Pressure boundaries probed by the observations:
    probed_press_bounds = np.array([
        (3.0, 6.0e-3),
        (0.5, 1.0e-3),
        (0.7, 3.0e-4),
        (0.3, 1.0e-3),
    ])

    pmasks = [
        (gcm_pressure<p[0]) & (gcm_pressure>p[1])
        for p in probed_press_bounds
    ]
    probed_press_mean = np.mean(probed_press_bounds, axis=1)
    probed_press_span = np.array(probed_press_bounds)[:,0] - probed_press_mean

    # VMR values probed within pressure bounds:
    nlayers = len(gcm_pressure)
    nchem = 2  # equilibrium or disequilibrium chemistry
    gcm_vmr = np.zeros((nmol, nchem, nphase, nlayers))
    gcm_vmr[0,0] = np.log10(gcm_vmr_H2O_equil)
    gcm_vmr[1,0] = np.log10(gcm_vmr_CH4_equil)
    gcm_vmr[0,1] = np.log10(gcm_vmr_H2O_diseq)
    gcm_vmr[1,1] = np.log10(gcm_vmr_CH4_diseq)

    gcm_vmr_mean = np.zeros((nmol, nchem, nphase))
    gcm_vmr_span = np.zeros((nmol, nchem, nphase))
    for j in range(nphase):
        pmask = pmasks[j]
        vmr_min = np.amin(gcm_vmr[:,:,j,pmask], axis=2)
        vmr_max = np.amax(gcm_vmr[:,:,j,pmask], axis=2)
        gcm_vmr_mean[:,:,j] = 0.5*(vmr_max+vmr_min)
        gcm_vmr_span[:,:,j] = 0.5*(vmr_max-vmr_min)
    # Set minimum span to 0.1 dex for a clearer plot
    gcm_vmr_span = np.clip(gcm_vmr_span, 0.1, np.inf)


    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Plot setup:
    rcParams['font.family'] = ['sans-serif']
    rcParams['font.sans-serif'] = ['Arial']
    rcParams['figure.dpi'] = 150.0
    rcParams['savefig.dpi'] = 300.0

    nx = 2
    ny = nphase//nx

    phase_labels = [
        'phase = 0.0 \nnight-side',
        'phase = 0.25\nevening terminator',
        'phase = 0.5 \nday-side',
        'phase = 0.75\nmorning terminator',
    ]

    labels = [
        'phase = 0.0',
        'phase = 0.25',
        'phase = 0.5',
        'phase = 0.75',
    ]
    letters = 'a) b) c)'.split()

    pmin = 2.0e-5
    pmax = 1e1

    # N-max models:
    gcm_cols = ['xkcd:green', 'cornflowerblue']
    gcm_labs = [
        'equilibrium chemistry',
        'disequilibrium chemistry',
    ]
    cols = np.array([plt.cm.plasma(0.95*k/(nmodels-1)) for k in range(nmodels)])
    edge_cols = np.ones_like(cols)
    for i in range(nmodels):
        edge_cols[i,0:3] = mc3.plots.alphatize(cols[i], 0.85, 'black')
    # Color-alpha
    cols[:,3] = 0.325
    edge_cols[:,3] = 0.55

    ranges = [
        (-10.0, -0.5),
        (-10.0, -0.5),
    ]

    handler_map = {
        model: lh.Handler([cols[j], edge_cols[j]], skew=(2+j//2)%3)
        for j,model in enumerate(models)
        if model_mask[j]
    }

    fontsize = 7.0
    dx = 0.19
    dy = 0.17
    ymin = 0.12
    ymax = 0.92


    fig = plt.figure(43)
    fig.set_size_inches(7.086, 2.7)
    plt.clf()
    for m in range(nmol):
        for j in range(nphase):
            ax = plt.axes([0.59+(dx+0.02)*m, ymin+(dy)*(3-j), dx, dy])
            ax.tick_params(which='both', direction='in', labelsize=fontsize-0.5)
            if j == 0:
                ax.text(
                    0.0, 1.04, letters[m+1], fontsize=fontsize,
                    weight='bold', va='bottom', ha='left', transform=ax.transAxes,
                )
            if m == 0:
                ax.text(
                    0.015, 0.95, labels[j], fontsize=fontsize,
                    va='top', ha='left', transform=ax.transAxes,
                )
            for i in range(nmodels):
                post = vmr_posteriors[i][j]
                if not model_mask[i]:
                    continue
                pdf = 0.9*post.pdf[m] / np.amax(post.pdf[m])
                x_shade = (
                    (post.xpdf[m] >= post.low_bounds[m]) &
                    (post.xpdf[m] <= post.high_bounds[m])
                )
                ax.plot(post.xpdf[m], pdf, color=edge_cols[i], lw=1.)
                ax.fill_between(
                    post.xpdf[m], 0, pdf*x_shade, facecolor=cols[i],
                    edgecolor='none', interpolate=False, zorder=-2,
                )
            ax.set_yticks([])
            ax.set_ylim(0, 1.3)
            ax.set_xticks(np.arange(-10, 0, 1), minor=True)
            ax.set_xticks([-9, -5, -1])
            ax.set_xlim(ranges[m])
            if j+1 < nphase:
                ax.set_xticklabels([])
            for k in range(2):
                ax.errorbar(
                    gcm_vmr_mean[m,k,j], 1.19-k*0.14,
                    xerr=gcm_vmr_span[m,k,j],
                    lw=1.25, capsize=1.8,
                    label=gcm_labs[k],
                    color=gcm_cols[k],
                )
            if m == 0 and j == 0:
                ax.legend(
                    models[model_mask], models[model_mask],
                    handler_map=handler_map,
                    loc=(0.25, 1.15), fontsize=fontsize, labelspacing=0.25,
                )
            if m == 1 and j == 0:
                ax.legend(loc=(-0.25, 1.35), fontsize=fontsize, handlelength=0.9)
        ax.set_xlabel(fr'$\log\ X_{{\rm {molecs[m]} }}$', fontsize=fontsize)

    # temperature profiles
    for i in range(nphase):
        rect = [0.055, ymin, 0.57, ymax]
        ax = mc3.plots.subplot(
            rect, margin=0.004, pos=i+1, nx=nx, ny=ny, ymargin=0.01,
        )
        ax.set_yscale('log')
        ax.set_xticks(np.arange(0, 4000, 250.0), minor=True)
        ax.tick_params(labelsize=fontsize-0.5, direction='in', which='both')
        ax.set_ylabel('Pressure (bar)', fontsize=fontsize)
        ax.set_xlabel('Temperature (K)', fontsize=fontsize)
        for k in range(nmodels):
            if not model_mask[k]:
                continue
            temp_median, temp_low, temp_high = temp_posteriors[k][i]
            ax.fill_betweenx(
                pressures[k], temp_low, temp_high,
                facecolor=cols[k], edgecolor=edge_cols[k],
                lw=1.0, label=models[k],
            )
        ax.plot(gcm_temperature[i], gcm_pressure, lw=1.25, color='black')
        ax.errorbar(
            300.0, probed_press_mean[i], probed_press_span[i],
            lw=1.25, capsize=1.8, color='0.05',
        )
        ax.text(
            0.75, 0.96, phase_labels[i], fontsize=fontsize,
            va='top', ha='center', transform=ax.transAxes,
        )
        if i//2 < 1:
            ax.set_xticklabels([])
        if i%2 == 1:
            ax.set_yticklabels([])
            ax.set_ylabel('')
        if i == 0:
            ax.text(
                0.01, 1.04, letters[i], fontsize=fontsize, weight='bold',
                va='bottom', ha='left', transform=ax.transAxes,
            )
        ax.set_xlim(100, 3500)
        ax.set_ylim(pmax, pmin)
    plt.savefig(figname)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append('free')
    main(sys.argv[1])

