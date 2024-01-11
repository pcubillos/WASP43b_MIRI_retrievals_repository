import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams


def main():
    """
    This Python script reproduces Extended-data Figure 6.

    Requirements
    ------------
    For better results use these versions (older version might still work):
        Python >=3.6
        numpy >= 1.25
        matplotlib >= 3.7.2

    Usage
    -----
    Run from the command line from the folder containing this file:
    $ python fig_WASP43b_retrieval_contributions.py
    """
    models = [
        'hydra',
        'pyratbay',
        'nemesis',
        'platon',
        'scarlet',
    ]
    nmodels = len(models)

    contributions = []
    pressures = []
    for i,model in enumerate(models):
        data = np.load(f'data/contribution_functions_{model}.npz')
        pressures.append(data['pressure'])
        contributions.append(data['contribution_function'])

    wavelength = data['wavelength']
    nbins = len(wavelength)
    phase = data['phase']
    nphase = len(phase)

    # Normalize CFs within [0,height]:
    height = 0.85
    for i,cf in enumerate(contributions):
        for j in range(nphase):
            for k in range(nbins):
                cf[j,k] *= height / np.amax(cf[j,k])
        contributions[i] = cf

    # The plot
    rcParams['font.family'] = ['sans-serif']
    rcParams['font.sans-serif'] = ['Arial']
    rcParams['figure.dpi'] = 150.0
    rcParams['savefig.dpi'] = 300.0

    color_maps = [
        plt.cm.viridis,
        plt.cm.viridis,
        plt.cm.viridis,
        plt.cm.inferno,
        plt.cm.inferno,
    ]

    titles = [
        '$\\bf{{a)}}$    Phase = 0.0',
        '$\\bf{{b)}}$    Phase = 0.25',
        '$\\bf{{c)}}$    Phase = 0.5',
        '$\\bf{{d)}}$    Phase = 0.75',
    ]

    cmax = 1.0/(nbins-0.4)
    fontsize = 7.0


    fig = plt.figure(0)
    fig.set_size_inches(7.086, 2.65)
    plt.clf()
    plt.subplots_adjust(0.06, 0.07, 0.995, 0.675, wspace=0.22)
    # The contribution functions:
    for j in range(nphase):
        ax = plt.subplot(1, 4, 1+j)
        ax.set_ylim(1e1, 1e-5)
        ax.set_xlim(0.0, 5.0)
        ax.set_yscale('log')
        ax.set_xticks([])
        ax.tick_params(which='both', direction='in', labelsize=fontsize)
        ax.tick_params(which='minor', length=0)
        ax.text(
            0.02, 1.02, titles[j], fontsize=fontsize,
            transform=ax.transAxes, va='bottom', ha='left',
        )
        ax.set_xlabel('Contribution functions', fontsize=fontsize)
        if j == 0:
            ax.set_ylabel('Pressure (bar)', fontsize=fontsize)
        for i in range(nmodels):
            plt.axvline(i, color='0.75', lw=1.0)
            plt.text(0.1+i/nmodels, 0.99, models[i], fontsize=fontsize,
                rotation=90, transform=ax.transAxes, va='top',
            )
            cm = color_maps[i]
            for k in range(nbins):
                col = cm(k*cmax)
                ax.plot(i+contributions[i][j,k], pressures[i], lw=1.5, color=col)

    # The color bars:
    cmap = color_maps[0]
    norm = mpl.colors.BoundaryNorm(wavelength, cmap.N)
    ax = plt.axes([0.06, 0.91, 0.6, 0.04])
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax, orientation='horizontal', ticks=[],
    )
    cmap = color_maps[3]
    norm = mpl.colors.BoundaryNorm(wavelength, cmap.N)
    ax = plt.axes([0.06, 0.87, 0.6, 0.04])
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax, orientation='horizontal', ticks=wavelength,
    )
    ax.set_xlabel('Wavelength (um)', fontsize=fontsize)
    ax.tick_params(axis='both', direction='in', labelsize=fontsize-0.5)

    ax.text(
        0.665, 0.92, "Free-chemistry retrievals",
        transform=plt.gcf().transFigure, ha='left', fontsize=fontsize,
    )
    ax.text(
        0.665, 0.875, "Thermochemical-equilibrium retrievals",
        transform=plt.gcf().transFigure, ha='left', fontsize=fontsize,
    )
    plt.savefig('plots/MIRI_WASP43b_retrieval_contribution_functions.pdf')


if __name__ == "__main__":
    main()

