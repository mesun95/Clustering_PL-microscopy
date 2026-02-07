import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
#%% New: PCA from sklearn for hypserpy data
# convert hyperspectral data to a 2D array where each row is a spectrum
def stack_spectra_columnwise(cube):
    """
    Stack all spectra vertically into a 2D array.
    Input: cube of shape (ny, nx, nspec)
    Output: array of shape (nspec, ny*nx), column-by-column stacking
    """
    ny, nx, nspec = cube.shape
    return cube.transpose(1, 0, 2).reshape(nx * ny, nspec)
# convert the stacked spectra back to a 3D cube
def unstack_spectra_columnwise(flat, ny, nx):
    """
    Restore stacked spectra back to a 3D cube (ny, nx, nspec)
    Input: stacked array of shape (nspec, ny*nx)
    """
    nspec = flat.shape[1]
    return flat.reshape(nx, ny, nspec).transpose(1, 0, 2)

#%%
from sklearn.decomposition import PCA

def sklearn_PCA(data,ScreePlot=False,n_PCs=None,
                saveplot=False, figname=None, savepath=None,
                threshold=None,
                *args, **kwargs):
    """
    Perform PCA using sklearn on hyperspectral data.
    :param data (np.ndarray): hyperspectral data
    :param ScreePlot (bool): whether to plot the explained variance ratio
    :param n_PCs (int, optional): number of principal components to consider for the scree plot
    :param saveplot (bool): whether to save the scree plot
    :param figname (str): name of the figure to save
    :param savepath (str): path to save the figure
    :param args : additional arguments for sklearn PCA
    :param kwargs: additional arguments for sklearn PCA
    :return:
    pca : trained sklearn PCA object
    component_spectra : array-like, shape (n_components, n_features)
    """
    # flatten the datacube to 2D (pixels × spectrum)
    flat_data = stack_spectra_columnwise(data)
    # apply PCA
    pca = PCA(*args, **kwargs)
    pca.fit(flat_data)
    # save all components in an array
    component_spectra = pca.components_

    if ScreePlot:
        EVR = pca.explained_variance_ratio_
        Acc_EVR = np.cumsum(EVR)

        # component indices (1-based)
        N = np.arange(1, len(EVR) + 1)

        if n_PCs is not None:
            EVR = EVR[:n_PCs]
            Acc_EVR = Acc_EVR[:n_PCs]
            N = N[:n_PCs]

        # --- plotting ---
        fig, ax1 = plt.subplots(figsize=(10, 8))

        # left y-axis: explained variance ratio
        ax1.plot(N, EVR, 'ro', markersize=5, label='Explained variance ratio')
        ax1.set_xlabel('Principal component index')
        ax1.set_ylabel('Explained variance ratio', color='r')
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor='r')

        ax1.xaxis.set_major_locator(MultipleLocator(2))
        ax1.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax1.tick_params(which='both', direction='in', right=True, top=True)

        # right y-axis: accumulated variance ratio
        ax2 = ax1.twinx()
        ax2.plot(N, Acc_EVR, 'bo-', markersize=4, label='Accumulated variance ratio')
        ax2.set_ylabel('Accumulated explained variance ratio', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        #ax2.set_ylim(0, 1.05)

        # threshold line
        if threshold is not None:
            ax2.axhline(threshold, color='gray', linestyle='--', linewidth=1)

        # title & layout
        #fig.suptitle('PCA explained and accumulated variance')
        fig.tight_layout()

        # save
        if saveplot:
            if figname is None:
                savename = 'PCA_scree_plot'
                print("Warning: No figure name provided, using default 'PCA_scree_plot'.")
            else:
                savename = figname

            if savepath is not None:
                plt.savefig(savepath + savename + '.png',
                            transparent=True, dpi=300)
            else:
                plt.savefig(savename + '.png',
                            transparent=True, dpi=300)
                print("Warning: No save path provided, saving in the current directory.")

        plt.show()

    return pca, component_spectra
#%% plot the PCA component spectra
def plot_PCs(component_spectra, x_axis, component_idx,
             x_label='Raman shift / cm$^{-1}$', y_label='Intensity / a.u.',
             fontsize=12,labelpad=10, labelsize=12,
             savefig=False, figname=None, savepath=None):
    """
    Plot the PCA component spectra.

    Parameters:
        component_spectra : array-like, shape (n_components, n_features)
            The PCA component spectra.
        x_axis : array-like, shape (n_features,)
            The x-axis values.
        component_idx : int or list of int, optional
            If int, the first n components will be plotted.
            If list, specific components will be plotted, counting from 1.
        x_label : str, optional
            The label of the x-axis.
        y_label : str, optional
            The label of the y-axis.
        fontsize : int, optional
            The font size of the labels.
        labelpad : int, optional
            The labelpad of the labels.
        labelsize : int, optional
            The label size of the ticks.
        savefig : bool, optional
            Whether to save the figure.
        figname : str, optional
            The name of the figure to save.
        savepath : str, optional
            The path to save the figure.
    Returns:
        None
    """
    if isinstance(component_idx, int):
        # Plot the first n components
        indices = list(range(component_idx))
    elif isinstance(component_idx, list):
        # Plot specific components
        indices = [idx - 1 for idx in component_idx]  # Convert to zero-based index
    else:
        raise ValueError("component_idx must be an int or a list of ints.")
    for idx in indices:
        fig, ax = plt.subplots()
        ax.plot(x_axis,component_spectra[idx], label=f'PC {idx+1}')
        ax.set_xlabel(x_label, fontsize=fontsize, labelpad=labelpad)
        ax.set_ylabel(y_label, fontsize=fontsize, labelpad=labelpad)
        # set the tick fontsize
        plt.tick_params(which='both', direction='in', right=True, top=True, labelsize=labelsize)
        ax.legend()
        plt.tight_layout()
        if savefig:
            if figname is None:
                print(f"Warning: No figure name provided, using default 'PC{idx}'.")
                savename = f'PC{idx+1}'
            else:
                savename = figname+f'_PC{idx+1}'
            if savepath is not None:
                plt.savefig(savepath+savename+'.png', transparent=True, dpi=300)
            else:
                plt.savefig(savename+'.png', transparent=True, dpi=300)
                print("Warning: No save path provided, saving in the current directory.")

        plt.show()
#%%
import math

def plot_PCs_combined(component_spectra, x_axis, component_idx,
                      x_label='Raman shift / cm$^{-1}$',
                      y_label='Intensity / a.u.',
                      fontsize=12, labelsize=12,
                      savefig=False, figname=None, savepath=None):
    """
    Plot several PCA component spectra in one figure with shared axes.

    Parameters:
        component_spectra : array-like, shape (n_components, n_features)
        x_axis : array-like, shape (n_features,)
        component_idx : int or list of int
            If int -> first n PCs (1..n)
            If list -> specific PCs (1-based indices)
        x_label, y_label : str
            Common axis labels
        fontsize, labelpad, labelsize : int
        savefig : bool
        figname : str
        savepath : str or None

    Returns:
        None
    """

    # --- handle indices ---
    if isinstance(component_idx, int):
        indices = list(range(component_idx))
    elif isinstance(component_idx, list):
        indices = [idx - 1 for idx in component_idx]
    else:
        raise ValueError("component_idx must be an int or a list of ints.")

    n = len(indices)

    # --- automatic grid layout (≈ square) ---
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, sharex=False, sharey=False,
                             figsize=(4*ncols, 3*nrows))

    # Flatten axes for easier iteration
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, indices):
        ax.plot(x_axis, component_spectra[idx], label=f'PC {idx+1}')
        ax.legend(fontsize=labelsize-1)
        ax.tick_params(which='both', direction='in',
                       right=True, top=True, labelsize=labelsize)

        # Remove individual axes labels
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Turn off any unused axes (if grid not perfectly filled)
    for ax in axes[n:]:
        ax.axis('off')

    # Shared labels
    fig.supxlabel(x_label, fontsize=fontsize, y=0.01)
    fig.supylabel(y_label, fontsize=fontsize, x=0.01)

    plt.tight_layout()

    # --- saving ---
    if savefig:
        savename = figname if figname is not None else "PCs_combined"
        if savepath is not None:
            plt.savefig(savepath + savename + '.png',
                        transparent=True, dpi=300)
        else:
            plt.savefig(savename + '.png', transparent=True, dpi=300)
            print("Warning: No save path provided, saving in the current directory.")

    plt.show()

#%% get the data with reduced dimensionality
def data_dim_reduced(data, pca, component_idx):
    """
    Project hyperspectral data onto the first n PCA components.

    Parameters
    ----------
    data : ndarray, shape (H, W, B)
        Original hyperspectral data cube.
    pca : sklearn.decomposition.PCA
        Fitted PCA object (trained on data reshaped to (-1, B)).
    component_idx : int or list
        Principal components to keep. If int, the first n components will be kept.
        If list, specific components will be kept, counting from 1.

    Returns
    -------
    projected_data : ndarray, shape (H, W, n_components)
        PCA-reduced hyperspectral cube.
    """
    if isinstance(component_idx, int):
        indices = np.arange(component_idx)
    elif isinstance(component_idx, list):
        indices = np.array([idx - 1 for idx in component_idx])
    else:
        raise ValueError("component_idx must be an int or a list of ints.")

    H, W, B = data.shape

    # Reshape to (n_pixels, n_bands)
    data_2d = stack_spectra_columnwise(data)

    # Project onto PCA space
    data_pca = pca.transform(data_2d)

    # Keep only first n components
    data_pca = data_pca[:,indices]

    # calculated total explained variance ratio
    EVR = pca.explained_variance_ratio_
    EVR_tot = np.sum(EVR[indices])
    print(f"Total explained variance: {EVR_tot}")

    # Reshape back to image cube
    projected_data = unstack_spectra_columnwise(data_pca, H, W)

    return projected_data
