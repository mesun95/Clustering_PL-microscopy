import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#%% Gaussian Mixture Model clustering
def gmm_clustering(
    datacube,
    n_components,
    covariance_type='full',
    random_state=0,
    normalize=True,
    return_probabilities=False,
    BIC=False, **kwargs
):
    """
    Cluster a 3D datacube using Gaussian Mixture Models.

    Parameters
    ----------
    datacube : ndarray, shape (H, W, C)
        Input data cube (e.g. 55×55×3)
    n_components : int
        Number of Gaussian components
    covariance_type : str
        'full', 'tied', 'diag', or 'spherical'
    random_state : int
        Random seed for reproducibility
    normalize : bool
        If True, apply StandardScaler normalization
    return_probabilities : bool
        If True, return posterior probabilities
    BIC : bool
        If True, also return Bayesian Information Criterion

    Returns
    -------
    labels_map : ndarray, shape (H, W)
        Cluster label for each pixel

    probs_map : ndarray, shape (H, W, n_components), optional
        Posterior probabilities

    bic_value : float, optional
        Bayesian Information Criterion
    """

    H, W, C = datacube.shape

    # reshape to (N_pixels, N_features)
    X = datacube.reshape(-1, C)

    # normalize features
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # fit GMM
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        **kwargs
    )
    gmm.fit(X)

    # predict labels
    labels = gmm.predict(X)
    labels_map = labels.reshape(H, W)

    outputs = [labels_map]

    # posterior probabilities
    if return_probabilities:
        probs = gmm.predict_proba(X)
        probs_map = probs.reshape(H, W, n_components)
        outputs.append(probs_map)

    # BIC
    if BIC:
        bic_value = gmm.bic(X)
        outputs.append(bic_value)
        print('BIC value:',bic_value)

    return gmm, tuple(outputs)
#%% show BIC trace
def gmm_BIC(datacube,
    components_max=20,
    covariance_type='full',
    random_state=0,
    normalize=True,
    plot=True):
    """
    Cluster a 3D datacube using Gaussian Mixture Models.

    Parameters
    ----------
    datacube : ndarray, shape (H, W, C)
        Input data cube (e.g. 55×55×3)
    components_max : int
        Maximum number of Gaussian components
    covariance_type : str
        'full', 'tied', 'diag', or 'spherical'
    random_state : int
        Random seed for reproducibility
    normalize : bool
        If True, apply StandardScaler normalization
    plot : bool
        If True, plot the BIC trace

    Returns
    -------
    bic_value : float, optional
        Bayesian Information Criterion (BIC)
    """

    # reshape to (N_pixels, N_features)
    X = datacube.reshape(-1, datacube.shape[-1])

    # normalize features
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # fit GMM
    bic_values = []
    for n_components in range(1, components_max+1):
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state
        )
        gmm.fit(X)
        bic_values.append(gmm.bic(X))
        idx_bic_min = np.argmin(bic_values)+1

    if plot:
        plt.plot(range(1, components_max+1), bic_values)
        # Mark the BIC min index with a grey vertical line
        plt.axvline(x=idx_bic_min, color='grey', linestyle='--', linewidth=1)
        plt.text(idx_bic_min, np.max(bic_values),
                 str(idx_bic_min), fontsize=14, color='grey')
        plt.xlabel('Number of components')
        plt.ylabel('BIC')
        plt.tick_params(which='both', direction='in',right=True, top=True)
        plt.tight_layout()
        plt.show()

    return idx_bic_min, bic_values
#%% Plot the spectra of each cluster
def plot_cluster_average_spectra(
    datacube,
    labels_map,
    x_axis,
    title='Average Spectrum per Cluster',
    x_label='Wavelength / nm',
    y_label='Intensity / a.u.',
    normalize=False,
    figsize=(7, 5),
        fontsize=12,
    cmap_name='tab20',
    savefig=False,
    figname=None,
    save_path=None
):
    """
    Plot the average spectrum of each cluster using consistent colors.

    Parameters
    ----------
    datacube : ndarray, shape (H, W, C)
        Original data cube
    labels_map : ndarray, shape (H, W)
        Cluster labels
    x_axis : ndarray, shape (C,)
        X-axis values (e.g. wavelength, Raman shift)
    title : str
        Plot title
    normalize : bool
        If True, normalize each spectrum to max=1
    figsize : tuple
        Figure size
    cmap_name : str
        Matplotlib colormap name (default: 'tab20')
    """

    H, W, C = datacube.shape
    labels_map = np.asarray(labels_map)

    if labels_map.shape != (H, W):
        raise ValueError("labels_map shape must match spatial dimensions of datacube")

    unique_labels = np.unique(labels_map)
    n_labels = unique_labels.max() + 1

    # use same discrete colormap as label map
    cmap = plt.cm.get_cmap(cmap_name)
    colors = ListedColormap(cmap.colors[:n_labels])


    plt.figure(figsize=figsize)

    for label in unique_labels:  # loop over clusters
        mask = labels_map == label
        spectra = datacube[mask]  # (N_pixels, C)
        mean_spectrum = spectra.mean(axis=0)

        if normalize:
            max_val = np.max(mean_spectrum)
            if max_val > 0:
                mean_spectrum /= max_val
        plt.plot(
            x_axis if x_axis is not None else np.arange(C),
            mean_spectrum,
            color=colors(label),
            label=f'Cluster {label}',
        )

    if title:
        plt.title(title, fontsize=fontsize)

    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.tick_params(which='both', direction='in', right=True, top=True)
    #plt.legend()
    plt.tight_layout()

    if savefig:
        if figname is None:
            print('Warning: figname not specified, saving as cluster_average_spectra.png')
            figname = 'cluster_average_spectra.png'
        plt.savefig(save_path + figname+'.png', dpi=300, transparent=True)
    plt.show()
#%% Get the inter-cluster distance
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
def get_distance(gmm):
    # Calculate Euclidean distance matrix between component means
    means = gmm.means_  # Shape: (n_components, n_features)

    # Compute pairwise Euclidean distances
    n_comp = means.shape[0]
    dist_matrix = np.zeros((n_comp, n_comp))

    for i in range(n_comp):
        for j in range(i + 1, n_comp):
            # Euclidean distance between means
            dist = np.linalg.norm(means[i] - means[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix

# Plot dendrogram based on the distrance matrix
def plot_dendrogram(dist_matrix, linkage_method='ward'):
    n_comp = dist_matrix.shape[0]
    # Convert to condensed form for hierarchical clustering
    # (upper triangle as a 1D array)
    condensed_dist = dist_matrix[np.triu_indices(n_comp, k=1)]

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method=linkage_method)

    # Plot dendrogram if requested
    if plot_dendrogram:
        plt.figure(figsize=(10, 5))
        dendrogram(linkage_matrix,
                   labels=[f'C{i}' for i in range(n_comp)],
                   leaf_rotation=90,
                   leaf_font_size=10)
        plt.title(f'Cluster Merging Hierarchy (Euclidean Distance)\n'
                  f'Distance between GMM component means')
        plt.xlabel('GMM Component')
        plt.ylabel('Euclidean Distance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def plot_gmm_component_dendrogram(
    gmm,
    linkage_method='ward',
    metric='euclidean',
    distance_threshold=None,
    figsize=(10, 5)
):
    """
    Plot dendrogram of GMM components based on mean vectors.

    Parameters
    ----------
    gmm : sklearn.mixture.GaussianMixture
        Fitted GMM model
    linkage_method : str
        'ward', 'complete', 'average', 'single'
    distance_threshold : float, optional
        Draw a horizontal line at this distance
    figsize : tuple
        Figure size
    """

    means = gmm.means_

    # hierarchical clustering
    Z = linkage(means, method=linkage_method, metric=metric)

    plt.figure(figsize=figsize)
    dendrogram(
        Z,
        labels=[f'C{i}' for i in range(means.shape[0])],
        leaf_rotation=90
    )

    if distance_threshold is not None:
        plt.axhline(
            y=distance_threshold,
            color='r',
            linestyle='--',
            label=f'Threshold = {distance_threshold:.2f}'
        )
        plt.legend()

    plt.title('Hierarchical Merging of GMM Components')
    plt.xlabel('GMM Component')
    plt.ylabel('Distance Between Component Means')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
#%% Refine the cluster by agglomerate
from sklearn.cluster import AgglomerativeClustering

def refine_gmm_clusters_by_distance(
    gmm,
    labels_map,
    distance_threshold,
    metric='euclidean',
    linkage='ward'
):
    """
    Refine GMM clusters using hierarchical clustering with a distance threshold.

    Parameters
    ----------
    gmm : sklearn.mixture.GaussianMixture
        Fitted GMM model
    labels_map : ndarray, shape (H, W)
        Original pixel-wise GMM labels
    distance_threshold : float
        Maximum distance for merging clusters
    linkage : str
        'ward', 'complete', 'average', or 'single'

    Returns
    -------
    refined_labels_map : ndarray, shape (H, W)
        Refined cluster labels
    component_to_cluster : ndarray, shape (n_components,)
        Mapping from GMM component → refined cluster
    """

    means = gmm.means_

    agg = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage=linkage,
        metric=metric,
    )

    component_labels = agg.fit_predict(means)

    refined_labels_map = component_labels[labels_map]

    return refined_labels_map, component_labels

#%% Visualize label map
def plot_label_map(
    labels_map,
    title='GMM Clustering Result',
    show_colorbar=True,
    interpolation='nearest',  # set to None or e.g. 'bilinear'
    figsize=(6, 6),
    cmap_name='tab20',
        fontsize=12,
    savefig=False,
    figname=None,
    save_path=None
):
    """
    Visualize a 2D cluster label map.

    Parameters
    ----------
    labels_map : ndarray, shape (H, W)
        Cluster labels
    title : str
        Plot title
    show_colorbar : bool
        Whether to show colorbar
    interpolation : str or None
        Interpolation method ('nearest', 'bilinear', 'bicubic', None)
    figsize : tuple
        Figure size
    """

    labels_map = np.asarray(labels_map)
    n_labels = int(labels_map.max()) + 1

    # use same discrete colormap as label map
    cmap = plt.cm.get_cmap(cmap_name)
    colors = ListedColormap(cmap.colors[:n_labels])

    plt.figure(figsize=figsize)
    im = plt.imshow(
        labels_map,
        cmap=colors,
        interpolation=interpolation
    )
    plt.title(title, fontsize=fontsize)
    plt.axis('off')

    if show_colorbar:
        cbar = plt.colorbar(im, ticks=np.arange(n_labels))
        cbar.set_label('Cluster Label', fontsize=fontsize)

    plt.tight_layout()
    if savefig:
        if figname is None:
            print("Warning: No figure name provided, using default 'label_map.png'.")
            figname = 'label_map.png'
        plt.savefig(save_path + figname + '.png', transparent=True, dpi=300)
    plt.show()

#%% Visualize probability maps
def plot_probability_maps(
    probs_map,
    component_idx,
    title=False,
    cmap='viridis',
    vmin=0.0,
    vmax=1.0,
    interpolation='nearest',
    figsize=(12, 4),
    fontsize=12,
    show_colorbar=True,
    savefig=False,
    figname=None,
    save_path=None
):
    """
    Visualize posterior probability maps from GMM clustering.

    Parameters
    ----------
    probs_map : ndarray, shape (H, W, K)
        Posterior probability maps returned by GMM
    component_indices : list or None
        Indices of components to visualize (default: all)
    titles : list or None
        Custom titles for each subplot
    cmap : str
        Colormap for probability visualization
    vmin, vmax : float
        Color scale limits (recommended [0, 1])
    interpolation : str
        Interpolation method for imshow
    figsize : tuple
        Figure size
    fontsize : int
        Font size for titles
    show_colorbar : bool
        Whether to display colorbars
    savefig : bool
        Whether to save the figure
    figname : str
        Output filename (without extension)
    save_path : str or None
        Directory to save figure
    """

    probs_map = np.asarray(probs_map)
    H, W, K = probs_map.shape


    fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = ax.imshow(probs_map[:, :, component_idx],cmap=cmap,vmin=vmin,vmax=vmax,interpolation=interpolation)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=fontsize)
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Posterior Probability', fontsize=fontsize)

    plt.tight_layout()

    if savefig:
        if figname is None:
            print("Warning: No figure name provided, using default 'probability_maps.png'.")
            figname = 'probability_maps'
        if save_path is None:
            raise ValueError("save_path must be provided when savefig=True")
        plt.savefig(
            save_path + figname + '.png',
            dpi=300,
            transparent=True
        )

    plt.show()

#%% Internal validation metric: variance of spectra in each cluster
def compute_cluster_spectral_variance(
    datacube,
    labels_map,
    normalize_by_bands=True,
):
    """
    Compute spectral intra-cluster variance using the same averaging
    method as plot_cluster_average_spectra.

    Parameters
    ----------
    datacube : ndarray, shape (H, W, C)
        Original hyperspectral datacube
    labels_map : ndarray, shape (H, W)
        Cluster labels
    normalize_by_bands : bool
        If True, divide variance by number of spectral bands

    Returns
    -------
    variances : dict
        {label: spectral intra-cluster variance}
    cluster_sizes : dict
        {label: number of pixels in cluster}
    """
    H, W, C = datacube.shape
    labels_map = np.asarray(labels_map)

    if labels_map.shape != (H, W):
        raise ValueError("labels_map shape must match spatial dimensions of datacube")

    variances = {}
    cluster_sizes = {}

    unique_labels = np.unique(labels_map)

    for label in unique_labels:
        mask = labels_map == label
        spectra = datacube[mask]  # (N_pixels, C)
        cluster_sizes[label] = spectra.shape[0]

        if spectra.shape[0] == 0:
            variances[label] = np.nan
            continue

        # same mean spectrum as your plotting function
        mean_spectrum = spectra.mean(axis=0)

        diff = spectra - mean_spectrum
        var = np.mean(np.sum(diff**2, axis=1))

        if normalize_by_bands:
            var /= C

        variances[label] = var

    return variances, cluster_sizes

def plot_cluster_spectral_variance(
    variances,
    cluster_sizes=None,
    figsize=(7, 5),
    fontsize=12,
    title=False,yscale='log',
    savefig=False,
    figname=None,
    save_path=None
):
    labels = np.array(list(variances.keys()))
    values = np.array(list(variances.values()))

    plt.figure(figsize=figsize)

    if cluster_sizes is not None:
        sizes = np.array([cluster_sizes[l] for l in labels])
        plt.scatter(labels, values, s=sizes, alpha=0.7)
    else:
        plt.scatter(labels, values, alpha=0.7)

    plt.xlabel("Cluster label", fontsize=fontsize)
    plt.ylabel("Spectral intra-cluster variance", fontsize=fontsize)
    if title:
        plt.title("Cluster compactness in spectral domain", fontsize=fontsize)
    plt.grid(True)
    plt.yscale(yscale)
    plt.tight_layout()
    if savefig:
        if figname is None:
            print("Warning: No figure name provided, using default 'cluster_spectral_variance.png'.")
            figname = 'cluster_spectral_variance'
        if save_path is None:
            raise ValueError("save_path must be provided when savefig=True")
        plt.savefig(
            save_path + figname + '.png',
            dpi=300,
            transparent=True
        )
    plt.show()

#%% Internal validation metric: SAM spectral angle mapper
def compute_cluster_spectral_sam(datacube, labels_map, deg=True):
    """
    Compute average SAM of each cluster using original datacube.

    Parameters
    ----------
    datacube : ndarray, shape (H, W, C)
        Hyperspectral datacube
    labels_map : ndarray, shape (H, W)
        Cluster labels
    deg : bool
        If True, returns angle in degrees (default). If False, radians.

    Returns
    -------
    sam_values : dict
        {label: mean SAM of pixels in cluster vs cluster mean spectrum}
    cluster_sizes : dict
        {label: number of pixels in cluster}
    """
    H, W, C = datacube.shape
    labels_map = np.asarray(labels_map)

    if labels_map.shape != (H, W):
        raise ValueError("labels_map shape must match spatial dimensions of datacube")

    sam_values = {}
    cluster_sizes = {}

    unique_labels = np.unique(labels_map)

    for label in unique_labels:
        mask = labels_map == label
        spectra = datacube[mask]  # (N_pixels, C)
        cluster_sizes[label] = spectra.shape[0]

        if spectra.shape[0] == 0:
            sam_values[label] = np.nan
            continue

        mean_spectrum = spectra.mean(axis=0)
        mean_norm = np.linalg.norm(mean_spectrum)
        if mean_norm == 0:
            sam_values[label] = np.nan
            continue

        # dot product and norms
        dots = spectra @ mean_spectrum  # shape (N_pixels,)
        norms = np.linalg.norm(spectra, axis=1) * mean_norm

        # safeguard division by zero
        cos_theta = np.clip(dots / norms, -1.0, 1.0)
        angles = np.arccos(cos_theta)  # radians

        if deg:
            angles = np.degrees(angles)

        sam_values[label] = np.mean(angles)

    return sam_values, cluster_sizes

def plot_cluster_sam(sam_values, cluster_sizes=None, figsize=(7,5), fontsize=12,
                     title=False,yscale='log',
                     savefig=False, figname=None, save_path=None):

    labels = np.array(list(sam_values.keys()))
    values = np.array(list(sam_values.values()))

    plt.figure(figsize=figsize)

    if cluster_sizes is not None:
        sizes = np.array([cluster_sizes[l] for l in labels])
        plt.scatter(labels, values, s=sizes, alpha=0.7)
    else:
        plt.scatter(labels, values, alpha=0.7)

    plt.xlabel("Cluster label", fontsize=fontsize)
    plt.ylabel("Mean SAM (°)", fontsize=fontsize)
    plt.yscale(yscale)
    if title:
        plt.title("Cluster spectral angle similarity", fontsize=fontsize)
    plt.grid(True)
    plt.tight_layout()
    if savefig:
        if figname is None:
            print("Warning: No figure name provided, using default 'cluster_sam.png'.")
            figname = 'cluster_sam'
        if save_path is None:
            raise ValueError("save_path must be provided when savefig=True")
        plt.savefig(
            save_path + figname + '.png',
            dpi=300,
            transparent=True
        )
    plt.show()
