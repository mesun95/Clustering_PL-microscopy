from ML_ResearchProject import HyperSpectralData as HSD
from ML_ResearchProject import PCA
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
#%%
plt.rc('font', size=20) #controls default text size
plt.rc('axes', titlesize=20) #fontsize of the title
plt.rc('axes', labelsize=20) #fontsize of the x and y labels
plt.rc('xtick', labelsize=20) #fontsize of the x tick labels
plt.rc('ytick', labelsize=20) #fontsize of the y tick labels
plt.rc('legend', fontsize=20) #fontsize of the legend
#%% load data (small size)
folder = r'C:\HyperspectralPL_data'
save = r'Y:\a. Personal folders\Mengru\Data\MS017_PEAMAPbI-MACl\MS017_250218\ML'
data_paths = HSD.h5_paths(folder,endswith='.h5')
data_g7, wl_g7, px_g7 = HSD.data_extract(data_paths[1], data_loc='/Datas/Data1', metadata_loc='/Datas/Data1',
    wl_attr='Axis1', x_axis_attr='Axis2', y_axis_attr='Axis3')
#%% reduce dimensionality
pca_g7, pcs_g7 = PCA.sklearn_PCA(data_g7,ScreePlot=True,n_PCs=20)
PCA.plot_PCs_combined(pcs_g7,wl_g7['Wavelength'],component_idx=3)
data_reduced_g7 = PCA.data_dim_reduced(data_g7,pca_g7,component_idx=3)
intint_g7 = HSD.intint_map(data_reduced_g7,wl_g7['Wavelength'],'PL',px_g7['Pixel size'],
                           fontsize=20,
                           #savefig=True, figname='MS017_250218_intint_G7_Map002',
                           savepath=save+'/')
#%% plot label map
from ML_ResearchProject import Clustering as CL

idx_min_g7, bic_g7 = CL.gmm_BIC(data_reduced_g7,components_max=20)
gmm_g7,labels_g7=CL.gmm_clustering(data_reduced_g7,n_components=8,return_probabilities=True,BIC=True)
CL.plot_label_map(labels_g7[0],cmap_name='tab20',figsize=(8,6),fontsize=20,title=None,
                  #savefig=True,figname='MS017_250218_G7_label_map',
                  save_path=save+'/')
CL.plot_cluster_average_spectra(data_g7,labels_map=labels_g7[0],x_axis=wl_g7['Wavelength'],fontsize=20,title=None,
                                #savefig=True,figname='MS017_250218_G7_Map002_average_spectra',
                                save_path=save+'/')
#%%
ref_folder = r'C:\HyperspectralPL_data'
ref_img_paths = HSD.h5_paths(ref_folder,endswith='.jpg',keywords=['G7','cropped'])
ref_g7_norm = np.fromfile(ref_img_paths[0], dtype=np.uint8)

import cv2
ref_g7 = cv2.imdecode(ref_g7_norm, cv2.IMREAD_COLOR)
ref_g7_rgb = cv2.cvtColor(ref_g7, cv2.COLOR_BGR2RGB)
ref_g7_resized = cv2.resize(ref_g7_rgb,(232, 232),interpolation=cv2.INTER_LINEAR)

labels_g7_resized = cv2.resize(labels_g7[0],(232, 232),interpolation=cv2.INTER_NEAREST)
#%%
from matplotlib.colors import ListedColormap
cmap = plt.cm.get_cmap('tab20')
colors = ListedColormap(cmap.colors[:8])
plt.imshow(ref_g7_resized)
#plt.imshow(labels_g7_resized,cmap=colors,alpha=0.4)
plt.show()

#%%
var_g7,cluster_sizes_g7 = CL.compute_cluster_spectral_variance(data_g7,labels_g7[0])
CL.plot_cluster_spectral_variance(var_g7,cluster_sizes_g7,figsize=(7,5),fontsize=20,title=None,
                                  #savefig=True,figname='MS017_250218_G7_spectral_variance',
                                  save_path=save+'/')

sam_g7,cluster_sizes_g7 = CL.compute_cluster_spectral_sam(data_g7,labels_g7[0])
CL.plot_cluster_sam(sam_g7,cluster_sizes_g7,figsize=(7,5),fontsize=20,title=None,
                                  #savefig=True,figname='MS017_250218_G7_spectral_sam',
                                  save_path=save+'/')

#%% Plot SAM and variance vs cluster label
def plot_sam_vs_variance(sam_values, variances, figsize=(7, 5), title=None,
                         savefig=False, figname=None, save_path=None):

    # --- Prepare data ---
    labels = np.array(list(variances.keys()))
    var_values = np.array([variances[l] for l in labels])
    sam_values_arr = np.array([sam_values[l] for l in labels])

    # --- Create figure and axes ---
    fig, ax1 = plt.subplots(figsize=figsize)

    color_sam = 'tab:blue'
    color_var = 'tab:red'

    # Left y-axis: SAM
    ax1.set_xlabel('Cluster label')
    ax1.set_ylabel('Mean SAM (°)', color=color_sam)
    ax1.set_yscale('log')
    ax1.scatter(labels, sam_values_arr, color=color_sam, label='Mean SAM $^{\circ}$', alpha=0.7)
    ax1.plot(labels, sam_values_arr, color=color_sam, linestyle='--', alpha=0.5)
    ax1.tick_params(axis='y', labelcolor=color_sam)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Right y-axis: Variance
    ax2 = ax1.twinx()
    ax2.set_ylabel('Spectral intra-cluster variance', color=color_var)
    ax2.set_yscale('log')
    ax2.scatter(labels, var_values, color=color_var, label='Variance', alpha=0.7)
    ax2.plot(labels, var_values, color=color_var, linestyle='--', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color_var)

    # Optional: title
    if title:
        plt.title('Cluster Compactness (Variance) vs Spectral Similarity (SAM)')

    # Legends
    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')

    plt.tight_layout()
    if savefig:
        if figname is None:
            figname = 'spectral_variance_sam.png'
            print('Warning: figure name is None. Using default name: {}'.format(figname))

        plt.savefig(save_path+figname, dpi=300, transparent=True)
    plt.show()

#%%
plot_sam_vs_variance(sam_g7,var_g7,figsize=(8,5),title=None,savefig=False,
                     #figname='MS017_250218_G7_spectral_variance_sam.png',
                     save_path=save+'/')

#%% Compare with k-means
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data_reduced_g7_2d = data_reduced_g7.reshape(-1,data_reduced_g7.shape[-1])
scaler = StandardScaler()
data_reduced_g7_2d_scaled = scaler.fit_transform(data_reduced_g7_2d)

kmeans_g7 = KMeans(n_clusters=8,random_state=0,n_init=20)
kmeans_labels_g7 = kmeans_g7.fit_predict(data_reduced_g7_2d_scaled)
# reshape back to original shape
kmeans_labels_g7 = kmeans_labels_g7.reshape(data_reduced_g7.shape[0],data_reduced_g7.shape[1])

CL.plot_label_map(kmeans_labels_g7,cmap_name='tab20',figsize=(8,6),fontsize=20,title=None,
                  #savefig=True,figname='MS017_250218_G7_label_map_kmeans',
                  save_path=save+'/')

CL.plot_cluster_average_spectra(data_g7,labels_map=kmeans_labels_g7,x_axis=wl_g7['Wavelength'],fontsize=20,title=None,
                                #savefig=True,figname='MS017_250218_G7_Map002_average_spectra_kmeans',
                                save_path=save+'/')

kmeans_var_g7, kmeans_cluster_sizes_g7 = CL.compute_cluster_spectral_variance(data_g7,labels_map=kmeans_labels_g7)
CL.plot_cluster_spectral_variance(kmeans_var_g7,kmeans_cluster_sizes_g7,figsize=(7,5),fontsize=20,title=None,
                                  #savefig=True,figname='MS017_250218_G7_kmeans_spectral_variance',
                                  save_path=save+'/')
kmeans_sam_g7, kmeans_cluster_sizes_g7 = CL.compute_cluster_spectral_sam(data_g7,labels_map=kmeans_labels_g7)
CL.plot_cluster_sam(kmeans_sam_g7,kmeans_cluster_sizes_g7,figsize=(7,5),fontsize=20,title=None,
                                    #savefig=True,figname='MS017_250218_G7_kmeans_spectral_similarity',
                                    save_path=save+'/')
#%% Plot SAM and variance vs cluster label (k-means)
plot_sam_vs_variance(kmeans_sam_g7,kmeans_var_g7,figsize=(8,5),title=None,savefig=False,
                     #figname='MS017_250218_G7_kmeans_spectral_variance_sam.png',
                     save_path=save+'/')

#%% Plot histograms
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_split_cluster_violins(
    data2d,
    labels_1,
    labels_2,
    exclude_label=None,
    figsize=(9, 5),
    inner="quartile",
    density_norm='count',
    ylabel='PL integrated intensity / a.u.',
    title='Cluster-wise Distribution: GMM vs k-means',
    legend_labels=['GMM', 'k-means'],
    legend_title = 'Clustering method',
    save_path=None
):
    """
    Plot split violin plots comparing GMM and k-means cluster distributions.

    Parameters
    ----------
    data2d : ndarray (H, W)
        2D data array.
    labels_1 : ndarray (H, W)
        Cluster labels 1 (e.g., GMM).
    labels_2 : ndarray (H, W)
        Cluster labels 2 (e.g., k-means).
    exclude_label : int or None
        Label to exclude (e.g., background = -1).
    figsize : tuple
        Figure size.
    density_norm : str
        'count' or 'area', see https://seaborn.pydata.org/generated/seaborn.violinplot.html.
    inner : str
        'quartile' or 'point', see https://seaborn.pydata.org/generated/seaborn.violinplot.html.
    ylabel : str
        Y-axis label.
    title : str
        Plot title.
    legend_labels : list of str
        Legend labels that must correspond to labels_1 and labels_2, e.g., ['GMM', 'k-means'].
    legend_title : str
        Legend title
    """

    if not (data2d.shape == labels_1.shape == labels_2.shape):
        raise ValueError("data2d, labels_1, and labels_2 must have the same shape")

    # Find common cluster labels
    clusters = np.intersect1d(
        np.unique(labels_1),
        np.unique(labels_2)
    )

    if exclude_label is not None:
        clusters = clusters[clusters != exclude_label]

    records = []

    for c in clusters:
        # Label map 1
        v_gmm = data2d[labels_1 == c]
        v_gmm = v_gmm[np.isfinite(v_gmm)]
        records.extend(
            [(c, val, legend_labels[0]) for val in v_gmm]
        )

        # Label map 2
        v_km = data2d[labels_2 == c]
        v_km = v_km[np.isfinite(v_km)]
        records.extend(
            [(c, val, legend_labels[1]) for val in v_km]
        )

    df = pd.DataFrame(
        records,
        columns=["Cluster", "Value", "Method"]
    )

    plt.figure(figsize=figsize)
    sns.violinplot(
        data=df,
        x="Cluster",
        y="Value",
        hue="Method",
        split=True,
        inner=inner,
        density_norm=density_norm,
        cut=0
    )

    plt.xlabel("Cluster")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title=legend_title)
    plt.tight_layout()
    #plt.ylim(600,800)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, transparent=True)
    plt.show()

intint_PL_g7 = HSD.intint_map(data_g7,wl_g7['Wavelength'],'PL',px_g7['Pixel size'])
plot_split_cluster_violins(intint_PL_g7,labels_g7[0],kmeans_labels_g7,exclude_label=-1,figsize=(10,8),
                           #save_path=save+'/MS017_250218_G7_intint_PL_split_violins_GMM_kmeans.png',
                           density_norm='count')

#%% Comparison with no PCA or different PCs
# no PCA
idx_bic_min_g7_noPCA, bic_g7_noPCA = CL.gmm_BIC(data_g7,components_max=20)
gmm_g7_noPCA,labels_g7_noPCA=CL.gmm_clustering(data_g7,n_components=idx_bic_min_g7_noPCA,return_probabilities=True,BIC=True)
CL.plot_label_map(labels_g7_noPCA[0],cmap_name='tab20',figsize=(8,6),fontsize=20,title=None,
                  #savefig=True,figname='MS017_250218_G7_label_map_noPCA',
                  save_path=save+'/')
CL.plot_cluster_average_spectra(data_g7,labels_map=labels_g7_noPCA[0],x_axis=wl_g7['Wavelength'],fontsize=20,title=None,
                                #savefig=True,figname='MS017_250218_G7_average_spectra_noPCA',
                                save_path=save+'/')
sam_g7_noPCA,cluster_sizes_g7_noPCA = CL.compute_cluster_spectral_sam(data_g7,labels_g7_noPCA[0])
var_g7_noPCA,cluster_sizes_g7_noPCA = CL.compute_cluster_spectral_variance(data_g7,labels_g7_noPCA[0])
plot_sam_vs_variance(sam_g7_noPCA,var_g7_noPCA,figsize=(8,5),title=None,savefig=False,
                     #figname='MS017_250218_G7_noPCA_spectral_variance_sam.png',
                     save_path=save+'/')

# 1 PC
data_reduced_g7_1PC = PCA.data_dim_reduced(data_g7,pca_g7,component_idx=1)
idx_bic_min_g7_1PC, bic_g7_1PC = CL.gmm_BIC(data_reduced_g7_1PC,components_max=20)
gmm_g7_1PC,labels_g7_1PC=CL.gmm_clustering(data_reduced_g7_1PC,n_components=idx_bic_min_g7_1PC,return_probabilities=True,BIC=True)
CL.plot_label_map(labels_g7_1PC[0],cmap_name='tab20',figsize=(8,6),fontsize=20,title=None,
                  #savefig=True,figname='MS017_250218_G7_label_map_1PC',
                  save_path=save+'/')
CL.plot_cluster_average_spectra(data_g7,labels_map=labels_g7_1PC[0],x_axis=wl_g7['Wavelength'],fontsize=20,title=None,
                                #savefig=True,figname='MS017_250218_G7_average_spectra_1PC',
                                save_path=save+'/')
sam_g7_1PC,cluster_sizes_g7_1PC = CL.compute_cluster_spectral_sam(data_reduced_g7_1PC,labels_g7_1PC[0])
var_g7_1PC,cluster_sizes_g7_1PC = CL.compute_cluster_spectral_variance(data_reduced_g7_1PC,labels_g7_1PC[0])
plot_sam_vs_variance(sam_g7_1PC,var_g7_1PC,figsize=(8,5),title=None,savefig=False,
                     #figname='MS017_250218_G7_1PC_spectral_variance_sam.png',
                     save_path=save+'/')

# 2 PCs
data_reduced_g7_2PC = PCA.data_dim_reduced(data_g7,pca_g7,component_idx=[1,2])
idx_bic_min_g7_2PC, bic_g7_2PC = CL.gmm_BIC(data_reduced_g7_2PC,components_max=20)
gmm_g7_2PC,labels_g7_2PC=CL.gmm_clustering(data_reduced_g7_2PC,n_components=idx_bic_min_g7_2PC,return_probabilities=True,BIC=True)
CL.plot_label_map(labels_g7_2PC[0],cmap_name='tab20',figsize=(8,6),fontsize=20,title=None,
                  #savefig=True,figname='MS017_250218_G7_label_map_2PC',
                  save_path=save+'/')
CL.plot_cluster_average_spectra(data_g7,labels_map=labels_g7_2PC[0],x_axis=wl_g7['Wavelength'],fontsize=20,title=None,
                                #savefig=True,figname='MS017_250218_G7_average_spectra_2PC',
                                save_path=save+'/')
sam_g7_2PC,cluster_sizes_g7_2PC = CL.compute_cluster_spectral_sam(data_reduced_g7_2PC,labels_g7_2PC[0])
var_g7_2PC,cluster_sizes_g7_2PC = CL.compute_cluster_spectral_variance(data_reduced_g7_2PC,labels_g7_2PC[0])
plot_sam_vs_variance(sam_g7_2PC,var_g7_2PC,figsize=(8,5),title=None,savefig=False,
                     #figname='MS017_250218_G7_2PC_spectral_variance_sam.png',
                     save_path=save+'/')

# 4 PCs
data_reduced_g7_4PC = PCA.data_dim_reduced(data_g7,pca_g7,component_idx=[1,2,3,4])
idx_bic_min_g7_4PC, bic_g7_4PC = CL.gmm_BIC(data_reduced_g7_4PC,components_max=20)
gmm_g7_4PC,labels_g7_4PC=CL.gmm_clustering(data_reduced_g7_4PC,n_components=idx_bic_min_g7_4PC,return_probabilities=True,BIC=True)
CL.plot_label_map(labels_g7_4PC[0],cmap_name='tab20',figsize=(8,6),fontsize=20,title=None,
                  #savefig=True,figname='MS017_250218_G7_label_map_4PC',
                  save_path=save+'/')
CL.plot_cluster_average_spectra(data_g7,labels_map=labels_g7_4PC[0],x_axis=wl_g7['Wavelength'],fontsize=20,title=None,
                                #savefig=True,figname='MS017_250218_G7_average_spectra_4PC',
                                save_path=save+'/')
sam_g7_4PC,cluster_sizes_g7_4PC = CL.compute_cluster_spectral_sam(data_reduced_g7_4PC,labels_g7_4PC[0])
var_g7_4PC,cluster_sizes_g7_4PC = CL.compute_cluster_spectral_variance(data_reduced_g7_4PC,labels_g7_4PC[0])
plot_sam_vs_variance(sam_g7_4PC,var_g7_4PC,figsize=(8,5),title=None,savefig=False,
                     #figname='MS017_250218_G7_4PC_spectral_variance_sam.png',
                     save_path=save+'/')

# 5 PCs
data_reduced_g7_5PC = PCA.data_dim_reduced(data_g7,pca_g7,component_idx=[1,2,3,4,5])
idx_bic_min_g7_5PC, bic_g7_5PC = CL.gmm_BIC(data_reduced_g7_5PC,components_max=20)
gmm_g7_5PC,labels_g7_5PC=CL.gmm_clustering(data_reduced_g7_5PC,n_components=idx_bic_min_g7_5PC,return_probabilities=True,BIC=True)
CL.plot_label_map(labels_g7_5PC[0],cmap_name='tab20',figsize=(8,6),fontsize=20,title=None,
                  #savefig=True,figname='MS017_250218_G7_label_map_5PC',
                  save_path=save+'/')
CL.plot_cluster_average_spectra(data_g7,labels_map=labels_g7_5PC[0],x_axis=wl_g7['Wavelength'],fontsize=20,title=None,
                                #savefig=True,figname='MS017_250218_G7_average_spectra_5PC',
                                save_path=save+'/')
sam_g7_5PC,cluster_sizes_g7_5PC = CL.compute_cluster_spectral_sam(data_reduced_g7_5PC,labels_g7_5PC[0])
var_g7_5PC,cluster_sizes_g7_5PC = CL.compute_cluster_spectral_variance(data_reduced_g7_5PC,labels_g7_5PC[0])
plot_sam_vs_variance(sam_g7_5PC,var_g7_5PC,figsize=(8,5),title=None,savefig=False,
                     #figname='MS017_250218_G7_5PC_spectral_variance_sam.png',
                     save_path=save+'/')

#%% Compare results using 3PC and 4PC
plot_split_cluster_violins(intint_PL_g7,labels_g7[0],labels_g7_4PC[0],exclude_label=-1,figsize=(10,8),
                           title = 'Cluster-wise Distribution vs dimensionality reduction',
                           legend_labels=['3 PCs', '4 PCs'],
                           legend_title='Components used',
                           #save_path=save+'/MS017_250218_G7_intint_PL_split_violins_GMM_kmeans_4PC.png',
                           density_norm='count')