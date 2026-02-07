import tables
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
#%% extract file paths from a folder
def h5_paths(folder, endswith='.h5', keywords=None):
    """
    Function to extract file paths from a folder based on file extension and keywords
    :param folder: Path to the folder
    :param endswith: File extension to filter (default: '.h5')
    :param keywords: List of keywords that must be present in the filename (default: None)
    :return: List of file paths that match the criteria
    """
    matched_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            # Check extension
            if not file.endswith(endswith):
                continue
            # If keywords provided, ensure they all appear in the filename
            if keywords:
                if not all(keyword in file for keyword in keywords):
                    continue
            matched_files.append(os.path.join(root, file))
    return matched_files
#%% save the input h5 dataset(s) to an HDF5 files (pytables)
def normalize_path(path: str):
    return os.path.abspath(os.path.normpath(path))
#%% extract data, wavelength and axes info from an H5 file of hyperspectral data
def data_extract(h5_path, data_loc='/Datas/Data1', metadata_loc=None,
                 wl_attr='Axis1',x_axis_attr='Axis2',decode=True, y_scale = False, y_axis_attr='Axis3'):
    """
    Function to extract data, signal axis and pixel size from an HDF5 (h5) file
    :param h5_path: Path to the HDF5 file
    :param data_loc: Location of the data in the file (default: '/Datas/Data1')
    :param metadata_loc: Location of the metadata in the file (default: None)
    :param wl_attr: Name of the wavelength axis attribute (default: 'Axis1')
    :param x_axis_attr: Name of the x axis attribute (default: 'Axis2')
    :param y_scale: Boolean to indicate if the y axis needs to be returned (default: False)
    :param y_axis_attr: Name of the y axis attribute (default: 'Axis3')
    :return: data, wavelength_axis, x_px_size, y_px_size (if y_scale is True)
    """
    import numpy as np
    h5_path = normalize_path(h5_path)
    h5_file = tables.open_file(h5_path, mode='r')
    data_node = h5_file.get_node(data_loc)
    data = data_node.read()
    if metadata_loc:
        metadata_node = h5_file.get_node(metadata_loc)
    else:
        metadata_node = data_node

    wavelength_axis = metadata_node.attrs[wl_attr]
    if decode:
        wavelength_unit = metadata_node.attrs[wl_attr+' Unit'].decode('latin-1')
    else:
        wavelength_unit = metadata_node.attrs[wl_attr+' Unit']
    wavelength_info = {'Wavelength':wavelength_axis, 'Unit':wavelength_unit}
    #(f"Wavelength axis unit: {metadata_node.attrs[wl_attr+' Unit'].decode('latin-1')}")
    x_axis = metadata_node.attrs[x_axis_attr]
    if decode:
        x_axis_unit = metadata_node.attrs[x_axis_attr+' Unit'].decode('latin-1')
    else:
        x_axis_unit = metadata_node.attrs[x_axis_attr+' Unit']
    #print(f"Pixel size in x axis unit: {metadata_node.attrs[x_axis_attr+' Unit'].decode('latin-1')}")

    # check if position axis is uniformly spaced
    def is_equally_spaced_np(arr, tol=0):
        arr = np.asarray(arr)
        diffs = np.diff(arr)
        return np.all(np.abs(diffs - diffs[0]) == tol)
    if is_equally_spaced_np(x_axis):
        x_px_size = x_axis[1] - x_axis[0]
        x_axis_info = {'Pixel size': x_px_size,'Unit': x_axis_unit}
    else:
        print("Warning: The x axis is not equally spaced. The full x axis array is returned.")
        x_axis_info = {'X axis': x_axis, 'Unit': x_axis_unit}
    if y_scale:
        y_axis = metadata_node.attrs[y_axis_attr]
        #print(f"Pixel size in y axis unit: {metadata_node.attrs[y_axis_attr + ' Unit'].decode('latin-1')}")
        if decode:
            y_axis_unit = metadata_node.attrs[y_axis_attr + ' Unit'].decode('latin-1')
        else:
            y_axis_unit = metadata_node.attrs[y_axis_attr + ' Unit']
        if is_equally_spaced_np(y_axis):
            y_px_size = y_axis[1] - y_axis[0]
            y_axis_info = {'Pixel size': y_px_size,'Unit': y_axis_unit}
        else:
            print("Warning: The y axis is not equally spaced. The full y axis array is returned.")
            y_axis_info = {'Y axis': y_axis, 'Unit': y_axis_unit}
        h5_file.close()
        return data, wavelength_info, x_axis_info, y_axis_info
    else:
        h5_file.close()
        return data, wavelength_info, x_axis_info

#%% get integrated intensity of the hyperspectral data
def get_intint(data, xaxis, spectral_range=None):
    """
    Get the integrated intensity of the hyperspectral data over the given wavelength or wavenumber range
    :param data (ndarray): hyperspectral data
    :param xaxis (np.ndarray): the wavelength or wavenumber axis
    :param spectral_range (tuple)(optional): the wavelength or wavenumber range, for example, (500, 700) in nm for PL
    :param processed_data (np.ndarray)(optional): the registered data
    :return: intint (np.ndarray): the integrated intensity map
    """
    if spectral_range is not None:
        index1 = abs(xaxis - spectral_range[0]).argmin()
        index2 = abs(xaxis- spectral_range[1]).argmin()
    else:
        index1 = 0
        index2 = data.shape[2]-1
    intint = data[:,:,index1:index2].sum(axis=2)
    return intint
#%% get the ideal scalebar length
def get_scalebar_length(data, pixel_to_mum, percent=0.133335):
    """
    Calculate the ideal scalebar length based on the data and pixel size. Best length of a scale bar. 13% of the length of the image.
    :param data: an image data
    :param pixel_to_mum: Pixel size in micrometers.
    :param percent: the percentage of the image length to be used for the scalebar, default is 0.133335 (13%).
    :return:
    len_in_pix (float): the length of the scalebar in pixels
    length (float): the length of the scalebar in micrometers
    width (float): the width of the scalebar in pixels
    """
    ideal_length_scale_bar = data.shape[1] *pixel_to_mum * percent  # 13% of the image length in micrometers

    # Work out how many pixels are required for the scalebar. If scale bar length is > 10 round to the nearest 5.
    if ideal_length_scale_bar > 10:
        n = (ideal_length_scale_bar - 10) / 5
        n = round(n)
        length = int(10 + 5 * n)
        len_in_pix = length / pixel_to_mum

    # Round to the nearest integer if between 1 and 10.
    elif (ideal_length_scale_bar <= 10) & (ideal_length_scale_bar >= 1):
        n = int(round(ideal_length_scale_bar))
        length = n
        len_in_pix = length / pixel_to_mum

    # Round to 1 decimal place if < 1.
    elif ideal_length_scale_bar < 1:
        n = round(ideal_length_scale_bar, 1)
        length = n
        len_in_pix = n / pixel_to_mum

    width = 0.06 * len_in_pix

    return len_in_pix, length, width
#%% adjust colorbar ticks based on the histogram of the data
def adjust_colorbar(data, bins=500, percentiles=(5, 95)):
    """
    Adjust the colorbar ticks based on the histogram of the data.
    :param data: 2D array-like data for which to adjust the colorbar
    :param bins: Number of bins for the histogram
    :param percentiles: Percentiles to use for adjusting the colorbar limits
    :return: vmin, vmax - adjusted colorbar limits
    """
    hist, bin_edges = np.histogram(data.flatten(), bins=bins)
    cum_counts = np.cumsum(hist)
    total_counts = np.sum(hist)

    lower_bound = bin_edges[np.where(cum_counts >= total_counts * percentiles[0] / 100)[0][0]]
    upper_bound = bin_edges[np.where(cum_counts <= total_counts * percentiles[1] / 100)[0][-1]]

    return lower_bound, upper_bound
#%% Plot an integrated intensity map
def intint_map(data, xaxis, data_type, px_size,
               spectral_range=None,
               frac_scalebar=0.133335,
               savefig=False, figname=None, savefile=False, filename=None, savepath=None,
               cbar_adj=True,
               fontsize=12,labelpad=10,cmap='viridis',
               **cbar_kwargs):
    """
    Plot an integrated intensity map over the given wavelength or wavenumber range
    :param data (ndarray): hyperspectral data
    :param xaxis (np.ndarray): the wavelength or wavenumber axis
    :param px_size (float): the pixel size in micrometers
    :param spectral_range (tuple)(optional): the wavelength or wavenumber range, for example, (500, 700) in nm for PL
    :param data_type (str): the type of the data, either 'PL' or 'Raman'
    :return: intint (np.ndarray): the integrated intensity map
    """
    intint = get_intint(data, xaxis=xaxis, spectral_range=spectral_range)
    # Get the scalebar length
    len_in_pix, length, width = get_scalebar_length(intint, px_size, percent=frac_scalebar)

    # Plot the map
    fig,ax = plt.subplots()
    if cbar_adj:
        vmin, vmax = adjust_colorbar(intint,**cbar_kwargs)
        cmap = ax.imshow(intint, vmin=vmin, vmax=vmax, cmap=cmap)
    else:
        cmap = ax.imshow(intint, cmap=cmap)
    ax.set_axis_off()
    scalebar = AnchoredSizeBar(ax.transData, len_in_pix, str(length) + ' μm', 4, pad=1,
                               borderpad=0.1, sep=5, frameon=False, size_vertical=width, color='white',
                               fontproperties={'size': 15, 'weight': 'bold'})
    ax.add_artist(scalebar)
    fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbar = fig.colorbar(cmap, ax=ax, format=fmt)
    cbar.set_label('{} integrated intensity / a.u.'.format(data_type), fontsize=fontsize, labelpad=labelpad)
    plt.tight_layout()
    if savefig:
        if figname is None:
            print('Please provide a figure name')
        else:
            plt.savefig(savepath+figname+'.png', transparent=True, dpi=300)
    plt.show()

    if savefile:
        if filename is None:
            print('Please provide a file name')
        else:
            np.savetxt(savepath+filename+'.txt', intint)
    return intint
