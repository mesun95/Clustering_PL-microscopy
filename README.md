# Clustering_PL-microscopy
Machine learning research project: Clustering-based Analysis for Photoluminescence Microscopy Data of Thin-Film Semiconductors

## Description of codes
### Code for k-means and SLIC on widefield PL data
“k-means_SLIC_Khaled.py”
The python code is divided into 4 main parts:
  1. Part (1/4) is for ROI selection for PL and Reflection TIFF images
  2. Part (2/4) is for Basic K-Means Clustering 
  3. Part (3/4) is for definition of superpixels for SLIC
  4. Part (4/4) is for showing clustering with Hybrid K-Means & SLIC

### Codes for GMMs clustering on hyperspectral PL data
Main code for clustering and plotting:  Clustering_HyperspectralPL.py
Helper functions are under the folder “ML_ResearchProject”.
  1. Clustering.py is used for clustering
  2. HyperSpectralData.py is used for hyperspectral data processing
  3. PCA.py is used for dimensionality reduction 
