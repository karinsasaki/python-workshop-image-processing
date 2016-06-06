# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 16:28:01 2016

@author:    Jonas Hartmann @ Gilmour Group @ EMBL Heidelberg

@descript:  A crude introduction on how to pipeline single-cell segmentation 
            data into downstream analyses such as clustering. 
            
            For people with limited experience in data analysis, this script is
            intended as an inspiration and incentive to think about possible
            advanced analyses downstream of segmentation. Solving the exercises
            without help may be difficult, so it may be a good idea to have a 
            look at the solutions to get some idea of how the problems should
            be approached. However, once the principles are understood, it is
            an important part of the learning experience to build one's own 
            implementation.
            
            More experienced people can use this script as a starting point to 
            exploring the data analysis packages provided for Python. It also 
            illustrates that Python readily allows the construction of complete
            and consistent analysis pipelines, from image preprocessing to  
            feature extraction to clustering (and back). The exercises will be
            doable, and are intended as an incentive to think about the concepts,
            also in regards to ones own data.
            
            There are a number of machine learning, clustering and other data
            analysis packages for Python. As a starting point, I recommend you
            look into the following:
            
            - scikit-learn  (scikit-learn.org/stable/)
            - scipy.cluster (docs.scipy.org/doc/scipy/reference/cluster.html)
            - networkx      (networkx.github.io/)
            
            For people interested in Bayesian methods (not covered here), I
            recommend the PyMC package (pymc-devs.github.io/pymc/).

@WARNING:   This exercise and the associated solutions are a BETA! They have
            been implemented in a limited amount of time and have not been
            tested extensively. Furthermore, the example data used is rather
            uniform in regards to many conventional features and thus is not
            ideal to illustrate clustering. Nevertheless, the principles and
            packages introduced here should serve as a good inspiration or
            starting point for further study.

@requires:  Python 2.7
            NumPy 1.9, scikit-image 0.11.3, matplotlib 1.5.1
            SciPy 0.16.0, scikit-learn 0.15.2, networkx 1.9.1
"""


#------------------------------------------------------------------------------

### PREPARATION

### Module imports
from __future__ import division    # Python 2.7 legacy
import numpy as np                 # Array manipulation package
import matplotlib.pyplot as plt    # Plotting package
import scipy.ndimage as ndi        # Multidimensional image operations


### Importing image and segmentation data from main tutorial

# Note: Loading from .npy is faster, so do the following once:
#filename = 'example_cells_1'
#import skimage.io as IO
#img = IO.imread(filename+'.tif')[0,:,:]
#img = ndi.filters.gaussian_filter(img,3) # Smoothing 
#np.save(filename+'_green',img)

# Loading from npy
filename = 'example_cells_1'
img = np.load(filename+'_green.npy')
seg = np.load(filename+'_segmented.npy')

# Some frequently used variables
labels = np.unique(seg)[1:]  # Labels of cells in segmentation
N = len(labels)              # Number of cells in segmentation


#------------------------------------------------------------------------------

### FEATURE EXTRACTION
# As discussed in the main tutorial, we can measure various quantities for each
# cell once the cells have been segmented. Any such quantity can be used as a
# feature to classify or cluster cells. Besides explicitly measured quantities,
# there are algorithms/packages that measure a whole bunch of features at once.

# All the extracted features together are called the 'feature space'. Each
# sample can be considered a point in a space that has as many dimensions as
# there are features. The feature space should be arranged as an array that
# has the shape (n_samples,n_features).

# EXERCISE 1: 
# Come up with at least 4 different features and measure them for each cell in  
# the segmentation of the main tutorial.

# Hint: For many measures of shape and spatial distribution, it is useful to 
#       first calculate the centroid of the segmented object and then think of
#       features relative to it.

# Hint: It can be advantageous to use measures that are largely independent of 
#       or normalized for cell size, so this factor does not end up dominating 
#       the features. Cells size itself can be a useful feature, though.

# Hint: Don't forget that we detected the membranes of each cell in the main
#       script. Importing this data may be useful for the calculation of
#       various features.

# Hint: Make sure you visualize your data!
#       It can be very useful to have a look at what a feature looks like when 
#       mapped to the actual image. This may already show interesting patterns, 
#       or should at least confirm that the extracted value is consistent with 
#       its rationale. For example, one could show a feature as a color-coded
#       semi-transparent overlay over the actual image.
#       Furthermore, box and scatter plots are great options for investigating 
#       how the values of a feature are distributed and how features relate to
#       each other..


#"""
### IMPORTANT NOTE: The following are just suggestions/ideas! Your solution
#   should be conceptually similar but not at all the same. This also goes for
#   the rest of the pipeline, both because there are multiple ways of achieving
#   the same or a similar goal, and also because your solutions here will 
#   change the outcome of the analyses further below!

### Initialize feature space
fspace = np.zeros((N,4),dtype=np.float)


### Get the centroids
cen = np.zeros((N,2))
for index,label in enumerate(labels):
    cen[index,:] = ndi.measurements.center_of_mass(seg==label)


### CELL SIZE (as approximated by segmentation area)

for index,label in enumerate(labels):
    
    # Count the number of pixels that are part of that object
    fspace[index,0] = np.sum(seg==label)


### STANDARD DEVIATION OF MEMBRANE INTENSITY

for index,label in enumerate(labels):
    
    # Dilation provides yet another way of detecting edge pixels
    edge = ndi.maximum_filter(seg!=label,size=3)
    edge[seg!=label] = 0
    
    # Stdev of intensity at the edge
    fspace[index,1] = np.std(img[edge==1])


### INTENSITY ASYMMETRY as the distance of the intensity-weighted center of
#   mass from the unweighted centroid

for index,label in enumerate(labels):
    
    # Intensity-weighted center of mass
    int_cen = ndi.measurements.center_of_mass(np.ma.array(img,mask=seg!=label))
    
    # Distance between the two centroids
    # Note: Below I use the pairwise distance function pdist from scipy's
    #       spatial package. This function is useful for a lot of things in
    #       data analsis, but to be more explicit one could also calculate the
    #       distance between two centroids using simple Pythagoras:
    #           diff = int_cen - cen[index,:]
    #           dist = np.sqrt(np.square(diff[0]) + np.square(diff[1]))
    from scipy.spatial.distance import pdist
    fspace[index,2] = pdist([int_cen,cen[index,:]])[0]


### ROUNDNESS as ratio of inscribed vs circumscribed circle

for index,label in enumerate(labels):
    
    # Distance transform to get circular distance around centroid
    dtr = np.ones_like(seg)
    dtr[int(cen[index,0]),int(cen[index,1])] = 0
    dtr = ndi.distance_transform_edt(dtr).astype(np.uint16)
    
    # Calculate circle radii and their ratio
    in_rad = np.min(np.ma.array(dtr,mask=seg==label))   # Maximum inscribed circle radius 
    out_rad = np.max(np.ma.array(dtr,mask=seg!=label))  # Minimum circumscribed circle
    fspace[index,3] = in_rad / out_rad                  # Ratio  


# EXERCISE 2:
# Find and use a feature extraction algorithm that returns a large feature set
# for each cell. The features could for example be related to shape or texture.

### DAISY descriptors
# Scikit-image provides an implementation of DAISY, an algorithm for the
# detection and extraction of local image features (similar to SIFT) based on
# a grayscale image. This example shows that DAISY can be used to extract
# features for segmented cells. It should be pointed out, however, that our
# membrane images are not what DAISY would typically be applied to; it is best
# used for images with more local patterning/texture.
# Docs: scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.daisy

# Preparations and looping over cells
fspace2 = np.zeros((N,200))
from skimage.feature import daisy
for index,label in enumerate(labels):
    
    # Crop image to the cell of interest
    non_zero_indices = np.nonzero(seg==label)
    y_min = np.min(non_zero_indices[0])
    y_max = np.max(non_zero_indices[0])
    x_min = np.min(non_zero_indices[1])
    x_max = np.max(non_zero_indices[1])
    cell_img = img[y_min:y_max,x_min:x_max]
    
    # Extract DAISY features and build a feature space with a subset of them
    # Note that some of the segmentation artefacts can cause an error here;
    # these cells are marked as NaN (not a number) and are excluded below.
    try:
        daisy_features = daisy(cell_img, step=2, radius=8)[0]
        fspace2[index,:] = daisy_features[0,:]
    except Exception:
        fspace2[index,:] = np.NaN

# Exclusion of cells that gave errors.
# By deleting them from both the feature space and 'labels', the feature space
# remains mapped onto the segmentation correctly.
keep = []
for index,label in enumerate(labels):
    if not np.isnan(np.sum(fspace2[index,:])):
        keep.append(index)
labels2 = labels[keep]
fspace2 = fspace2[keep]


### SOME VISUALIZATION

# Transparent overlays can show how the features relate to the image (here: ROUNDNESS)
mapped = np.zeros_like(img,dtype=np.float)
for index,label in enumerate(labels):
    mapped[seg==label] = fspace[index,3] * 255
plt.imshow(img,cmap='gray',interpolation='none')
plt.imshow(mapped,interpolation='none',alpha=0.7) 
plt.show()

# Scatterplots can show relations of features
plt.scatter(fspace[:,0],fspace[:,1])
plt.xlabel("CELL SIZE")
plt.ylabel("STDEV OF MEM INT")
plt.show()

# Boxplots can show the spread of features
# Note the different scales of the features here. This is addressed in the next section!
plt.boxplot(fspace,labels=["CELL SIZE","STDEV OF MEM INT", "INT ASYM", "ROUNDNESS"])
plt.show()


# Note: You can save and later reload the features spaces you've generated so
# you don't need to run the feature extraction each time you run the script.

np.save(filename+"_fspace",fspace)
np.save(filename+"_fspace2",fspace2)
np.save(filename+"_labels2",labels2)
#"""

#------------------------------------------------------------------------------

### NORMALIZATION AND STANDARDIZATION
# Many classification and clustering algorithms need features to be normalized
# and/or standardized, otherwise the absolute size of the feature could affect 
# the result (for example, you could get a different result if you use cell 
# size in um or in pixels, because the absolute numbers are different).

# Normalization in this context generally means scaling your features to each
# range from 0 to 1. Standardization means centering the features around zero
# and scaling them to "unit variance" by dividing by their standard deviation.
# A more elaborate version of this which often provides a good starting point
# is a "whitening transform", which is implemented in the scipy.cluster module.

# It's worthwhile to read up on normalization/standardization so you avoid 
# introducing errors/biases. For example, normalization of data with outliers
# will compress the 'real' data into a very small range. Thus, outliers should
# be removed before normalization/standardization.

# EXERCISE 3:
# Find a way to remove outliers from your feature space.

# A simple approach: Compute the standard deviation and consider all values
# that are more than 3 sigma from the mean as outliers.
# Note: Assumes normal distribution of the data!

# Get data
fspace = np.load(filename+"_fspace.npy")

# Find outliers
stdevs = np.std(fspace,axis=0)
outliers = np.abs(fspace-np.mean(fspace,axis=0)) > 3 * stdevs

# Remove outliers (whilst preserving mapping of fspace to labels)
keep = np.where(np.sum(outliers,axis=1)==0)[0]
labels = labels[keep]
fspace = fspace[keep]


# EXERCISE 4: 
# Standardize, normalize and/or whiten your feature space as you deem fit,
# either by transforming the data yourself or using a module function.

# Center the data around the mean
fspace_c = fspace - np.mean(fspace,axis=0) 

# Whiten
from scipy.cluster.vq import whiten
fspace_w = whiten(fspace_c)


# Note: Don't forget to visualize your data again!

# The same boxplot as above shows the effect of our normalization very clearly.
plt.boxplot(fspace_w,labels=["CELL SIZE","STDEV OF MEM INT", "INT ASYM", "ROUNDNESS"])
plt.show()


#------------------------------------------------------------------------------

### PRINCIPAL COMPONENT ANALYSIS (PCA)
# The principal components of a feature space are the axes of greatest variance
# of the data. By transforming our data to this "relevance-corrected" coordinate
# system, we can achieve two things:
# 1) Usually, most of the variance in a dataset falls onto just a few principal
#    components, so we can ignore the other ones as irrelevant, thus reducing
#    the number of features whilst maintaining all information. This is very
#    useful to facilitate subsequent analyses.
# 2) Just PCA on its own can yield nice results. For example, different cell
#    populations that are not clearly separated by any single feature may 
#    appear separated along a principal component. Furthermore, principal 
#    components may correlate with other features of the data, which can be an
#    interesting result on its own.

# EXERCISE 5:
# Perform a PCA on your feature space and investigate the results.

# Hint: You may want to use the PCA implementation of scikit-learn
#       Algorithms in sklearn are provided as "estimator objects". The general
#       workflow for using them is to first instantiate the estimator object,
#       passing general parameters, then to fit the estimator to your data and
#       finally to extract various results from the fitted estimator.

### Performing PCA with scikit-learn
from sklearn.decomposition import PCA
pca_estimator = PCA(copy=True)   # Instantiate estimator
pca_estimator.fit(fspace_w)      # Fit data
print "PC-explained variance:", pca_estimator.explained_variance_ratio_  # Retrieve information
fspace_pca = pca_estimator.transform(fspace_w)                           # Transform data to PC space

### Some plotting

# Scatterplot of pc1 vs pc2
plt.scatter(fspace_pca[:,0],fspace_pca[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# Scatterplots of pc1 vs the features
# Reveals that features 2 (STDEV OF MEM INT) and 3 (INT ASYM) correlate well
# with PC1, hence indicating that PC1 characterizes the intensity distribution.
fig,axx = plt.subplots(2,2)
axx[0,0].scatter(fspace_w[:,0],fspace_pca[:,0])
axx[0,0].set_title("PC1 vs Feature 1")
axx[0,1].scatter(fspace_w[:,1],fspace_pca[:,0]) 
axx[0,1].set_title("PC1 vs Feature 2")
axx[1,0].scatter(fspace_w[:,2],fspace_pca[:,0]) 
axx[1,0].set_title("PC1 vs Feature 3")
axx[1,1].scatter(fspace_w[:,3],fspace_pca[:,0])
axx[1,1].set_title("PC1 vs Feature 4")
plt.show()


#------------------------------------------------------------------------------

### K-MEANS CLUSTERING
# If you expect that you can split your population into distinct groups, an 
# easy way of doing so in an unsupervised fashion is k-means clustering. 
# K-means partitions samples into clusters based on their proximity to the 
# cluster's mean.

# EXERCISE 6:
# Perform k-means clustering on your data. To do so, you have to assume the
# number of clusters. To begin with, just try it with 5 clusters. Try doing the
# PCA for raw, normalized and PCA-transformed data to see the difference. Don't 
# forget to visualize your result.


### K-means clustering using scipy

# Perform the actual clustering
k = 5
import scipy.cluster.vq as svq
k_centroids,k_labels = svq.kmeans2(fspace_pca,k)

# Visualization in PC space
plt.scatter(fspace_pca[:,0],fspace_pca[:,1],c=k_labels,s=30) # Color coded data
plt.scatter(k_centroids[:,0],k_centroids[:,1],s=150,c=range(k),marker='*',) # Centroids
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# Transparent overlay on the image
mapped = np.zeros_like(img,dtype=np.float)
for index,label in enumerate(labels):
    mapped[seg==label] = (k_labels[index] + 1) / np.max(k_labels+1) * 255
plt.imshow(img,cmap='gray',interpolation='none')
plt.imshow(mapped,interpolation='none',alpha=0.7) 
plt.show()


# ADDITIONAL EXERCISE:
# Can you think of and implement a simple way of objectively choosing the 
# number of clusters for k-means?

### A simple answer is the so-called "Elbow Plot"
# Assuming that we would like to minimize the intra-cluster standard dev with
# as few clusters as possible, we can plot the resulting stdev for different 
# numbers of clusters and choose the number above which the reduction in the
# stdev for additional clusters is marginal.

# Package the PCA into a function, together with the extraction of the stdev
def pca_stdev(data,k):
    
    # Perform the pca with the current number of clusters
    k_centroids,k_labels = svq.kmeans2(data,k)
    
    # For each cluster...
    stdevs = np.zeros(k)
    for i in range(k):
        
        # Find the data points associated to the cluster
        cluster_indices = np.where(k_labels==i)[0]
        cluster_coords = data[cluster_indices,:]
        
        # Calculate the distance of these datapoints from the centroid
        from scipy.spatial.distance import cdist
        cluster_dists = cdist(cluster_coords,np.expand_dims(k_centroids[i],0))
        
        # Calculate the stdev of these distances
        stdevs[i] = np.std(cluster_dists)
    
    # Average the stdevs for the different clusters
    full_stdev = np.mean(stdevs)
    
    return full_stdev

# Run for a range of cluster numbers
n_clusters = 13
k_stdevs = []
np.random.seed(999) # See note 2 below
for test_k in range(1,n_clusters):
    k_stdevs.append(pca_stdev(fspace_pca,test_k))

# Plot the elbow plot
plt.plot(range(1,n_clusters),k_stdevs)
plt.xlabel("Number of Clusters")
plt.ylabel("Average Cluster STDEV")
plt.show()

# Note 1: The resulting elbow plot is not particularly nice (no clean bend), 
#         which makes it difficult to choose the number of clusters and, more
#         importantly, indicates that this feature space cannot really be 
#         partitioned into sensible clusters, at least not with k-means. This 
#         is not unexpected, since we do not see separate populations when 
#         looking at the scatterplots of our features.
#         For a nice example of an elbow plot, see the result of the PCA in the
#         section "GRAPH-BASED ANALYSIS" below.

# Note 2: The kmeans implementation used here is initialized with random 
#         centroid positions that are then optimized by the algorithm. 
#         Consequently, you will get a slightly different result each time
#         you run the code. Here, I have ensured reproducibility by adding the
#         line "np.random.seed(999)", which will ensure that the random number
#         generator will produce the same starting positions each time.
#         However, to make sure that you get a representative elbow plot, it 
#         would make more sense to run the entire algorithm several times with
#         different seeds and then use the average elbow plot.


#------------------------------------------------------------------------------

### tSNE ANALYSIS
# Although PCA is great to reduce and visualize high-dimensional data, it only
# works well on linear relationships and global trends. Therefore, alternative 
# algorithms optimized for non-linear, local relationships have also been
# created.

# These algorithms tend to be quite complicated and going into them is beyond 
# the scope of this course. This example is intended as a taste of what is out
# there and to show people who already know about these methods that they are
# implemented in Python. Note that it can be risky to use these algorithms if 
# you do not know what you are doing, so it may make sense to read up and/or to 
# consult with an expert before you do this kind of analysis.

# This example uses the tSNE implementation in scikit-learn.

# As with the PCA, the first step is to instantiate an estimator object
# Note 1: Here I enforce a dimensionality reduction to the 2 most important
# components for illustration purposes. The same syntax could be used in the
# skimage implementation of pca to do dimensionality reduction.
# Note 2: random_state works much like numpy.random.seed.
from sklearn.manifold import TSNE
model = TSNE(n_components=2, random_state=999)

# Then we fit this model to our data and transform the data to the new
# frame of reference (here done in one step directly, unlike with the pca,
# where I showed how to do it in two steps)
fspace_tsne = model.fit_transform(fspace_w)

# Let's compare the result to the pca, colored for feature 2
fig,axx = plt.subplots(1,2)
axx[0].scatter(fspace_pca[:,0],fspace_pca[:,1],c=fspace_w[:,1])
axx[0].set_title("PC1 vs PC2")
axx[1].scatter(fspace_tsne[:,0],fspace_tsne[:,1],c=fspace_w[:,1])
axx[1].set_title("tSNE1 vs tSNE2")
plt.axis('equal')
plt.show()

# With the 4 features chosen, the two algorithms do not yield vastly different
# results. However, with different features or different raw data, it would be 
# a different story. Feel free to try it with your features (or with your own
# data - but again, make sure you know what you're looking at before making 
# any claims based on the result!).


#------------------------------------------------------------------------------

### GRAPH-BASED ANALYSIS
# Graphs are a universal way of mathematically describing relationships, be 
# they based on similarity, interaction, or virtually anything else. Despite 
# their power, graph-based analyses have so far not been used extensively on 
# biological imaging data, but as microscopes and analysis algorithms improve,
# they become increasingly feasible and will likely become very important in
# the future.

# The networkx module provides various functions for importing and generating
# graphs, for operating and analyzing graphs and for exporting and visualizing
# graphs. The following example shows how a simple graph based on our feature  
# space could be built and visualized. In doing so, it introduces the networkx 
# Graph object, which is the core of the networkx module.

### For this example, I will be using fspace2 generated by DAISY, so I first
### need to clean it and reduce its dimensionality, much like we did above.

### Outlier removal and normalization

# Get data
fspace2 = np.load(filename+"_fspace2.npy")
labels2 = np.load(filename+"_labels2.npy")

# Find outliers
# Note: This approach removes many cells, and is thus likely not appropriate
# for the DAISY features. However, I am keeping it for now since it does not
# really matter for this data.
stdevs = np.std(fspace2,axis=0)
outliers = np.abs(fspace2-np.mean(fspace2,axis=0)) > 3 * stdevs

# Remove outliers (whilst preserving mapping of fspace to labels)
keep = np.where(np.sum(outliers,axis=1)==0)[0]
labels2 = labels2[keep]
fspace2 = fspace2[keep]
print fspace2.shape
# Center the data around the mean
fspace2_c = fspace2 - np.mean(fspace2,axis=0) 

# Whiten
from scipy.cluster.vq import whiten
fspace2_w = whiten(fspace2_c)


### Now I'm using PCA as a dimensionality reduction from 200 features to 20, 
### based on a nice elbow plot of how much data variability each PC explains.

# Reduction to 50 components for plotting
from sklearn.decomposition import PCA
pca_estimator = PCA(n_components=50,copy=True)
pca_estimator.fit(fspace2_w)    

# Elbow plot
plt.plot(range(1,51),pca_estimator.explained_variance_ratio_)
plt.xlabel("PC")
plt.ylabel("Explained Variance Ratio")
plt.show()

# Reduction to 20 components and transformation for use in tSNE
pca_estimator = PCA(n_components=20,copy=True)
pca_estimator.fit(fspace2_w)  
fspace2_pca = pca_estimator.transform(fspace2_w)


### Now I'm running tSNE to reduce the data to two features, which allows
### plotting it in 2D

# tSNE
from sklearn.manifold import TSNE
model = TSNE(n_components=2, random_state=999)
fspace2_tsne = model.fit_transform(fspace2_pca)

# PCA for comparison
pca_model = PCA(n_components=2,copy=True)
fspace2_pca2d = pca_model.fit_transform(fspace2_pca)  

# Let's compare the result to the pca, colored for feature 2
fig,axx = plt.subplots(1,2)
axx[0].scatter(fspace2_pca2d[:,0],fspace2_pca2d[:,1],c=fspace2_w[:,1])
axx[0].set_title("PC1 vs PC2")
axx[1].scatter(fspace2_tsne[:,0],fspace2_tsne[:,1],c=fspace2_w[:,1])
axx[1].set_title("tSNE1 vs tSNE2")
plt.axis('equal')
plt.show()


### Now we begin the construction of the actual graph
# As mentioned above, graphs are a very generic way of representing relations,
# so there are also many ways of generating a graph. Here, I am thresholding
# the pairwise euclidean distance of the cells after dimensionality reduction
# to 20 features by PCA. In other words, I draw an edge between two nodes in 
# the graph if they are sufficiently close together in feature space.
# The easiest way to construct such a graph is to generate an "adjacency
# matrix", a symmetric array of shape (n_cells,n_cells) where 1 denotes an
# edge and 0 no edge between the cells of the corresponding row/column.

# Calculating pairwise distances
from scipy.spatial.distance import pdist,squareform
graph_pdists = squareform(pdist(fspace2_pca))

# Thresholding pairwise distances
# The threshold is chosen such that the closest 10% of cells will be connected.
# The resulting array is the adjacency matrix of our graph.
graph_thresh = graph_pdists < np.percentile(graph_pdists,10)

# Cleanup: Since the matrix is symmetric, remove one half of it (and half the diagonal)
graph_thresh = np.triu(graph_thresh,k=1)

# Display the resulting adjacency matrix
plt.imshow(graph_thresh,interpolation='none',cmap='gray')
plt.show()


### Bringing in networkx

# From the adjacency matrix, we can generate a networkx graph object
import networkx as nx
G = nx.from_numpy_matrix(graph_thresh)

# Once we have a networkx graph object, there are many possibilities for 
# operating on, analyzing, partitioning and visualizing this graph. Here, I 
# will limit myself to showing some examples of visualization.

# Display graph, get positions based on graph itself (force-based distribution)
# and colors from PC1
pos = nx.fruchterman_reingold_layout(G)
nx.draw(G,pos=pos,cmap=plt.get_cmap('gist_rainbow'),node_color=fspace2_pca[:,0])
plt.show()

# Display graph, get positions from PC1 and PC2 and colors from PC1
pos_dict_pca = {}
for index in range(len(labels2)):
    pos_dict_pca[index] = fspace2_pca2d[index,:]
nx.draw(G,pos=pos_dict_pca,cmap=plt.get_cmap('gist_rainbow'),node_color=fspace2_pca[:,0])
plt.show()

# Display graph nicely overlayed over cells and segmentation

# Use centroids as node positions
pos_dict_img = {}
for index,label in enumerate(labels2):
    cen_coords = ndi.measurements.center_of_mass(seg==label)
    pos_dict_img[index] = (cen_coords[1],cen_coords[0])

# Map PC1 onto the segmentation
pca_mapped = np.zeros_like(seg)
for index,label in enumerate(labels2):
    pca_mapped[seg==label] = fspace2_pca[index,0]
    
# Generate image and overlay
plt.imshow(img,interpolation='none',cmap='gray')
plt.imshow(np.ma.array(pca_mapped,mask=pca_mapped==0),interpolation='none',cmap=plt.get_cmap('gist_rainbow'),alpha=0.5)
nx.draw(G,pos=pos_dict_img,cmap=plt.get_cmap('gist_rainbow'),node_size=100,node_color=fspace2_pca[:,0],alpha=0.5,edge_color='0.5')
plt.tight_layout()
plt.show()

# Note that graph visualization is not an easy task and the use of dedicated 
# software such as Cytoscape may be advisable on occasion. Networkx supports 
# export of Graphs to various other software


#------------------------------------------------------------------------------

### BONUS: XCKD-STYLE PLOTS

# CONGRATULATIONS! You made it to the very end of this debaucherous tutorial,
# so you now get to see what is probably the most fantastic functionality in
# matplotlib: plotting in the style of the xkcd webcomic!

# Days spent programming
t = np.arange(0,600)

# Average quality of code produced
q = np.zeros_like(t)
q[0:100] = t[0:100] * 2                       # When you're a total n00b.
q[100:200] = q[99] + np.arange(100) ** 2.0    # When everything is suddenly amazing.
q[200:600] = np.random.randint(0,q[199]*2.0,80).repeat(5) # Aaaaand... when you've gone insane. 

# Plotting in xkcd-style
plt.xkcd(scale=0.5,length=100,randomness=1)
plt.plot(t,q)
plt.xlabel("Days Spent Programming")
plt.ylabel("Average Quality of Code Produced")
plt.tick_params(axis='both', which='both',bottom='off',top='off',left='off',right='off')
plt.show()


