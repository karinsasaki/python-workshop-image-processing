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
            SciPy 0.16.0, scikit-learn 0.15.2, networksx 1.9.1
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







# EXERCISE 2:
# Find and use a feature extraction algorithm that returns a large feature set
# for each cell. The features could for example be related to shape or texture.








# Note: You can save and later reload the features spaces you've generated with
# np.save and np.load, so you don't need to run the feature extraction each 
# time you run the script.



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









# EXERCISE 4: 
# Standardize, normalize and/or whiten your feature space as you deem fit,
# either by transforming the data yourself or using a module function.








# Note: Don't forget to visualize your data again and compare to the raw data!



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











# ADDITIONAL EXERCISE:
# Can you think of and implement a simple way of objectively choosing the 
# number of clusters for k-means?











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

# This is not an exercise, just an example for you to study.
# You can find the code in the solutions file.



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

# This is not an exercise, just an example for you to study.
# You can find the code in the solutions file.



#------------------------------------------------------------------------------

### BONUS: XCKD-STYLE PLOTS

# CONGRATULATIONS! You made it to the very end of this debaucherous tutorial,
# so you now get to see what is probably the most fantastic functionality in
# matplotlib: plotting in the style of the xkcd webcomic!

# This is not an exercise, just an example for you to study.
# You can find the code in the solutions file.


