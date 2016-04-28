# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 00:12:38 2015

@author:    Jonas Hartmann @ Gilmour Group @ EMBL Heidelberg
            Karin Sasaki @ CBM @ EMBL Heidelberg
            
@descript:  This is the batch version of 'tutorial_pipeline.py', which is an  
            example pipeline for the segmentation of 2D confocal fluorescence 
            microscopy images of a membrane marker in confluent epithel-like 
            cells. This batch version serves to illustrate how such a pipeline
            can be run automatically on multiple images.
            
            The pipeline is optimized to run with the provided example images,
            which are dual-color spinning-disc confocal micrographs (40x) of
            two membrane-localized proteins during zebrafish early embryonic
            development (~10hpf).
            
            'tutorial_pipeline.py' shows how to make the pipeline. In tis tutorial 
            we show you how to run the pipeline on a batch of images that are saved
            in the current directory.

@requires:  Python 2.7
            NumPy 1.9, SciPy 0.15
            scikit-image 0.11.2, tifffile 0.3.1
"""

#%%
#------------------------------------------------------------------------------
#  Step 0 - (RE)CHECK PIPELINE WORKS

# You should already have the final pipeline that segments and quantifies one of the example images example_cells_1.png or example_cells_2.png. 
# Make sure it is working, by running it (on IPython, type %run file_name, on the terminal, type python file_name). Check that you do not get errors and the output is what you expect.


#%%
#------------------------------------------------------------------------------
# Step 1 - BATCH VERSION: FUNCTIONALIZED PIPELINE

#------------------------
# Part A - Import required packages and modules for pipeline built
#------------------------

from __future__ import division    # Python 2.7 legacy
import numpy as np                 # Array manipulation package
import matplotlib.pyplot as plt    # Plotting package
import scipy.ndimage as ndi        # Image processing package


#------------------------
# Part B - Define the pipeline function
#------------------------

#   Define a function that takes the one input, a string filename, and returns two arguments: the final segmentation and the quantified data. 


def pipeline(filename):
""" Batch version of tutorial_pipeline.py """

#%%
#------------------------------------------------------------------------------
# SECTION 2 - IMPORT AND SLICE DATA
    
    # Check input file exists. If it does not, then warn and abort, otherwise report that the pipeline is being executed
    from os.path import isfile
    if not isfile(filename):
        from warnings import warn
        warn("Could not find file" + filename + '.tif')
        return
    else:
        print "Starting pipeline for", filename
    
    # Import tif files
    import skimage.io as io               # Image file manipulation module
    img = io.imread(filename)             # Importing multi-color tif file
    
    # Check that everything is in order
    print "  Loaded array has shape", img.shape               # Printing array shape; 2 colors, 930 by 780 pixels
    
    # Slicing: We only work on one channel for segmentation
    green = img[0,:,:]
    
    
#%%    #------------------------------------------------------------------------------
# SECTION 3 - PREPROCESSING: (I) SMOOTHING AND (II) ADAPTIVE THRESHOLDING
# It's standard to smoothen images to reduce technical noise - this improves
# all subsequent image processing steps. Adaptive thresholding allows the
# masking of foreground objects even if the background intensity varies across
# the image.

# -------
# Part I
# -------
    
# Gaussian smoothing    
    sigma = 3                                                # Smoothing factor for Gaussian
    green_smooth = ndi.filters.gaussian_filter(green,sigma)  # Perform smoothing
    
    
# -------
# Part II
# -------
    
    # Create an adaptive background
    struct = ((np.mgrid[:31,:31][0] - 15)**2 + (np.mgrid[:31,:31][1] - 15)**2) <= 15**2  # Create a disk-shaped structural element
    bg = ndi.filters.generic_filter(green_smooth,np.mean,footprint=struct)               # Run a mean filter over the image using the disc
    
    # Threshold using created background
    green_thresh = green_smooth >= bg
    
    # Clean by morphological hole filling
    green_thresh = ndi.binary_fill_holes(np.logical_not(green_thresh))
    
    
#%%
#------------------------------------------------------------------------------
# SECTION 5 - SEGMENTATION OF CELLS: (I) SEEDING BY DISTANCE TRANSFORM AND (II) EXPANSION BY WATERSHED

# More advanced segmentation is usually a combination of seeding and expansion.
# In seeding, we want to find a few pixels for each cell that we can assign to
# said cell with great certainty. These 'seeds' are then expanded to partition
# regions of the image where cell affiliation is less clear-cut.

# -------
# Part I
# -------
    
    # Distance transform on thresholded membranes
    # Advantage of distance transform for seeding: It is quite robust to local 
    # "holes" in the membranes.
    green_dt= ndi.distance_transform_edt(green_thresh)
    
    # Dilating (maximum filter) of distance transform improves results
    green_dt = ndi.filters.maximum_filter(green_dt,size=10) 
    
    # Retrieve and label the local maxima
    from skimage.feature import peak_local_max
    green_max = peak_local_max(green_dt,indices=False,min_distance=10)  # Local maximum detection
    green_max = ndi.label(green_max)[0]                                 # Labeling
    


# -------
# Part II
# -------

# Watershedding is a relatively simple but powerful algorithm for expanding
# seeds. The image intensity is considered as a topographical map (with high
# intensities being "mountains" and low intensities "valleys") and water is
# poured into the valleys from each of the seeds. The water first labels the
# lowest intensity pixels around the seeds, then continues to fill up. The cell
# boundaries are where the waterfronts between different seeds touch.

    
    # Get the watershed function and run it
    from skimage.morphology import watershed
    green_ws = watershed(green_smooth,green_max)
    
    
     #%%
     #------------------------------------------------------------------------------
# SECTION 6 - SEGMENTATION OF CELL EDGES: 
# Finding cell edges is very useful for many purposes. In our example, edge
# intensities are a measure of membrane intensities, which may be a desired
# readout. The length of the edge (relative to cell size) is also a quite
# informative feature about the cell shape. Finally, showing colored edges is
# a nice way of visualizing segmentations.

    
    # Define the edge detection function
    def edge_finder(footprint_values):
        if (footprint_values == footprint_values[0]).all():
            return 0
        else:
            return 1
        
    # Iterate the edge finder over the segmentation
    green_edges = ndi.filters.generic_filter(green_ws,edge_finder,size=3)
    
    # Label the detected edges based on the underlying cells
    green_edges_labeled = green_edges * green_ws    
    
#%%
#------------------------------------------------------------------------------
# SECTION 7 - POSTPROCESSING: REMOVING CELLS AT THE IMAGE BORDER
# Since segmentation is never perfect, it often makes sense to remove artefacts
# after the segmentation. For example, one could filter out cells that are too
# big, have a strange shape, or strange intensity values. Similarly, supervised 
# machine learning can be used to identify cells of interest based on a 
# combination of various features. Another example of cells that should be 
# removed are those at the image boundary.
    
       # Create a mask for the image boundary pixels
    boundary_mask = np.ones_like(green_ws)   # Initialize with all ones
    boundary_mask[1:-1,1:-1] = 0             # Set middle square to 0
    
       # Iterate over all cells in the segmentation
    current_label = 1
    for cell_id in np.unique(green_ws):
        
           # If the current cell touches the boundary, remove it
        if np.sum((green_ws==cell_id)*boundary_mask) != 0:
            green_ws[green_ws==cell_id] = 0
            
           # This is to keep the labeling continuous, which is cleaner
        else:
            green_ws[green_ws==cell_id] = current_label
            current_label += 1
        
    
#%%
#------------------------------------------------------------------------------
# SECTION 8 - MEASUREMENTS: SINGLE-CELL AND MEMBRANE READOUTS

# Now that the cells in the image are nicely segmented, we can quantify various
# readouts for every cell individually. Readouts can be based on the intensity
# in the original image, on intensities in other channels or on the size and
# shape of the cells themselves.
    
    # Initialize a dict for results of choice
    results = {"cell_id":[], "green_mean":[], "red_mean":[],"green_mem_mean":[], 
               "red_mem_mean":[],"cell_size":[],"cell_outline":[]}
    
    # Iterate over segmented cells
    for cell_id in np.unique(green_ws)[1:]:
        
        # Mask the pixels of the current cell
        cell_mask = green_ws==cell_id    
        
        # Get the current cell's values
        # Note that the original raw data is used for quantification!
        results["cell_id"].append(cell_id)
        results["green_mean"].append(np.mean(img[0,:,:][cell_mask]))
        results["red_mean"].append(np.mean(img[1,:,:][cell_mask]))
        results["green_mem_mean"].append(np.mean(img[0,:,:][np.logical_and(cell_mask,green_edges)]))
        results["red_mem_mean"].append(np.mean(img[1,:,:][np.logical_and(cell_mask,green_edges)]))    
        results["cell_size"].append(np.sum(cell_mask))
        results["cell_outline"].append(np.sum(np.logical_and(cell_mask,green_edges)))
    
    
    
#%%    
#------------------------------------------------------------------------------
# SECTION 9 - REPORT END OF PIPELINE

#   In order to keep track of when the function has completed for one image, at the end of the function print a statement that clearly indicates this.
    
    print "  Completed pipeline for", filename
    
#%%    
#------------------------------------------------------------------------------
# SECTION 10 - RETURN PARSMETERS


    return green_ws, results 
    
    
    
    
#%%   
#------------------------------------------------------------------------------
# Step 2 -  BATCH VERSION: EXECUTION SCRIPT

# Now that the pipeline function is defined, we can run it for each image file
# in a directory and collect the results as they are returned.


#------------------------
# Part A - Get the current working directory
#------------------------

#   Define a variable input_dir with the path to the directory whether the images are saved. 
from os import getcwd, listdir   
input_dir = getcwd()         


#------------------------
# Part B - Generate a list of image filenames
#------------------------

# Make a list of files in the directory
filelist = listdir(input_dir)

# Collect the file names only for files that are tifs 
tiflist = [filename for filename in filelist if filename[-4:]=='.tif'] 

# Check that you have the right files in tiflist. 
print "Found", len(tiflist), "tif files in target directory\n"


#------------------------
# Part C/D - Loop over the tiflist, run the pipeline and collect the results
#------------------------

# Initialise two dicts, all_results and all_segmentations, where you will collect the output of the pipeline, for each image.
all_results = []
all_segmentations = []

#   Initialise counter for sucessfully ran the pipeline
success_counter = 0

#   Write a for loop that goes through every file in the tiflist. Run the function pipeline for each file and allocate the output to new variables. Add the output to the variables all_results and all_segmentations, usign append.
for filename in tiflist:
    try:  # We may not want everything to fail if there is an error in just one file
        seg,results = pipeline(filename)
        all_results.append(results)
        all_segmentations.append(seg)
        success_counter += 1
    except Exception:
        from warnings import warn
        warn("There was an exception in " + filename + "!!!")

        
#------------------------
# Part E - Print a short summary  
#------------------------

    # How many succesfully analysied images
print "\nSuccessfully analyzed", success_counter, "of", len(tiflist), "images"

    # How many cells segmented.
num_cells = 0
for resultDict in all_results:
    num_cells = num_cells + len(resultDict["cell_id"])
    
print "Detected", num_cells, "cells in total"



#------------------------
# Part F - Quick visualisation of results
#------------------------

# To give cells from different images different colors
colors = plt.cm.jet(np.linspace(0,1,len(all_results)))

# Plot a scatter plot for all data and save image.  
for image_id,resultDict in enumerate(all_results):
    plt.scatter(resultDict["cell_size"],resultDict["red_mem_mean"],color=colors[image_id])
#plt.xlabel("cell size")
#plt.ylabel("red membrane mean")
#plt.show()
plt.savefig('all_cells_scatter.png', bbox_inches='tight')  


#------------------------
# Part G - Save all the segmentations as a "3D" tif
#------------------------

#   Convert all_segmentations to a numpy array
all_segmentations = np.array(all_segmentations)

#   Save the image to a tif file using imsave from tifffile
from tifffile import imsave
imsave("BATCH_segmentations.tif",all_segmentations,bigtiff=True)


#------------------------
# Part H - Save the data as a txt file 
#------------------------

# Save the data as a txt file (for excel etc...)
with open("BATCH_results.txt","w") as txt_out:                         # Open an empty file 
    txt_out.write(''.join(key+'\t' for key in results.keys()) + '\n')  # Write the headers
    for image_id,filename in enumerate(tiflist):                       # For each analyzed image...
        txt_out.write(filename + "\n")                                 # ...write the filename
        resultDict = all_results[image_id]                             # ...extract the corresponding results                   
        for index,value in enumerate(resultDict["cell_id"]):           # ...iterate over cells...
            txt_out.write(''.join(str(resultDict[key][index])+'\t' for key in resultDict.keys()) + '\n')   # ...and write cell data
    
   
    
#%%   
#------------------------------------------------------------------------------
# Step 4 -  (OPTIONAL EXERCISE) RATIOMETRIC NORMALIZATION TO CONTROL CHANNEL

# To correct for technical variability it is often useful to have an internal  control (e.g. some fluorophore that we expect to be the same between all   analyzed conditions) and then normalize other measurements to that control.

# For example, we can assume that our green channel is just a generic membrane  marker, whereas the red channel is a labelled protein of interest. Thus, using the red/green ratio instead of the raw values from the red channel may yield a clearer result when comparing intensity measurements of the red protein of interest between different conditions.

#------------------------
# Part A - Create the ratio
#------------------------

for image_id,resultDict in enumerate(all_results):
    all_results[image_id]["red_green_mem_ratio"] = [resultDict["red_mem_mean"][i] / resultDict["green_mem_mean"][i] for i in range(len(resultDict["cell_id"]))]


#------------------------
# Part B - Make a scatter plot, this time with the ratio
#------------------------

colors = plt.cm.jet(np.linspace(0,1,len(all_results))) # To give cells from different images different colors
for image_id,resultDict in enumerate(all_results):
    plt.scatter(resultDict["cell_size"],resultDict["red_green_mem_ratio"],color=colors[image_id])
#plt.xlabel("cell size")
#plt.ylabel("red_mem_mean")
plt.savefig('all_cells_ratio_scatter.png', bbox_inches='tight')  


#------------------------
# Part C - Analysis
#------------------------

# What can you conclude from this output?

# Note: This doesn't really make much of a difference here (except for some
#       outliers; artefacts), but depending on the type of data and the 
#       question, normalizing with internal controls can be crucial to arrive
#       at the correct conclusion!

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# THIS IS THE END OF THE TUTORIAL.
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
