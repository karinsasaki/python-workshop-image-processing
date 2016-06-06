# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 00:12:38 2015

@author:    Jonas Hartmann @ Gilmour Group @ EMBL Heidelberg

@descript:  This is the batch version of 'py_examplePipeline.py', which is an  
            example pipeline for the segmentation of 2D confocal fluorescence 
            microscopy images of a membrane marker in confluent epithel-like 
            cells. This batch version serves to illustrate how such a pipeline
            can be run automatically on multiple images.
            
            The pipeline is optimized to run with the provided example images,
            which are dual-color spinning-disc confocal micrographs (40x) of
            two membrane-localized proteins during zebrafish early embryonic
            development (~10hpf).

@requires:  Python 2.7
            NumPy 1.9, SciPy 0.15
            scikit-image 0.11.2, tifffile 0.3.1
"""


# IMPORT STUFF
# Some basic data/image manipulation tools will be used throughout the script,
# So it makes sense to import them at the start.

from __future__ import division    # Python 2.7 legacy
import numpy as np                 # Array manipulation package
import matplotlib.pyplot as plt    # Plotting package
import scipy.ndimage as ndi        # Image processing package


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

""" BATCH VERSION: FUNCTIONALIZED PIPELINE """
# The entire pipeline below is now fitted into a function that takes the input
# filename as an argument and returns the final segmentation and the measured
# data. Any instances where an image would be shown have been removed so that
# the pipeline can continue running without user supervision

def pipeline(filename):


    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    # IMPORT AND SLICE DATA
    
    """ BATCH VERSION: CHECK IF INPUT FILENAME EXISTS """
    # If it does not, then warn and abort,
    # otherwise report that the pipeline is being executed
    from os.path import isfile
    if not isfile(filename):
        from warnings import warn
        warn("Could not find file" + filename + '.tif')
        return
    else:
        print "Starting pipeline for", filename
    
    # Import tif files
    import skimage.io as io               # Image file manipulation module
    img = io.imread(filename)         # Importing multi-color tif file
    img = np.array(img)                   # Converting MultiImage object to numpy array
    
    # Check that everything is in order
    print "  Loaded array has shape", img.shape               # Printing array shape; 2 colors, 930 by 780 pixels
#    plt.imshow(img[0,:,:],interpolation='none',cmap='gray')   # Showing one of the channels (notice "interpolation='none'"!)
#    plt.show()
    
    # Slicing: We only work on one channel for segmentation
    green = img[0,:,:]
    
    
    #------------------------------------------------------------------------------
    
    # PREPROCESSING: SMOOTHING AND ADAPTIVE THRESHOLDING
    # It's standard to smoothen images to reduce technical noise - this improves
    # all subsequent image processing steps. Adaptive thresholding allows the
    # masking of foreground objects even if the background intensity varies across
    # the image.
    
    # Gaussian smoothing
    sigma = 3                                                # Smoothing factor for Gaussian
    green_smooth = ndi.filters.gaussian_filter(green,sigma)  # Perform smoothing
    
    # Create an adaptive background
    #struct = ndi.iterate_structure(ndi.generate_binary_structure(2,1),24)               # Create a diamond-shaped structural element
    struct = ((np.mgrid[:31,:31][0] - 15)**2 + (np.mgrid[:31,:31][1] - 15)**2) <= 15**2  # Create a disk-shaped structural element
    bg = ndi.filters.generic_filter(green_smooth,np.mean,footprint=struct)               # Run a mean filter over the image using the disc
    
    # Threshold using created background
    green_thresh = green_smooth >= bg
    
    # Clean by morphological hole filling
    green_thresh = ndi.binary_fill_holes(np.logical_not(green_thresh))
    
    # Show the result
#    plt.imshow(green_thresh,interpolation='none',cmap='gray')
#    plt.show()
    
    
    #------------------------------------------------------------------------------
    
    # (SIDE NOTE: WE COULD BE DONE NOW)
    # If the data is very clean and/or we just want a quick look, we could simply
    # label all connected pixels now and consider the result our segmentation.
    
    # Labeling connected components
#    green_label = ndi.label(green_thresh)[0]
#    plt.imshow(green_label,interpolation='none')
#    plt.show() 
    
    # However, to also partition the membranes to the cells, to generally improve  
    # the segmentatation (e.g. split cells that end up connected here) and to 
    # handle more complicated morphologies or to deal with lower quality data, 
    # this approach is not sufficient.
    
    
    #------------------------------------------------------------------------------
    
    # SEGMENTATION: SEEDING BY DISTANCE TRANSFORM
    # More advanced segmentation is usually a combination of seeding and expansion.
    # In seeding, we want to find a few pixels for each cell that we can assign to
    # said cell with great certainty. These 'seeds' are then expanded to partition
    # regions of the image where cell affiliation is less clear-cut.
    
    # Distance transform on thresholded membranes
    # Advantage of distance transform for seeding: It is quite robust to local 
    # "holes" in the membranes.
    green_dt= ndi.distance_transform_edt(green_thresh)
#    plt.imshow(green_dt,interpolation='none')
#    plt.show()
    
    # Dilating (maximum filter) of distance transform improves results
    green_dt = ndi.filters.maximum_filter(green_dt,size=10) 
#    plt.imshow(green_dt,interpolation='none')
#    plt.show()
    
    # Retrieve and label the local maxima
    from skimage.feature import peak_local_max
    green_max = peak_local_max(green_dt,indices=False,min_distance=10)  # Local maximum detection
    green_max = ndi.label(green_max)[0]                                 # Labeling
    
    # Show maxima as masked overlay
#    plt.imshow(green_smooth,cmap='gray',interpolation='none')
#    plt.imshow(np.ma.array(green_max,mask=green_max==0),interpolation='none') 
#    plt.show()
    
    
    #------------------------------------------------------------------------------
    
    # SEGMENTATION: EXPANSION BY WATERSHED
    # Watershedding is a relatively simple but powerful algorithm for expanding
    # seeds. The image intensity is considered as a topographical map (with high
    # intensities being "mountains" and low intensities "valleys") and water is
    # poured into the valleys from each of the seeds. The water first labels the
    # lowest intensity pixels around the seeds, then continues to fill up. The cell
    # boundaries are where the waterfronts between different seeds touch.
    
    # Get the watershed function and run it
    from skimage.morphology import watershed
    green_ws = watershed(green_smooth,green_max)
    
    # Show result as transparent overlay
    # Note: For a better visualization, see "FINDING CELL EDGES" below!
#    plt.imshow(green_smooth,cmap='gray',interpolation='none')
#    plt.imshow(green_ws,interpolation='none',alpha=0.7) 
#    plt.show()
    
    # Notice that the previously connected cells are now mostly separated and the
    # membranes are partitioned to their respective cells. 
    # ...however, we now see a few cases of oversegmentation!
    # This is a typical example of the trade-offs one has to face in any 
    # computational classification task. 
    
    
    #------------------------------------------------------------------------------
    
    # POSTPROCESSING: REMOVING CELLS AT THE IMAGE BORDER
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
    
    # Show result as transparent overlay
#    plt.imshow(green_smooth,cmap='gray',interpolation='none')
#    plt.imshow(np.ma.array(green_ws,mask=green_ws==0),interpolation='none',alpha=0.7) 
#    plt.show()
    
    
    #------------------------------------------------------------------------------
    
    # MEASUREMENTS: FINDING CELL EDGES
    # Finding cell edges is very useful for many purposes. In our example, edge
    # intensities are a measure of membrane intensities, which may be a desired
    # readout. The length of the edge (relative to cell size) is also a quite
    # informative feature about the cell shape. Finally, showing colored edges is
    # a nice way of visualizing segmentations.
    
    # How this works: The generic_filter function (see further below) iterates a 
    # structure element (in this case a 3x3 square) over an image and passes all
    # the values within that element to some arbitrary function (in this case 
    # edge_finder). The edge_finder function checks if all these pixels are the 
    # same; if they are, the current pixel is not at an edge (return 0), otherwise 
    # it is (return 1). generic_filter takes the returned values and organizes them
    # into an image again by setting the central pixel of each 3x3 square to the
    # respective return value from edge_finder.
    
    # Define the edge detection function
    def edge_finder(footprint_values):
        if (footprint_values == footprint_values[0]).all():
            return 0
        else:
            return 1
        
    # Iterate the edge finder over the segmentation
    green_edges = ndi.filters.generic_filter(green_ws,edge_finder,size=3)
    
    # Label the detected edges based on the underlying cells
#    green_edges_labeled = green_edges * green_ws
    
    # Show them as masked overlay
#    plt.imshow(green_smooth,cmap='gray',interpolation='none')
#    plt.imshow(np.ma.array(green_edges_labeled,mask=green_edges_labeled==0),interpolation='none') 
#    plt.show()
    
    
    #------------------------------------------------------------------------------
    
    # MEASUREMENTS: SINGLE-CELL READOUTS
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
    
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------

    """ BATCH VERSION: RETURN RESULTING SEGMENTATION AND DATA """
    
    print "  Completed pipeline for", filename
    return green_ws, results


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

""" BATCH VERSION: EXECUTION SCRIPT """
# Now that the pipeline function is defined, we can run it for each image file
# in a directory and collect the results as they are returned.


# User-specified input directory (if None, working directory is used)
input_dir = "C:\Users\Jonas Hartmann\Data\c) EMBL PhD\Z Side Projects\Karin GitLab\GIT_Karin MATLAB IP Tutorial"


# Generate a list of image filenames
from os import getcwd, listdir   # System tools
if not input_dir:                # Check if user specified an input directory
    input_dir = getcwd()         # Otherwise use the working directory
filelist = listdir(input_dir)    # List of files in the directory
tiflist = [filename for filename in filelist if filename[-4:]=='.tif']   # Get tifs only
print "Found", len(tiflist), "tif files in target directory\n"           # Report


# Loop over the tiflist, run the pipeline and collect the results
all_results = []
all_segmentations = []
success_counter = 0
for filename in tiflist:
    try:  # We may not want everything to fail if there is an error in just one file
        seg,results = pipeline(filename)
        all_results.append(results)
        all_segmentations.append(seg)
        success_counter += 1
    except Exception:
        from warnings import warn
        warn("There was an exception in " + filename + "!!!")

        
# Print a short summary  
print "\nSuccessfully analyzed", success_counter, "of", len(tiflist), "images"
print "Detected", sum([len(resultDict["cell_id"]) for resultDict in all_results]), "cells in total"


# Show one of our classical scatter plots for all data
colors = plt.cm.jet(np.linspace(0,1,len(all_results))) # To give cells from different images different colors
for image_id,resultDict in enumerate(all_results):
    plt.scatter(resultDict["cell_size"],resultDict["red_mem_mean"],color=colors[image_id])
plt.xlabel("cell size")
plt.ylabel("red_mem_mean")
plt.show()


# Ratiometric normalization to control channel
# To correct for technical variability it is often useful to have an internal
# control (e.g. some fluorophore that we expect to be the same between all 
# analyzed conditions) and then normalize other measurements to that control.

# For example, we can assume that our green channel is just a generic membrane
# marker, whereas the red channel is a labeled protein of interest.
# Thus, using the red/green ratio instead of the raw values from the red 
# channel may yield a clearer result when comparing intensity measurements of 
# the red protein of interest between different conditions.

# TODO: A better example for this (where it actually makes a difference) would
#       be nice. This depends on the example data used at the end, though.

# The ratio could be created like this
for image_id,resultDict in enumerate(all_results):
    all_results[image_id]["red_green_mem_ratio"] = [resultDict["red_mem_mean"][i] / resultDict["green_mem_mean"][i] for i in range(len(resultDict["cell_id"]))]

# Show the same scatter plot again but with the ratio
# Note: This doesn't really make much of a difference here (except for some
#       outliers; artefacts), but depending on the type of data and the 
#       question, normalizing with internal controls can be crucial to arrive
#       at the correct conclusion!
colors = plt.cm.jet(np.linspace(0,1,len(all_results))) # To give cells from different images different colors
for image_id,resultDict in enumerate(all_results):
    plt.scatter(resultDict["cell_size"],resultDict["red_green_mem_ratio"],color=colors[image_id])
plt.xlabel("cell size")
plt.ylabel("red_mem_mean")
plt.show()


# Save all the segmentations as a "3D" tif
all_segmentations = np.array(all_segmentations)
from tifffile import imsave
imsave("BATCH_segmentations.tif",all_segmentations,bigtiff=True)


# Save the data as a txt file (for excel etc...)
with open("BATCH_results.txt","w") as txt_out:                         # Open an empty file 
    txt_out.write(''.join(key+'\t' for key in results.keys()) + '\n')  # Write the headers
    for image_id,filename in enumerate(tiflist):                       # For each analyzed image...
        txt_out.write(filename + "\n")                                 # ...write the filename
        resultDict = all_results[image_id]                             # ...extract the corresponding results                   
        for index,value in enumerate(resultDict["cell_id"]):           # ...iterate over cells...
            txt_out.write(''.join(str(resultDict[key][index])+'\t' for key in resultDict.keys()) + '\n')   # ...and write cell data
    
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------



