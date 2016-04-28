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
            
            'tutorial_pipeline.py' shows how to make the pipeline.

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

# Since you know the pipeline requires certain modules and packages, you 
# should import those here, at the beginning of the program.

from __future__ import division # Python 2.7 legacy
import --- as np                # Array manipulation package numpy as np
import --- as plt               # Plotting package matplotlib.pyplot as plt
import --- as ndi               # Image processing package scipy.ndimage as ndi

#------------------------
# Part B - Define the pipeline function
#------------------------

# Define a function that: 
# - takes one input, a string filename, 
# - returns two arguments, the final segmentation and the quantified data,
# - reports when the pipeline has finalised.
#
# To do this, you need to copy the pipeline you developed, up to section 8 (you can also exclude section 4), and paste it inside the function.
# In order to have the pipeline continue running without user supervision, remove (or comment out) any instances where an image would be shown.

# Recall that to define a new function the syntax is 
# def function_name(input arguments): 
#   """function documentation string"""
#   function procedure
#   return [expression]
#

--- pipeline(filename):

#%%
#------------------------------------------------------------------------------
# SECTION 2 - IMPORT AND SLICE DATA
    
    """ BATCH VERSION """

    # Report that the pipelineis being executed
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
    
    
#%%
#------------------------------------------------------------------------------
# SECTION 4 - (SIDE NOTE: WE COULD BE DONE NOW)
# If the data is very clean and/or we just want a quick look, we could simply
# label all connected pixels now and consider the result our segmentation.
    
    
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
    
    
     #%%
     #------------------------------------------------------------------------------
# SECTION 6 - SEGMENTATION OF CELL EDGES: 
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
    
       # Show result as transparent overlay
   #    plt.imshow(green_smooth,cmap='gray',interpolation='none')
   #    plt.imshow(np.ma.array(green_ws,mask=green_ws==0),interpolation='none',alpha=0.7) 
   #    plt.show()
    
    
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

# Now that the pipeline function is defined, we can run it for each image file in a directory and collect the results as they are returned.


#------------------------
# Part A - Get the current working directory
#------------------------

# Define a variable input_dir with the path to the directory whether the images are saved. 
    
    # import getcwd and listdir from os
--- os --- getcwd, listdir   

    # get the current working directory with getcwd
input_dir = ---


#------------------------
# Part B - Generate a list of image filenames
#------------------------

# (i) Make a list varialbe containing the names of all the files in the directory, usign the function listdir(). (Suggested variable name'filelist'). listdir() is a function of the package os.
filelist = listdir(---)

# (ii) From the above list, collect the file names only for files that are tifs and allocate to a new list variable with name 'tiflist'. Here it is useful to use a for loop to loop over all the names in the filelist and using an if statement and slicing (indexing) to check if the current string contains the ending '.tif'.
for filename --- filelist---
    if filename[---] == '.tif';
        tiflist.---(filename)
        
# (iii) Now, check that you have the right files in tiflist. You can either print the number of files in the list, or print all the names in the list.
print "Found", ---(tiflist), "tif files in target directory\n"




#------------------------
# Part C - Loop over the tiflist, run the pipeline and collect the results
#------------------------

# (i) Initialise two dicts, all_results and all_segmentations, where you will collect the output for running the pipeline on all images.
all_results = []
all_segmentations = []

#   Write a for loop that goes through every file in the tiflist

#   Run the function pipeline for each file and allocate the output to new variables; remember that this pipeline returns two arguments, so you need two output variables. Recall that the returned arguments are order-specific.
#   Add the output to the variables all_results and all_segmentations, usign append.

        
# (ii) Write a for loop that goes through every file in the tiflist:
    # For each filein tiflist
--- filename --- tiflist:       
  
         # Run the function pipeline and allocate the output to new variables; remember that this pipeline returns two arguments, so you need two output variables. Recall that the returned arguments are order-specific.
    seg,results = ---(filename)
        
         # Add the output to the variables all_results and all_segmentations, usign .append()
    all_results.---(results)
    all_segmentations.---(seg)
    

# (iii) Check your understanding:
    # Try to remember what are the entries in the variables all_results and all_segmentations. Are they arrays? lists? dicts? also, what variable type are all_results and all_segmentations?
            
#------------------------
# Part E - Print a short summary  
#------------------------

# Find out how many cells in total were detected, from all the images: 

    # Initialise a counter num_cells to 0
num_cells = 0    

        # Use a for loop that goes through all_results; for each entry (image segmented)
for resultDict in all_results: 
    
            # identify how many entries there are in the entry with key "cell_id" and add this value to the counter
    num_cells = num_cells + len(resultDict["cell_id"])       
    
    # print a statement that reports the final count of  cells detected, for all images segmented
print "Detected", num_cells, "cells in total"


#------------------------
# Part F - Quick visualisation of results
#------------------------


# (i) Plot a scatter plot for all data and save image:

    # Loop through all_results and scatter plot the cell_size vs the red_membrane_mean. Remember to use a for loop and the function enumerate.
for image_id,resultDict in ---(all_results):
    plt.scatter(resultDict[---],---["red_mem_mean"])
        
    # Save the image to a png file using plt.savefig. 
plt.savefig(---all_cells_scatter.png---, bbox_inches='tight')  


#------------------------
# Part G - Save all the segmentations as a "3D" tif
#------------------------

# (i) Convert all_segmentations to a numpy array
all_segmentations = np.array(---)

# (ii) Check the dimensions of this array
print 'Dimensions of all_segmentations'--- all_segmentations.---()

# (iii) Save the image to a tif file using imsave from tifffile
    
    # Import the tifffile module
from tifffile --- imsave
    
    # Save the image using imsave
---("BATCH_segmentations.tif",all_segmentations,bigtiff=True)


#------------------------
# Part H - Save the data as a txt file 
#------------------------

# Open an empty file object (using with open), name it and specify the file format to .txt and the mode to write.
--- ---("BATCH_results.txt","w") as txt_out:   
    
    # Write the headers of the data (which are they key names of the dict containing the results for each input image), separated with tabs ('\t'). (You need .write() and .join())
    txt_out.write(''.---(key+'\t' for key in results.keys()) + '\n') 
    
    # Loop through each filename in tiflist (i.e. for each analyzed image...), (using a for loop and enumerate of tiflist)
    for image_id,filename in enumerate(tiflist)--- 
        
        # ...write the filename  
         txt_out.write(filename + "\n")
                  
        # ...extract the corresponding results from all_results
        resultDict = all_results[image_id]  
        
        # ...iterate over all the cells (using a for loop and enumerate of resultsDict["cell_id"])
        for index,value in ---(resultDict["cell_id"]):
            
            # ...write cell data, iterating over all the cells (using a for loop and enumerate of resultsDict["cell_id"])
            txt_out.---(''.join(str(resultDict[key][index])+'\t' for key in resultDict.keys()) + '\n')  