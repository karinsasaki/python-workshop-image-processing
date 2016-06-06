# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 00:12:38 2015

@author:    Created by Jonas Hartmann @ Gilmour Group @ EMBL Heidelberg
            Edited by Karin Sasaki @ CBM @ EMBL Heidelberg
            
@descript:  This is the batch version of 'tutorial_pipeline.py', which is an  
            example pipeline for the segmentation of 2D confocal fluorescence 
            microscopy images of a membrane marker in confluent epithel-like 
            cells. This batch version serves to illustrate how such a pipeline
            can be run automatically on multiple images that are saved in the
            current directory.
            
            The pipeline is optimized to run with the provided example images,
            which are dual-color spinning-disc confocal micrographs (40x) of
            two membrane-localized proteins during zebrafish early embryonic
            development (~10hpf).

@requires:  Python 2.7
            NumPy 1.9, SciPy 0.15
            scikit-image 0.11.2, tifffile 0.3.1
"""


#%%
#------------------------------------------------------------------------------
#  SECTION 0 - SET UP

# 1. (Re)check that the segmentation pipeline works!
# You should already have the final pipeline that segments and quantifies one of the example images. Make sure it is working by running it in one go from start to finish. Check that you do not get errors and that output is what you expect.

# 2. Check that you have the right data!
# We provide two images ('example_cells_1.tif' and 'example_cells_2.tif') to test the batch version of the pipeline. Make sure you have them both ready in the working directory.

# 3. Deal with Python 2.7 legacy
from __future__ import division

# 4. EXERCISE: Import modules required by the pipeline
import numpy as np                 # Array manipulation package
import matplotlib.pyplot as plt    # Plotting package
import scipy.ndimage as ndi        # Image processing package


#%%
#------------------------------------------------------------------------------
# SECTION 1 - PACKAGE PIPELINE INTO A FUNCTION

# The goal of this script is to repeatedly run the segmentation algorithm you
# programmed in tutorial_pipeline.py. The easiest way of packaging code to run
# it multiple times is to make it into a function.

# EXERCISE
# Define a function that...
#     ...takes one argument as input: a filename as a string
#     ...returns two outputs: the final segmentation and the quantified data
#     ...reports that it is finished with the current file just before returning the result.

# To do this, you need to copy the pipeline you developed, up to section 8, and paste it inside the function. Since the pipeline should run without any supervision by the user, remove (or comment out) any instances where an image would be shown. You can also exclude section 4. Make sure everything is set up such that the function can be called and the entire pipeline will run with the filename that is passed to the function.

# Recall that to define a new function the syntax is 
# def function_name(input arguments): 
#   """function documentation string"""
#   function procedure
#   return [expression]

def pipeline(filename):
    
    # Report that the pipeline is being executed
    print "  Starting pipeline for", filename
    
    # Import tif file
    import skimage.io as io               # Image file manipulation module
    img = io.imread(filename)             # Importing multi-color tif file
    
    # Slicing: We only work on one channel for segmentation
    green = img[0,:,:]
    
    
    #------------------------------------------------------------------------------
    # PREPROCESSING AND SIMPLE CELL SEGMENTATION:
    # (I) SMOOTHING AND (II) ADAPTIVE THRESHOLDING
    
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
    from skimage.filters import rank            # Import module containing mean filter function
    bg = rank.mean(green_smooth, selem=struct)  # Run a mean filter over the image using the disc
    
    # Threshold using created background
    green_mem = green_smooth >= bg
    
    # Clean by morphological hole filling
    green_mem = ndi.binary_fill_holes(np.logical_not(green_mem))
    
    
    #------------------------------------------------------------------------------
    # IMPROVED CELL SEGMENTATION BY SEEDING AND EXPANSION: 
    # (I) SEEDING BY DISTANCE TRANSFORM
    # (II) EXPANSION BY WATERSHED
    
    # -------
    # Part I
    # -------
    
    # Distance transform on thresholded membranes
    # Advantage of distance transform for seeding: It is quite robust to local 
    # "holes" in the membranes.
    green_dt= ndi.distance_transform_edt(green_mem)
    
    # Dilating (maximum filter) of distance transform improves results
    green_dt = ndi.filters.maximum_filter(green_dt,size=10) 
    
    # Retrieve and label the local maxima
    from skimage.feature import peak_local_max
    green_max = peak_local_max(green_dt,indices=False,min_distance=10)  # Local maximum detection
    green_max = ndi.label(green_max)[0]                                 # Labeling
    
    
    # -------
    # Part II
    # -------
    
    # Get the watershed function and run it
    from skimage.morphology import watershed
    green_ws = watershed(green_smooth,green_max)
    
    
    #------------------------------------------------------------------------------
    # IDENTIFICATION OF CELL EDGES
    
    # Define the edge detection function
    def edge_finder(footprint_values):
        if (footprint_values == footprint_values[0]).all():
            return 0
        else:
            return 1
        
    # Iterate the edge finder over the segmentation
    green_edges = ndi.filters.generic_filter(green_ws,edge_finder,size=3)

    
    #------------------------------------------------------------------------------
    # POSTPROCESSING: REMOVING CELLS AT THE IMAGE BORDER    
    
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
    

    #------------------------------------------------------------------------------
    # MEASUREMENTS: SINGLE-CELL AND MEMBRANE READOUTS    
    
    # Initialize a dict for results of choice
    results = {"cell_id":[], "green_mean":[], "red_mean":[],"green_membrane_mean":[], 
               "red_membrane_mean":[],"cell_size":[],"cell_outline":[]}
    
    # Iterate over segmented cells
    for cell_id in np.unique(green_ws)[1:]:
        
        # Mask the pixels of the current cell
        cell_mask = green_ws==cell_id  
        edge_mask = np.logical_and(cell_mask,green_edges)
        
        # Get the current cell's values
        # Note that the original raw data is used for quantification!
        results["cell_id"].append(cell_id)
        results["green_mean"].append(np.mean(img[0,:,:][cell_mask]))
        results["red_mean"].append(np.mean(img[1,:,:][cell_mask]))    
        results["green_membrane_mean"].append(np.mean(img[0,:,:][edge_mask]))    
        results["red_membrane_mean"].append(np.mean(img[1,:,:][edge_mask]))    
        results["cell_size"].append(np.sum(cell_mask))
        results["cell_outline"].append(np.sum(edge_mask))


    #------------------------------------------------------------------------------
    # REPORT PROGRESS AND RETURN RESULTS

    print "  Completed pipeline for", filename    

    return green_ws, results 


#%%   
#------------------------------------------------------------------------------
# SECTION 2 - EXECUTION SCRIPT

# Now that the pipeline function is defined, we can run it for each image file in a directory and collect the results as they are returned.


#------------------------
# Part A - Get the current working directory
#------------------------

# Define a variable 'input_dir' with the path to the current working directory, where the images should be saved. In principle, you can also specify any other path where you store your images.
    
# Import 'getcwd' and retrieve working directory
from os import getcwd 
input_dir = getcwd()   


#------------------------
# Part B - Generate a list of image filenames
#------------------------

# Make a list of files in the directory
from os import listdir
filelist = listdir(input_dir)

# Collect the file names only for files that are tifs 

# Note: This is an elegant solution using "list comprehension".
tiflist = [filename for filename in filelist if filename[-4:]=='.tif'] 

# This is the more classical solution. It does exactly the same:
tiflist = []
for filename in filelist:
    if filename[-4:]=='.tif':
        tiflist.append(filename)

# Check that you have the right files in tiflist. 
print "Found", len(tiflist), "tif files in target directory"


#------------------------
# Part C - Loop over the tiflist, run the pipeline for each filename and collect the results
#------------------------

# Initialise 'all_results' and 'all_segmentations' to store output.
all_results = []
all_segmentations = []

# Initialise a counter to count how many times the pipeline is run succesfully
success_counter = 0

# Run the actual batch processing
print "Running batch processing"
for filename in tiflist: # For every file in tiflist...
    
    # Exception handling so the program can move on if one image fails for some reason.
    try:  
    
        # Run the pipeline
        seg,results = pipeline(filename)  
        
        # Add the results to our collection lists
        all_results.append(results)       
        all_segmentations.append(seg)
        
        # Update the success counter
        success_counter += 1
        
    # What to do if something goes wrong.
    except Exception:
        
        # Warn the user, then carry on with the next file
        from warnings import warn
        warn("There was an exception in " + filename + "!!!")


#------------------------
# Part D - Print a short summary  
#------------------------

# How many images were successfully analyzed?
print "Successfully analyzed", success_counter, "of", len(tiflist), "images"

# How many cells were segmented in total.
num_cells = 0 
for resultDict in all_results: # For each image...
    num_cells = num_cells + len(resultDict["cell_id"]) # ...add the number of cells detected.

# Print a statement that reports the final count of cells detected, for all images segmented.    
print "Detected", num_cells, "cells in total"


#------------------------
# Part E - Quick visualisation of results
#------------------------

# Scatter plot of red membrane mean intensity over cell size. 

# Prepare colormap to color cells from each image differently
colors = plt.cm.jet(np.linspace(0,1,len(all_results)))

# For each analyzed image...
for image_id,resultDict in enumerate(all_results):
    
    # ...add the datapoints to the plot.
    plt.scatter(resultDict["cell_size"],resultDict["red_membrane_mean"],color=colors[image_id])

# Label axes
plt.xlabel("cell size")
plt.ylabel("red membrane mean")

# Show or save result
plt.savefig('BATCH_all_cells_scatter.png', bbox_inches='tight')
plt.show()


#------------------------
# Part F - Save all the segmentations as a "3D" tif
#------------------------

# Convert 'all_segmentations' to a 3D numpy array
all_segmentations = np.array(all_segmentations)

# Save the result to a tif file using the 'imsave' function from the 'tifffile' module
from tifffile import imsave
imsave("BATCH_segmentations.tif",all_segmentations,bigtiff=True)


#------------------------
# Part G - Save the quantification data as a txt file 
#------------------------

# Open an empty file using the context manager ('with')
with open("BATCH_results.txt","w") as txt_out:         

    # Write the headers (note the use of a "list comprehension"-style in-line for-loop)
    txt_out.write(''.join(key+'\t' for key in results.keys()) + '\n')
    
    # For each analyzed image...
    for image_id,filename in enumerate(tiflist):                       
        
        # ...write the filename
        txt_out.write(filename + "\n")
        
        # ...extract the corresponding results                   
        resultDict = all_results[image_id]
        
        # ...iterate over cells...
        for index,value in enumerate(resultDict["cell_id"]):           
            
            # ...and write cell data (note the use of a "list comprehension"-style in-line for-loop)
            txt_out.write(''.join(str(resultDict[key][index])+'\t' for key in resultDict.keys()) + '\n')   


#%%   
#------------------------------------------------------------------------------
# SECTION 4 -  RATIOMETRIC NORMALIZATION TO CONTROL CHANNEL

# To correct for technical variability it is often useful to have an internal control, e.g. some fluorophore that we expect to be the same between all analyzed conditions, and use it to normalize other measurements.

# For example, we can assume that our green channel is just a generic membrane marker, whereas the red channel is a labelled protein of interest. Thus, using the red/green ratio instead of the raw values from the red channel may yield a clearer result when comparing intensity measurements of the red protein of interest between different images.

#------------------------
# Part A - Create the ratio
#------------------------

# For each image...
for image_id,resultDict in enumerate(all_results):
    
    # Calculate red/green ratio and save it under a new key in result_dict. Done for each cell using list comprehension.
    all_results[image_id]["red_green_mem_ratio"] = [resultDict["red_membrane_mean"][i] / resultDict["green_membrane_mean"][i] for i in range(len(resultDict["cell_id"]))]


#------------------------
# Part B - Make a scatter plot, this time with the ratio
#------------------------

# Scatterplot of red/green ratio over cell size.

# Prepare colormap to color the cells of each image differently
colors = plt.cm.jet(np.linspace(0,1,len(all_results)))

# For each image...
for image_id,resultDict in enumerate(all_results):
    
    # ...add the data points to the scatter.
    plt.scatter(resultDict["cell_size"],resultDict["red_green_mem_ratio"],color=colors[image_id])
    
# Label axes
plt.xlabel("cell size")
plt.ylabel("red membrane mean")

# Show or save result
plt.savefig('BATCH_all_cells_ratio_scatter.png', bbox_inches='tight')  
plt.show()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# THIS IS THE END OF THE TUTORIAL.
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
