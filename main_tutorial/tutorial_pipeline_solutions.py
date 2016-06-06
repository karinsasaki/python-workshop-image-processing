# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 00:12:38 2015

@author:    Jonas Hartmann @ Gilmour Group @ EMBL Heidelberg
            Edited by Karin Sasaki @ CBM @ EMBL Heidelberg
            
@descript:  This is an example pipeline for the segmentation of 2D confocal
            fluorescence microscopy images of a membrane marker in confluent
            epithel-like cells. It exemplifies many fundamental concepts of 
            automated image processing and segmentation. 
            
            The pipeline is optimized to run with the provided example images,
            which are dual-color spinning-disc confocal micrographs (40x) of
            two membrane-localized proteins during zebrafish early embryonic
            development (~10hpf).
            
            'tutorial_pipeline_batch.py' shows how the same pipeline could be
            adapted to run automatically on multiple images in a directory.
            
@requires:  Python 2.7
            NumPy 1.9, SciPy 0.15
            scikit-image 0.11.2, tifffile 0.3.1
"""

#%%
#------------------------------------------------------------------------------
# SECTION 0 - SET UP

# 1. Remember that you can develop this pipeline using 
#    a) a simple text editor and running it on the terminal
#    b) the Spyder IDE or 
#    c) a Jupyter notebook.

# 2. Make sure that all your python and image files are in the same directory, then make that directory your working directory.
#    - On the terminal, type "cd dir_path", replacing dir_path for the path of the directory
#    - In Spyder and Jupyter nootebook it can be done interactively.

# 3. Python is continuously under development to get better and better. In some rare cases, these new improvements need to be specifically imported to be active. One such case is the division operation in Python 2.7, which has some undesired behavior for the division of integer numbers. We can easily fix this by importing the new and improved division function from Python 3. It makes sense to do this at the start of all Python 2.7 scripts.
from __future__ import division

# 4. This script consists of explanations and exercises that guide you to complete the pipeline. It is designed to give you a guided experience of what "real programming" is like. This is one of the reasons why the pre-tutorial is provided as a Jupyter Notebook, but this main tutorial is not; we, and our colleagues, mostly develop programs using a text editor and the terminal.  In that same spirit, if you already have access to the solutions, we recommend that you try to solve the tutorial alone, without looking at them. 

# 5. If you are not feeling comfortable with the exercises, there is a partially-solved version that you can also follow. 


#%%
#------------------------------------------------------------------------------
# SECTION 1 - IMPORT MODULES AND PACKAGES

from __future__ import division    # Python 2.7 legacy
import numpy as np                 # Array manipulation package
import matplotlib.pyplot as plt    # Plotting package
import scipy.ndimage as ndi        # Image processing package


#%%
#------------------------------------------------------------------------------
# SECTION 2 - IMPORT AND PREPARE DATA

# Image processing essentially means carrying out mathematical operations on images. For this purpose, it is useful to represent image data in orderly data structures called "arrays" for which many mathematical operations are well defined. Arrays are grids with rows and columns that are filled with numbers; in the case of image data, those numbers correspond to the pixel values of the image. Arrays can have any number of dimensions (or "axes"). For example, a 2D array could represent the x and y axis of a normal image, a 3D array could contain a z-stack (xyz), a 4D array could also have multiple channels for each image (xyzc) and a 5D array could have time on top of that (xyzct).

# EXERCISE
# We will now proceed to import the image data, verifying we get what we expect and specifying the data we will work with. Before you start, it makes sense to have a quick look at the data in Fiji/imagej so you know what you are working with.

# Specify filename
filename = "example_cells_1.tif"

# Import tif files
import skimage.io as io               # Image file manipulation module
img = io.imread(filename)             # Importing multi-color tif file

# Check that everything is in order
print type(img)     # Check that img is a variable of type ndarray
print img.dtype     # Check data type is 8uint
print "Loaded array has shape", img.shape  # Printing array shape; 2 colors, 930 by 780 pixels

# Show image
plt.imshow(img[0,:,:],interpolation='none',cmap='gray')   # Showing one of the channels (notice "interpolation='none'"!)
plt.show()

# Slicing: We only work on one channel for segmentation
green = img[0,:,:]


#%%
#------------------------------------------------------------------------------
# SECTION 3 - PREPROCESSING AND SIMPLE CELL SEGMENTATION:
#            (I) SMOOTHING AND (II) ADAPTIVE THRESHOLDING

# -------
# Part I
# -------

# Gaussian smoothing
sigma = 3                                                # Smoothing factor for Gaussian
green_smooth = ndi.filters.gaussian_filter(green,sigma)  # Perform smoothing

# visualise 
plt.imshow(green_smooth,interpolation='none',cmap='gray')
plt.show()


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

# Show the result
plt.imshow(green_mem,interpolation='none',cmap='gray')
plt.show()



#%%
#------------------------------------------------------------------------------
# SECTION 4 - CONNECTED COMPONENTS LABELING (OR: "WE COULD BE DONE NOW")

# If the data is clean and we just want a very quick cell or membrane segmentation, we could be done now. All we would still need to do is to label the individual cells - in other words, to give each separate "connected component" an individual number.

# Labeling connected components
green_components = ndi.label(green_mem)[0] 
plt.imshow(green_components,interpolation='none', cmap='gray')    
plt.show() 

# The result you get here should look not to bad but will likely still have some problems. For example, some cells will be connected because there were small gaps between them in the membrane. Also, the membranes themselves are not partitioned to the individual cells, so we cannot make measurements of membrane intensities for each cell. These problems can be resolved by means of a "seeding-expansion" strategy, which we will implement below.



#%%
#------------------------------------------------------------------------------
# SECTION 5 - IMPROVED CELL SEGMENTATION BY SEEDING AND EXPANSION: 
#             (I) SEEDING BY DISTANCE TRANSFORM
#             (II) EXPANSION BY WATERSHED
#
# Part I - Seeding refers to the identification of 'seeds', a few pixels that can assigned to each particular cell with great certainty. If available, a channel showing the cell nuclei is often used for seeding. However, using the membrane segmentation we have developed above, we can also generate relatively reliable seeds without the need to image nuclei.
# Part II - The generated seeds are expanded into regions of the image where the cell assignment is less clear-cut than in the seed region itself. The goal is to expand each seed exactly up to the borders of the corresponding cell, resulting in a full segmentation. The watershed technique is the most common algorithm for expansion.

# -------
# Part I
# -------

# Distance transform on thresholded membranes
# Advantage of distance transform for seeding: It is quite robust to local 
# "holes" in the membranes.
green_dt= ndi.distance_transform_edt(green_mem)
plt.imshow(green_dt,interpolation='none')
plt.show()

# Dilating (maximum filter) of distance transform improves results
green_dt = ndi.filters.maximum_filter(green_dt,size=10) 
plt.imshow(green_dt,interpolation='none')
plt.show()

# Retrieve and label the local maxima
from skimage.feature import peak_local_max
green_max = peak_local_max(green_dt,indices=False,min_distance=10)  # Local maximum detection
green_max = ndi.label(green_max)[0]                                 # Labeling

# Show maxima as masked overlay
plt.imshow(green_smooth,cmap='gray',interpolation='none')
plt.imshow(np.ma.array(green_max,mask=green_max==0),interpolation='none') 
plt.show()


# -------
# Part II
# -------

# Watershedding is a relatively simple but powerful algorithm for expanding seeds. The image intensity is considered as a topographical map (with high  intensities being "mountains" and low intensities "valleys") and water is poured into the valleys, starting from each of the seeds. The water first labels the lowest intensity pixels around the seeds, then continues to fill up. The cell boundaries (the 'mountains') are where the "waterfronts" between different seeds ultimately touch and stop expanding.

# Get the watershed function and run it
from skimage.morphology import watershed
green_ws = watershed(green_smooth,green_max)

# Show result as transparent overlay
# Note: For a better visualization, see "FINDING CELL EDGES" below!
plt.imshow(green_smooth,cmap='gray',interpolation='none')
plt.imshow(green_ws,interpolation='none',alpha=0.7) 
plt.show()

# OBSERVATION
# Note that the previously connected cells are now mostly separated and the membranes are partitioned to their respective cells. Depending on the quality of the seeding, however, there may now be some cases of oversegmentation (a single cell split into multiple segmentation objects). This is a typical example of the trade-off between specificity and sensitivity one always has to face in computational classification tasks. As an advanced task, you can try to think of ways to fuse the wrongly oversegmented cells back together.   


#%%
#------------------------------------------------------------------------------
# SECTION 6 - IDENTIFICATION OF CELL EDGES

# Now that we have a full cell segmentation, we can retrieve the cell edges, that is the pixels bordering neighboring cells. This is useful for many purposes; in our case, for example, edge intensities are a good measure of membrane intensity, which may be a desired readout. The length of the edge (relative to cell size) is also an informative feature about the cell shape. Finally, showing colored edges is a nice way of visualizing cell segmentations.

# There are many ways of identifying edge pixels in a fully labeled segmentation. It can be done using erosion or dilation, for example, or it can be done in an extremely fast and fully vectorized way (for this, see "Vectorization" in the optional advanced content). Here, we use a slow but intuitive method that also serves to showcase the 'generic_filter' function in ndimage.

# 'ndi.filters.generic_filter' is a powerful way of quickly iterating any function over numpy arrays (including functions that use a structuring element). 'generic_filter' iterates a structure element over all the values in an array and passes the corresponding values to a user-defined function. The result returned by this function is then allocated to the pixel in the image that corresponds to the origin of the se. Check the documentation to find out more about the arguments for 'generic_filter'.

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

# Show them as masked overlay
plt.imshow(green_smooth,cmap='gray',interpolation='none')
plt.imshow(np.ma.array(green_edges_labeled,mask=green_edges_labeled==0),interpolation='none') 
plt.show()




#%%
#------------------------------------------------------------------------------
# SECTION 7 - POSTPROCESSING: REMOVING CELLS AT THE IMAGE BORDER

# Segmentation is never perfect and it often makes sense to remove artefacts afterwards. For example, one could filter out objects that are too small, have a very strange shape, or very strange intensity values. Note that this is equivalent to the removal of outliers in data analysis and should only be done for good reason and with caution.

# As an example of postprocessing, we will now filter out a particular group of problematic cells: those that are cut off at the image border.


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
plt.imshow(green_smooth,cmap='gray',interpolation='none')
plt.imshow(np.ma.array(green_ws,mask=green_ws==0),interpolation='none',alpha=0.7) 
plt.show()


#%%
#------------------------------------------------------------------------------
# SECTION 8 - MEASUREMENTS: SINGLE-CELL AND MEMBRANE READOUTS

# Now that the cells and membranes in the image are segmented, we can quantify various readouts for every cell individually. Readouts can be based on the intensity in different channels in the original image or on the size and shape of the cells themselves.

# To exemplify how different properties of cells can be measured, we will quantify the following:
    # Cell ID (so all other measurements can be traced back to the cell that was measured)
    # Mean intensity of each cell, for each channel
    # Mean intensity at the membrane of each cell, for each channel
    # The cell size, in terms of the number of pixels that make up the cell
    # The cell outline length, in terms of the number of pixels that make up the cell boundary

# We will use a dictionary to collect all the information in an orderly fashion.


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


#%%    
#------------------------------------------------------------------------------
# SECTION 9 - SIMPLE ANALYSIS AND VISUALIZATION

# Now that you have collected the readouts to a dictionary you can analyse them in any way you wish. This section shows how to do basic plotting and analysis of the results, including mapping the data back onto the image (as a 'heatmap') and producing boxplots, scatterplots and a linear fit. A more in-depth example of how to couple image analysis into advanced data analysis can be found in 'data_analysis' in the 'optional_advanced_material' directory.


# (i) Print out the results you want to see
for key in results.keys():
    print "\n" + key
    print results[key]


# (ii) Make box plots of the cell and membrane intensities, for both channels.
plt.boxplot([results[key] for key in results.keys()][2:-1],labels=results.keys()[2:-1])
plt.show()


# (iii) 
# Make a scatter plot to show if there is a dependancy of the membrane intensity (either channel) on the cell size (for example) and add the linear fit to scatter plot, to see the correlation.

    # Import the module stats from scipy       
from scipy import stats                  

    # Linear fit of cell size vs membrane intensity
linfit = stats.linregress(results["cell_size"],results["red_membrane_mean"]) 
                
    # Make scatter plot. 
plt.scatter(results["cell_size"],results["red_membrane_mean"])
plt.xlabel("cell size")
plt.ylabel("red_membrane_mean")

    # Define the equation of the line that fits the data, using an anonymous function
fit = lambda x: linfit[0] * x + linfit[1]

    # Get the fitted values (for graph limits)
ax = plt.gca()        
x_lims = ax.get_xlim()
fit_vals = map(fit,x_lims)

    # Plot the line 
plt.gca().set_autoscale_on(False)         # Prevent the figure from rescaling when line is added, by using plt.gca().set_autoscale_on(False)
plt.plot(x_lims,fit_vals,'r-',lw=2)
plt.show()

    

# (iv) Print out results from stats analysis
linnames = ["slope","intercept","r-value","p-value","stderr"]            # Names of esults of stats.linregress
print "\nLinear fit of cell size to red membrane intensity"              # Header
for index,value in enumerate(linfit):                                    # For each value...
    print "  " + linnames[index] + "\t\t" + str(value)                   # ...print the result
print "  r-squared\t\t" + str(linfit[2]**2)                              # Also print R-squared


# (v) Map the cell size and cell membrane back onto the image.
sizes_8bit = results["cell_size"] / max(results["cell_size"]) * 255  # Map to 8bit
size_map = np.zeros_like(green_ws,dtype=np.uint8)                    # Initialize image
for index,cell_id in enumerate(np.unique(green_ws)[1:]):             # Iterate over cells
    size_map[green_ws==cell_id] = sizes_8bit[index]                  # Assign corresponding cell size to cell pixels


plt.imshow(green_smooth,cmap='gray',interpolation='none')            # Set grayscale background image
plt.imshow(np.ma.array(size_map,mask=size_map==0),interpolation='none',alpha=0.7)  # Colored overlay
plt.show()


# (vi)
# Note that this seems to return a highly significant p-value but a very low
# correlation coefficient (r-value). We also would not expect this correlation
# to be present in our data. This should prompt several considerations:
#   1) What does this p-value actually mean? See help(stats.linregress)
#   2) Since we have not filtered properly for artefacts (e.g. "cells" of very
#      small size), they might bias this particular fit.
#   3) We're now working with a lot of datapoints. This can skew statistical
#      analyses! To some extent, we can correct for this by multiple testing
#      correction and by comparison with randomized datasets. Additionaly, a 
#      closer look at Bayesian statistics is highly recommended for people 
#      working with large datasets.


#%%
#------------------------------------------------------------------------------
# SECTION 10 - MEASUREMENTS: WRITING OUTPUT

# There are several ways of presenting the output of a program. Data can be saved to files in a human-readable format (e.g. text files (e.g. to import into Excel), images, etc), or written to language-specific files for future use (i.e. instead of having to run the whole program again). Here you will learn some of this possibilities.


# (i)
# Write an image to a tif (could be opened e.g. in Fiji)

# Get file handling function
from tifffile import imsave                                            

# Save array to tif
imsave(filename+"_labeledEdges.tif",green_edges_labeled,bigtiff=True)  



# (ii)
# Write a figure to a png or pdf

# Recreate scatter plot from above
plt.scatter(results["cell_size"],results["red_membrane_mean"])  
plt.xlabel("cell size")                                    
plt.ylabel("red membrane mean")

# Save to png (rasterized)
plt.savefig(filename+'_scatter.png', bbox_inches='tight')  

# Save to pdf (vectorized)
plt.savefig(filename+'_scatter.pdf', bbox_inches='tight')  



# (iii)
# Write a python file that can be reloaded in other Python programs
import json
with open('k_resultsDict.json', 'w') as fp:
    json.dump(results, fp)
    
# This could be loaded again in this way:
#with open(filename+'_resultsDict.json', 'r') as fp:
#    results = json.load(fp)


# (iv)
# Write a text file of the numerical data gathered  (could be opened e.g. in Excel)
with open(filename+"_output.txt","w") as txt_out:                                                # Open an empty file object (with context manager)
    txt_out.write(''.join(key+'\t' for key in results.keys()) + '\n')                            # Write the headers
    for index,value in enumerate(results["cell_id"]):                                            # Iterate over cells
        txt_out.write(''.join(str(results[key][index])+'\t' for key in results.keys()) + '\n')   # Write cell data
            

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# THIS IS THE END OF THE TUTORIAL.
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


