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
# SECTION 1 - IMPORT MODULES AND PACKAGES

from __future__ import division    # Python 2.7 legacy
import numpy as np                 # Array manipulation package
import matplotlib.pyplot as plt    # Plotting package
import scipy.ndimage as ndi        # Image processing package


#%%
#------------------------------------------------------------------------------
# SECTION 2 - IMPORT AND SLICE DATA

# Specify filename
filename = "example_cells_1.tif"

# Import tif files
import skimage.io as io               # Image file manipulation module
img = io.imread(filename)             # Importing multi-color tif file

# Check that everything is in order
print type(img)     # Check that img is a variable of type ndarray
print img.dtype     # Check data type is 8uint
print "Loaded array has shape", img.shape                 # Printing array shape; 2 colors, 930 by 780 pixels

# Show image
plt.imshow(img[0,:,:],interpolation='none',cmap='gray')   # Showing one of the channels (notice "interpolation='none'"!)
plt.show()

# Slicing: We only work on one channel for segmentation
green = img[0,:,:]


#%%
#------------------------------------------------------------------------------
# SECTION 3 - PREPROCESSING AND SEGMENTATION OF CELLS:
#            (I) SMOOTHING AND (II) ADAPTIVE THRESHOLDING

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

# visualise 
plt.imshow(green_smooth,interpolation='none',cmap='gray')
plt.show()

# -------
# Part II
# -------

# Create an adaptive background
#struct = ndi.iterate_structure(ndi.generate_binary_structure(2,1),24)  # Create a diamond-shaped structural element
struct = ((np.mgrid[:31,:31][0] - 15)**2 + (np.mgrid[:31,:31][1] - 15)**2) <= 15**2  # Create a disk-shaped structural element
bg = ndi.filters.generic_filter(green_smooth,np.mean,footprint=struct) # Run a mean filter over the image using the disc

# Threshold using created background
green_thresh = green_smooth >= bg

# Clean by morphological hole filling
green_thresh = ndi.binary_fill_holes(np.logical_not(green_thresh))

# Show the result
plt.imshow(green_thresh,interpolation='none',cmap='gray')
plt.show()



#%%
#------------------------------------------------------------------------------
# SECTION 4 - (SIDE NOTE: WE COULD BE DONE NOW)
# If the data is very clean and/or we just want a quick look, we could simply
# label all connected pixels now and consider the result our segmentation.

# Labeling connected components
green_components = ndi.label(green_thresh)[0] 
plt.imshow(green_components,interpolation='none', cmap='gray')    
plt.show() 

# However, to also partition the membranes to the cells, to generally improve  
# the segmentatation (e.g. split cells that end up connected here) and to 
# handle more complicated morphologies or to deal with lower quality data, 
# this approach is not sufficient.



#%%
#------------------------------------------------------------------------------
# SECTION 5 - SEGMENTATION OF CELLS CONTINUED: (I) SEEDING BY DISTANCE TRANSFORM AND (II) EXPANSION BY WATERSHED

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
plt.imshow(green_smooth,cmap='gray',interpolation='none')
plt.imshow(green_ws,interpolation='none',alpha=0.7) 
plt.show()

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

# Segmentation is never perfect and it often makes sense to remove artefacts
# afterwards. For example, one could filter out cells that are too
# big, have a strange shape, or strange intensity values. Similarly, supervised 
# machine learning can be used to identify cells of interest based on a 
# combination of various features. 

# Another example of cells that should be removed are those that are cut at the image boundary. This is what we will do now:

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

# Now that the cells in the image are nicely segmented, we can quantify various
# readouts for every cell individually. Readouts can be based on the intensity
# in the original image, on intensities in other channels or on the size and
# shape of the cells themselves.

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
# SECTION 9 - ANALYSIS AND EXPORTING OUTPUT OF PROGRAM

# MOTIVATION
# Now that you have collected the readouts to a dict, you can analyse them, by 
# printing out the results, mapping some of the readouts back to the image, 
# e.g. the cells' size or membranes, doing statistical analysis of the data, 
# such as making boxplots, scatter plots and fitting a line to the data and 
# analysing the line.


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



# (vii)
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


