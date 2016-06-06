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
import --- as np                 # Array manipulation package
import --- as plt    # Plotting package
import --- as ndi        # Image processing package


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

--- pipeline(filename):
    
    # Report that the pipelineis being executed
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
    
    # (i) Import the function 'getcwd' from the module 'os'
--- os --- getcwd 

    # (ii) Get the name of the current working directory with 'getcwd'
input_dir = ----   



#------------------------
# Part B - Generate a list of image filenames
#------------------------

# (i) Make a list variable containing the names of all the files in the directory, using the function 'listdir' from the module 'os'. (Suggested variable name: 'filelist')
--- os --- listdir
filelist = listdir(----)


# (ii) From the above list, collect the filenames of only those files that are tifs and allocate them to a new list variable named 'tiflist'. Here, it is useful to use a for-loop to loop over all the names in 'filelist' and to use an if-statement combined with slicing (indexing) to check if the current string ends with the characters '.tif'.
tiflist = []
--- filename --- filelist:
    --- filename[-4:]=='.tif':
        tiflist.---(filename)

# (iii) Double-check that you have the right files in tiflist. You can either print the number of files in the list, or print all the names in the list.
print "Found", ---(tiflist), "tif files in target directory"


#------------------------
# Part C - Loop over the tiflist, run the pipeline for each filename and collect the results
#------------------------

# (i) Initialise two empty lists, 'all_results' and 'all_segmentations', where you will collect the quantifications and the segmented images, respectively, for each file.
all_results = []
all_segmentations = []

# (ii) Write a for-loop that goes through every file in the tiflist. Within this for loop, you should:
print "Running batch processing"
--- filename --- tiflist: # For every file in tiflist...
         
         # Run the pipeline function and allocate the output to new variables; remember that this pipeline returns two arguments, so you need two output variables.
    seg,results = ---(---)  

         # Add the output to the variables 'all_results' and 'all_segmentations', respectively. You can use the '.append' method to add them to the lists.
    all_results.---(results)       
    all_segmentations.---(seg)

# (iii) Check your understanding:
    # Try to think about the complete data structure of 'all_results' and 'all_segmentations'. What type of variable are they? What type of variable to they contain? What data is contained within these variables? You can try printing things to fully understand the data structure.

# (iv) [OPTIONAL] Exception handling
    # It would be a good idea to make sure that not everything fails (i.e. the program stops and you loose all data) if there is an error in just one file. To avoid this, you can include a "try-except block" in your loop. To learn about handling exceptions (errors) in this way, visit http://www.tutorialspoint.com/python/python_exceptions.htm and https://wiki.python.org/moin/HandlingExceptions. Also, remember to include a warning message when the pipeline fails and to print the name of the file that caused the error, making a diagnosis possible. To do this properly, you should use the function 'warn' from the module 'warnings'. Finally, you may want to count how many times the pipeline runs correctly and print that number at the end, informing the user how many out of the total number of images were successfully segmented.
# see below


# you can use the for loop below instead of the one above
#--- filename --- tiflist: # For every file in tiflist...
#    
#    # Exception handling so the program can move on if one image fails for some reason.
#    try---  
#    
#        # Run the pipeline
#        seg,results = ---(---)  
#        
#        # Add the results to our collection lists
#        all_results.---(results)       
#        all_segmentations.---(seg)
#        
#        # Update the success counter
#        success_counter += 1
#        
#    # What to do if something goes wrong.
#    except Exception:
#        
#        # Warn the user, then carry on with the next file
#        from warnings import warn
#        warn("There was an exception in " --- filename + "!!!")

    

#------------------------
# Part D - Print a short summary  
#------------------------

# Find out how many cells in total were detected, from all the images: 

    # (i) Initialise a counter 'num_cells' to 0
num_cells = 0 
    
    # (ii) Use a for loop that goes through 'all_results';
for --- in ---: # For each image...
    
            # For each entry, identify how many cells were segmented in the image (e.g. by getting the length of the "cell_id" entry in the result dict). Add this length to the counter.
    num_cells = num_cells + len(---["cell_id"]) # ...add the number of cells detected.
        
    # (iii) Print a statement that reports the final count of cells detected, for all images segmented.
print "Detected", ---, "cells in total"


#------------------------
# Part E - Quick visualisation of results
#------------------------

# (i) Plot a scatter plot for all data and save the image:

    # Loop through all_results and scatter plot 'cell_size' vs 'the red_membrane_mean'. Remember to use a for-loop and the function 'enumerate'.
for image_id,resultDict in ---(all_results):
    # ...add the datapoints to the plot.
    plt.---(resultDict["cell_size"],resultDict[---])

# Label axes
plt.x---("cell size")
plt.---label("red membrane mean")
            
    # Save the image to a png file using 'plt.savefig'. 
plt.---('BATCH_all_cells_scatter.png', bbox_inches='tight')
plt.show()


# (ii) [OPTIONAL] You may want to give cells from different images different colors:
    
    # Use the module 'cm' (for colormaps) from 'plt' and choose a colormap, e.g. 'jet'. 
    
    # Create the colormap with the number of colors required for the different images (in this example just 2). You can use 'range' or 'np.linspace' to ensure that you will always have the correct number of colors required, irrespective of the number of images you run the pipeline on. This colormap needs to be defined before making the plots.
    
    # When generating the scatter plot, use the parameter 'color' to use a different color from your colormap for each image you iterate through. Using 'enumerate' for the iterations makes this easier. For more info on 'color' see the docs of 'plt.scatter'.

# Note: Use either the version below or the one above
# Prepare colormap to color cells from each image differently
colors = plt.---.jet(np.linspace(0,---,len(all_results)))

# For each analyzed image...
for image_id,resultDict in ---(all_results):
    
    # ...add the datapoints to the plot.
    plt.---(resultDict["cell_size"],resultDict[---],color=---[image_id])

# Label axes
plt.x---("cell size")
plt.---label("red membrane mean")

# Show or save result
plt.---('BATCH_all_cells_scatter.png', bbox_inches='tight')
plt.show()


#------------------------
# Part F - Save all the segmentations as a "3D" tif
#------------------------

# (i) Convert 'all_segmentations' to a 3D numpy array (instead of a list of 2D arrays)
all_segmentations = np.array(---)

# (ii) Save the result to a tif file using the 'imsave' function from the 'tifffile' module
from tifffile import imsave
imsave("BATCH_segmentations.tif",---,bigtiff=True)

# (iii) Have a look at the file in Fiji/ImageJ. The quality of segmentation across multiple images (that you did not use to optimize the pipeline) tells you how robust your pipeline is.


#------------------------
# Part G - Save the quantification data as a txt file 
#------------------------

# Saving your data as tab- or comma-separated text files allows you to import it into other programs (excel, Prism, R, ...) for further analysis and visualization.

# (i) Open an empty file object using "with open" (as explained at the end of the pipeline tutorial). Specify the file format to '.txt' and the mode to write ('w').
--- ---("BATCH_results.txt","w") --- txt_out:         

# (ii) The headers of the data are the key names of the dict containing the result for each input image (i.e. 'cell_id', 'green_mean', etc.). Write them on the first line of the file, separated by tabs ('\t'). You need the methods 'string.join' and 'file.write'.
    txt_out.---(''.---(key+'\t' for key in results.keys()) + '\n')
    
# (iii) Loop through each filename in 'tiflist' (using a for-loop and enumerate of 'tiflist'). For each filename...
    for image_id,filename in ---(tiflist):                       
        
    # ...write the filename itself. 
        txt_out.---(--- + "\n")
        
    # ...extract the corresponding results from 'all_results'.
        resultDict = all_results[---]
        
    # ...iterate over all the cells (using a for-loop and 'enumerate' of 'resultsDict["cell_id"]') and...
        for index,value in ---(resultDict["cell_id"]):           
            
        # ...write the data of the cell, separated by '\t'. 
            txt_out.---(''.---(str(resultDict[key][index])+'\t' for key in resultDict.keys()) + '\n')   



#%%   
#------------------------------------------------------------------------------
# SECTION 4 -  RATIOMETRIC NORMALIZATION TO CONTROL CHANNEL

# To correct for technical variability it is often useful to have an internal control, e.g. some fluorophore that we expect to be the same between all analyzed conditions, and use it to normalize other measurements.

# For example, we can assume that our green channel is just a generic membrane marker, whereas the red channel is a labelled protein of interest. Thus, using the red/green ratio instead of the raw values from the red channel may yield a clearer result when comparing intensity measurements of the red protein of interest between different images.

#------------------------
# Part A - Create the ratio
#------------------------

# (i) Calculate the ratio of red membrane mean intensity to green membrane mean intensity for each cell in each image. Add the results to the 'result_dict' of each image with a new key, for example 'red_green_mem_ratio'.

# For each image...
for image_id,resultDict in ---(all_results):
    
    # Calculate red/green ratio and save it under a new key in result_dict. Done for each cell using list comprehension.
    all_results[image_id]["red_green_mem_ratio"] = [resultDict["---"][i] --- resultDict["---"][i] --- i --- range(len(resultDict["cell_id"]))]


#------------------------
# Part B - Make a scatter plot, this time with the ratio
#------------------------
# (i) Recreate the scatterplot from Section 3, part E, but plotting the ratio over cell size rather than the red membrane mean intensity.
# Prepare colormap to color the cells of each image differently
colors = plt.cm.jet(np.linspace(0,1,len(all_results)))

# For each image...
for ---,--- in ---(all_results):
    
    # ...add the data points to the scatter.
    plt.---(resultDict["---"],resultDict["---"],color=colors[image_id])
    
# Label axes
plt.xlabel("cell size")
plt.ylabel("red membrane mean")

# Show or save result
plt.---('BATCH_all_cells_ratio_scatter.png', bbox_inches='tight')  
plt.---()


# (ii) Compare the two plots. Does the outcome match your expectations? Can you explain the newly 'generated' outliers?

# Note: Depending on the type of data and the question, normalizing with internal controls can be crucial to arrive at the correct conclusion. However, as you can see from the outliers here, a ratio is not always the ideal approach to normalization. When doing data analysis, you may want to spend some time thinking about how best to normalize your data. Testing different outcomes using 'synthetic' data (created using random number generators) can also help to confirm that your normalization (or your analysis in general) does not bias your results.


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# THIS IS THE END OF THE TUTORIAL.
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
