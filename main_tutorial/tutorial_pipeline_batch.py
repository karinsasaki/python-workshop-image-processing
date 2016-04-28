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

# Since you know the pipeline requires certain modules and packages, you 
# should import those here, at the beginning of the program.


from __future__ import division    # Python 2.7 legacy
# Array manipulation package numpy as np
# Plotting package matplotlib.pyplot as plt
# Image processing package scipy.ndimage as ndi


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


#%%   
#------------------------------------------------------------------------------
# Step 2 -  BATCH VERSION: EXECUTION SCRIPT

# Now that the pipeline function is defined, we can run it for each image file in a directory and collect the results as they are returned.


#------------------------
# Part A - Get the current working directory
#------------------------

# Define a variable input_dir with the path to the directory whether the images are saved. 
    
    # import getcwd and listdir from os

    # get the current working directory with getcwd


#------------------------
# Part B - Generate a list of image filenames
#------------------------

# (i) Make a list varialbe containing the names of all the files in the directory, usign the function listdir(). (Suggested variable name'filelist'). listdir() is a function of the package os.

# (ii) From the above list, collect the file names only for files that are tifs and allocate to a new list variable with name 'tiflist'. Here it is useful to use a for loop to loop over all the names in the filelist and using an if statement and slicing (indexing) to check if the current string contains the ending '.tif'.

# (iii) Now, check that you have the right files in tiflist. You can either print the number of files in the list, or print all the names in the list.


#------------------------
# Part C - Loop over the tiflist, run the pipeline and collect the results
#------------------------

# (i) Initialise two dicts, all_results and all_segmentations, where you will collect the output for running the pipeline on all images.

# (ii) Write a for loop that goes through every file in the tiflist:
    # For each filein tiflist
         
         # Run the function pipeline and allocate the output to new variables; remember that this pipeline returns two arguments, so you need two output variables. Recall that the returned arguments are order-specific.

         # Add the output to the variables all_results and all_segmentations, usign .append()

# (iii) Check your understanding:
    # Try to remember what are the entries in the variables all_results and all_segmentations. Are they arrays? lists? dicts? also, what variable type are all_results and all_segmentations?


#------------------------
# Part D - **Optional** improvement for running the pipeline
#------------------------

# It would be a good idea to make sure that not everything fails (program stops 
# and you loose all data) if there is an error in just one file. To avoid this,
# you can include a try block with an except statement. To learn about handling
# exceptions, visit http://www.tutorialspoint.com/python/python_exceptions.htm 
# and https://wiki.python.org/moin/HandlingExceptions .
# Also, remember to include a warning message when the pipeline fails and 
# to print the name of that file. This is useful specially in cases when      
# you have more than just a handful of images to go through. For this you     
# need to import warnings from warn.

# Additionally, you might like to count the number of files for which the 
# pipeline was run successfully. For that purpose, you would need to initialise 
# (to zero) a counter, named, e.g. success_counter and everytime the pipeline 
# runs successfully, add 1 to the counter. 
# After the pipeline has run for all the images, report by printing a         
# statement showing for how many images (out of the total) the pipeline run 
# successfully.


#------------------------
# Part E - Print a short summary  
#------------------------

# Find out how many cells in total were detected, from all the images: 

    # Initialise a counter num_cells to 0
    
        # Use a for loop that goes through all_results; for each entry (image segmented)
    
            # identify how many entries there are in the entry with key "cell_id" and add this value to the counter
    
    # print a statement that reports the final count of  cells detected, for all images segmented


#------------------------
# Part F - Quick visualisation of results
#------------------------

# (i) Plot a scatter plot for all data and save image:

    # Loop through all_results and scatter plot the cell_size vs the red_membrane_mean. Remember to use a for loop and the function enumerate.
    
    # Save the image to a png file using plt.savefig. 

# (ii) Optionally, you can give cells from different images different colors:
    
    # Use the module cm (for colormaps) from plt and choose a colormap, e.g. jet. 
    
    # Create the color map with the number of colors required for the different images, (in this example, 2). Use np.linspace.
    
    # This colormap needs to be defined before making the plots.
    
    # When doing the scatter plot, use the parameter color (see the documentation for plt.scatter for more info).
    

#------------------------
# Part G - Save all the segmentations as a "3D" tif
#------------------------

# (i) Convert all_segmentations to a numpy array

# (ii) Check the dimensions of this array

# (iii) Save the image to a tif file using imsave from tifffile
    
    # Import the tifffile module
    
    # Save the image using imsave


#------------------------
# Part H - Save the data as a txt file 
#------------------------


# Open an empty file object (using with open), name it and specify the file format to .txt and the mode to write.

    # Write the headers of the data (which are they key names of the dict containing the results for each input image), separated with tabs ('\t'). (You need .write() and .join())

    # Loop through each filename in tiflist (i.e. for each analyzed image...), (using a for loop and enumerate of tiflist)

        # ...write the filename  
         
        # ...extract the corresponding results from all_results
    
        # ...iterate over all the cells (using a for loop and enumerate of resultsDict["cell_id"])
    
            # ...write cell data, iterating over all the cells (using a for loop and enumerate of resultsDict["cell_id"])
    
        

#%%   
#------------------------------------------------------------------------------
# Step 4 -  RATIOMETRIC NORMALIZATION TO CONTROL CHANNEL

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
plt.xlabel("cell size")
plt.ylabel("red_mem_mean")
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
