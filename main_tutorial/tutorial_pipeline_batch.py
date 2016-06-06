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
    # Array manipulation package numpy as np
    # Plotting package matplotlib.pyplot as plt
    # Image processing package scipy.ndimage as ndi


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


#%%   
#------------------------------------------------------------------------------
# SECTION 2 - EXECUTION SCRIPT

# Now that the pipeline function is defined, we can run it for each image file in a directory and collect the results as they are returned.


#------------------------
# Part A - Get the current working directory
#------------------------

# Define a variable 'input_dir' with the path to the current working directory, where the images should be saved. In principle, you can also specify any other path where you store your images.
    
    # (i) Import the function 'getcwd' from the module 'os'

    # (ii) Get the name of the current working directory with 'getcwd'


#------------------------
# Part B - Generate a list of image filenames
#------------------------

# (i) Make a list variable containing the names of all the files in the directory, using the function 'listdir' from the module 'os'. (Suggested variable name: 'filelist')

# (ii) From the above list, collect the filenames of only those files that are tifs and allocate them to a new list variable named 'tiflist'. Here, it is useful to use a for-loop to loop over all the names in 'filelist' and to use an if-statement combined with slicing (indexing) to check if the current string ends with the characters '.tif'.

# (iii) Double-check that you have the right files in tiflist. You can either print the number of files in the list, or print all the names in the list.


#------------------------
# Part C - Loop over the tiflist, run the pipeline for each filename and collect the results
#------------------------


# (i) Initialise two empty lists, 'all_results' and 'all_segmentations', where you will collect the quantifications and the segmented images, respectively, for each file.

# (ii) Write a for-loop that goes through every file in the tiflist. Within this for loop, you should:
         
         # Run the pipeline function and allocate the output to new variables; remember that this pipeline returns two arguments, so you need two output variables.

         # Add the output to the variables 'all_results' and 'all_segmentations', respectively. You can use the '.append' method to add them to the lists.

# (iii) Check your understanding:
    # Try to think about the complete data structure of 'all_results' and 'all_segmentations'. What type of variable are they? What type of variable to they contain? What data is contained within these variables? You can try printing things to fully understand the data structure.

# (iv) [OPTIONAL] Exception handling
    # It would be a good idea to make sure that not everything fails (i.e. the program stops and you loose all data) if there is an error in just one file. To avoid this, you can include a "try-except block" in your loop. To learn about handling exceptions (errors) in this way, visit http://www.tutorialspoint.com/python/python_exceptions.htm and https://wiki.python.org/moin/HandlingExceptions. Also, remember to include a warning message when the pipeline fails and to print the name of the file that caused the error, making a diagnosis possible. To do this properly, you should use the function 'warn' from the module 'warnings'. Finally, you may want to count how many times the pipeline runs correctly and print that number at the end, informing the user how many out of the total number of images were successfully segmented.


#------------------------
# Part D - Print a short summary  
#------------------------

# Find out how many cells in total were detected, from all the images: 

    # (i) Initialise a counter 'num_cells' to 0
    
    # (ii) Use a for loop that goes through 'all_results';
    
            # For each entry, identify how many cells were segmented in the image (e.g. by getting the length of the "cell_id" entry in the result dict). Add this length to the counter.
    
    # (iii) Print a statement that reports the final count of cells detected, for all images segmented.


#------------------------
# Part E - Quick visualisation of results
#------------------------

# (i) Plot a scatter plot for all data and save the image:

    # Loop through all_results and scatter plot 'cell_size' vs 'the red_membrane_mean'. Remember to use a for-loop and the function 'enumerate'.
    
    # Save the image to a png file using 'plt.savefig'. 


# (ii) [OPTIONAL] You may want to give cells from different images different colors:
    
    # Use the module 'cm' (for colormaps) from 'plt' and choose a colormap, e.g. 'jet'. 
    
    # Create the colormap with the number of colors required for the different images (in this example just 2). You can use 'range' or 'np.linspace' to ensure that you will always have the correct number of colors required, irrespective of the number of images you run the pipeline on. This colormap needs to be defined before making the plots.
    
    # When generating the scatter plot, use the parameter 'color' to use a different color from your colormap for each image you iterate through. Using 'enumerate' for the iterations makes this easier. For more info on 'color' see the docs of 'plt.scatter'.
    

#------------------------
# Part F - Save all the segmentations as a "3D" tif
#------------------------

# (i) Convert 'all_segmentations' to a 3D numpy array (instead of a list of 2D arrays)

# (ii) Save the result to a tif file using the 'imsave' function from the 'tifffile' module

# (iii) Have a look at the file in Fiji/ImageJ. The quality of segmentation across multiple images (that you did not use to optimize the pipeline) tells you how robust your pipeline is.


#------------------------
# Part G - Save the quantification data as a txt file 
#------------------------

# Saving your data as tab- or comma-separated text files allows you to import it into other programs (excel, Prism, R, ...) for further analysis and visualization.

# (i) Open an empty file object using "with open" (as explained at the end of the pipeline tutorial). Specify the file format to '.txt' and the mode to write ('w').

# (ii) The headers of the data are the key names of the dict containing the result for each input image (i.e. 'cell_id', 'green_mean', etc.). Write them on the first line of the file, separated by tabs ('\t'). You need the methods 'string.join' and 'file.write'.

# (iii) Loop through each filename in 'tiflist' (using a for-loop and enumerate of 'tiflist'). For each filename...

    # ...write the filename itself. 
         
    # ...extract the corresponding results from 'all_results'.
    
    # ...iterate over all the cells (using a for-loop and 'enumerate' of 'resultsDict["cell_id"]') and...
    
        # ...write the data of the cell, separated by '\t'. 
    
        

#%%   
#------------------------------------------------------------------------------
# SECTION 4 -  RATIOMETRIC NORMALIZATION TO CONTROL CHANNEL

# To correct for technical variability it is often useful to have an internal control, e.g. some fluorophore that we expect to be the same between all analyzed conditions, and use it to normalize other measurements.

# For example, we can assume that our green channel is just a generic membrane marker, whereas the red channel is a labelled protein of interest. Thus, using the red/green ratio instead of the raw values from the red channel may yield a clearer result when comparing intensity measurements of the red protein of interest between different images.

#------------------------
# Part A - Create the ratio
#------------------------

# (i) Calculate the ratio of red membrane mean intensity to green membrane mean intensity for each cell in each image. Add the results to the 'result_dict' of each image with a new key, for example 'red_green_mem_ratio'.


#------------------------
# Part B - Make a scatter plot, this time with the ratio
#------------------------

# (i) Recreate the scatterplot from Section 3, part E, but plotting the ratio over cell size rather than the red membrane mean intensity.

# (ii) Compare the two plots. Does the outcome match your expectations? Can you explain the newly 'generated' outliers?

# Note: Depending on the type of data and the question, normalizing with internal controls can be crucial to arrive at the correct conclusion. However, as you can see from the outliers here, a ratio is not always the ideal approach to normalization. When doing data analysis, you may want to spend some time thinking about how best to normalize your data. Testing different outcomes using 'synthetic' data (created using random number generators) can also help to confirm that your normalization (or your analysis in general) does not bias your results.


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# THIS IS THE END OF THE TUTORIAL.
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
