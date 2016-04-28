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

# 1. Make sure that all your python and image files are in the same directory and make that your working directory. 

# 2. Remember that you can develop this pipeline using a) a simple text editor and running it on the terminal, b) using Spyder or c) a Jupyter notebook. For all of them, you need to navigate to the directory where all your files are saved.
    # On the terminal, type cd dir_path, replacing dir_path for the path of the directory
    # On Spyder and Jupyter nootebook it can be done interactively.


#%%
#------------------------------------------------------------------------------
# SECTION 1 - IMPORT MODULES AND PACKAGES

# Modules in Python are simply Python files with the .py extension, which  implement a set of functions. The purpose of writting modules is to group  related code together, to make it easier to understand and use, sort of as a  'black box'.

# Packages are name spaces that contain multiple packages and modules themselves. You can think of them as 'directories'.

# To be able to use the functions in a particular package or module, you need  to 'import' that module or package. To do so, you need to use the import command. For example, to import the package numpy, which enables the manipulation of arrays, you would do:

# import the array manipulation package
import numpy as np    

# Then we can use a module from this package:

# Create an array
# a = np.array([1, 2, 3])  


# EXERCISE
# Some basic data/image manipulation tools will be used throughout this image  processing and segmentation script, so it makes sense to import them  at the  start. Let's get you started:

from __future__ import division      # Python 2.7 legacy (This particular module imports the division function of python3)

# Now, using the import command as above, import:
    # The plotting module matplotlib.pyplot as plt
    # The image processing package scypy.ndimage as ndi


#%%
#------------------------------------------------------------------------------
# SECTION 2 - IMPORT AND SLICE DATA

# Image processing is the manipulation of images, using certain operations that are mathematical in nature. For the purposes of image processing, it is often useful to convert images  into ``arrays'' - objects that follow an orderly arrangement, in rows and  columns, and for which many mathematical operations are well defined. The entries of the arrays correspond to pixel values of the images.


# EXERCISE
# We will now proceed to importing the image data, verifying we get what we expect and specifying the data we will work with:


# (i) Specify the filename: 

    # create a string variable with the name of the file to be imported. (Suggested name for variable 'filename')


# (ii) Import tif files:

    # import the image file manipulation module io from the package skimage, as io
    
    # import the multi-color tif file with the imread function of io. (Suggested name for variable 'img')


# (iii) Check that everything is in order:

    # Check img is a variable of type ndarray - use the command type
    
    # Print the array shape - use the command shape - and check you understand the output (recall that the image has 2 color channels and is 930 by 780 pixels)
    
    # Check the data type, using the command dtype, it should be uint8
    
    # Visualise one of the channels - use the commands plt.imshow and  plt.show. Check the documentation or plt.imshow and note the parameters that can be specified, such as the color map (cmap) and interpolation.


# (iv) Allocate the green channel to a new variable:

    # For segmentation, we only work on one channel, so we need allocate the green channel to a new variable, by slicing. (Suggested name for variable 'green'). To check whether the green channel is the fist or the second, you can use fiji or ask.
    # Recall the image has three dimensions, two (rows and columns) defining the size of the image in terms of pixels, and one defining the number of channels. To slice the array, you need to index each dimension to specify what you want from it.
    # For example, array x below has only one dimension.
    # x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # To obtain all the entries except the last, we slice the array as follows
    # x[:-1]
    # indicating we want everything up to the entry before the last.


#%%
#------------------------------------------------------------------------------
# SECTION 3 - PREPROCESSING AND SEGMENTATION OF CELLS:
#            (I) SMOOTHING AND (II) ADAPTIVE THRESHOLDING
#
# Part I - It's standard to smoothen images to reduce technical noise - this  improves all subsequent image processing steps. 
# Part II - Adaptive thresholding allows the masking of foreground objects even if the background intensity varies across the image. In this case, the forground objects we want are the cell membrane.


# ------
# Part I
# ------
# EXERCISE 
# Gaussian smoothing:

    # The Gaussian blur is a type of image-blurring filter that uses a Gaussian function (which also expresses the normal distribution in statistics) for calculating the transformation to apply to each pixel in the image. The smoothing factor for the Gaussian is simply the value of the standard deviation of the Gaussian distribution.

    # (i)
    # Decide on a smoothing factor for the Gaussian - in other words, create a variable with an integer value. (Suggested name for variable 'sigma')
    
    # (ii)
    # Perform the smoothing on green. To do so, use the Gaussian filter function ndi.filters.gaussian_filter (from the image processing package ndi, which was imported at the start of the tutorial).  Check out the documentation of scipy, for how to implement this  function. Allocate the output to a new variable. (Suggested name for  variable 'green_smooth')
    
    # (iv)
    # Visualise the result using plt.imshow and plt.show; compare with the original image visualised int he step above. Does the output make sense? Is this what you expected? 


# -------
# Part II
# -------

# BRIEF INTRO TO MORPHOLOGICAL PROCESSING
# 
# Images may contain numerous imperfections. Morphological image processing is a process that pursues the goal of removing these imperfections, by using a collection of operations that account for the form, shape or structure (morphology) of features in the image. 
#
# Morphological techniques probe an image with a small shape or template called a structuring element (se). An se is a binary image, i.e. an array of pixels, with each entry having a value of zero or one. The array dimensions specify the size of the se, e.g. a 3x3 array below, and the pattern of ones and zeros specifies the shape of the se, e.g. a square or cross, as below;
# 1 1 1         0 1 0
# 1 1 1         1 1 1
# 1 1 1         0 1 0
# The ``origin'' of the se is one of its pixels, usually the (middle) entry; entry (2,2) in the two examples above.
#
# Morphological processing consists of 
# (1) positioning the se at all possible locations in the image, 
# (2) comparing it with the corresponding image pixels, 
# (3) calculating a value (e.g. the average, median, etc) taking into account only the pixels of the image that have a corresponding 1 in the se, and 
# (4) allocating that value to the pixel in the image that corresponds to the  origin of the se.
#
# This procedure generates another array with the same dimensions as the input image, but, possibly, with different values for each of its entries. 
# 
# For example, allocating the average of all the pixel values in the se is called applying a mean filter. Another example is, with a binary input image, to test whether the element "fits" within the neighbourhood; that is, if for each of the pixels of the se that are set to 1, the corresponding image pixels are also 1. This process is called doing an erotion. A similar  concept is dilation, where you would test if the se "hits".



# EXERCISE

# Create an adaptive background (adaptive because the we have a grayscale and not binary image and the background intensties vary accross the image) and segment the cell membranes. 

# Two steps are involved:

# Step 1. create the background by allocating to each pixel the average values of its neighbourhood. In practice, this involves running a mean filter with a specific structuring element, accross the image.

# Step 2. since the green marker identifies the cell membrane, in the image obtained above, the pixels at the cell membranes should have the highest values; you want segment these by creating a binary image that has 1 at the membrane pixels and 0 elsewhere.


# Step 1

# (i) Create a disk-shaped structuring element and asign it to a new variable: 

    # Use the command below, replacing i for a specific integer to change the size of the se. Make sure that you understand how this commad is working, by printing parts of it and visualising it - try to answer the following questions: Is this se really disc-shaped? how can I check? why is this shape appropiate? what size should be appropiate and how to decide that? 

# struct = np.mgrid[:i,:i][0] - np.floor(i/2)**2 + (np.mgrid[:i,:i][1] - np.floor(i/2))**2) <= np.floor(i/2)**2


# (ii) Create the background: 

    # Run a mean filter (outlined above) over the image, using the disc structure, and assign the output to a new variable (suggested name for variable 'bg'). Use the function ndi.filters.generic_filter. Think about why a mean filter is used, could you use a different function, e.g. the maximum, the median, as combination of functions? What would be the results?

    # How ndi.filters.generic_filter works: The generic_filter function iterates a structuring element (e.g. a 3x3 square) over an image and passes all the values within that element to a predefine function (e.g. mean, or defined by you, e.g. edge_finder in section 6). The return of the function is allocated to the pixel in the image that corresponds to the origin of the se. See the documentation to find out about the arguments for the function.
    
    
# (iii) Visualise the result using plt.imshow. Compare with the images plotted above. Does it make sense?
    
    
# Step 2  

# (iv) Threshold using created background, to obtain the cells membrane:

    # The background that you have created is a local averaging of pixel values of the original image. Consequently, some of the pixel values in the original image will be higher than the highest pixel value of the background image. These correspond to the cell membrane. 

    # Using a relational operator allocate pixels in the original image that are greater than their corresponding in the background image, to 1, and 
    # everything else to zero. Remember to allocate the output to a new variable. (Suggested name for variable 'green_thresh')
    
    
# (v) Visualise and understand the output. What do you obeserve? Is this "good enough"? 


# (vi) Clean by morphological hole filling:

    # From the step above you should see the cells' boundaries. However, some of them seem to have gaps. To ammend this, we need to 'fill holes'. The scipy function ndi.binary_fill_holes is helpful here. (Since you are cleaning the image from the last step, we recommend using the same variable name). (FYI, this is  another instance of a morphological processing, where you  do a dilation followed by an erotion.)
    
# (vii) Visulaise the final result.
    
    
#%%
#------------------------------------------------------------------------------
# SECTION 4 - (SIDE NOTE: WE COULD BE DONE NOW)

# If the data is clean and/or we just want a quick look, we could simply label all connected pixels now and consider the result our segmentation.

# EXERCISE

# (i) Label connected components:

     # Collections of consecutive pixels that have the same value (0 or 1) are considered as connected regions and, if the segmentatin has been done correctly, they correspond to specific objects of the image, for example, different cells. So we label them with integer values using the command ndi.label. (Suggested name for variable 'green_components'). 
    
# (ii) Visualise the output.

# OBSERVATION
# To also partition the membranes of the cells, to generally improve the segmentatation (e.g. split cells that end up connected here) and to handle more complicated morphologies or to deal with lower quality data, this approach is not sufficient.



#%%
#------------------------------------------------------------------------------
# SECTION 5 - SEGMENTATION OF CELLS CONTINUED: (I) SEEDING BY DISTANCE TRANSFORM AND (II) EXPANSION BY WATERSHED

# More advanced segmentation is usually a combination of seeding and expansion. 

# -------
# Part I
# -------

# In seeding, we want to find a few pixels for each cell that we can assign to  said cell with great certainty. These 'seeds' are then expanded (with watershed in part II) to partition regions of the image where cell affiliation is less clear-cut.


# EXERCISE

# (i) Distance transform on thresholded membranes:
    # The result of a distance transform is an image that look similar to the input image, except that the intensities of points inside foreground regions are changed to represent the distance to the closest boundary from each point. The advantage of distance transform for seeding is that it is robust to local "holes" in the membranes.

    # To apply a distance transform, use the command ndi.distance_transform_edt. (Suggested name for variable 'green_dt'). 
    
    
# (ii) Visualise the output.


# (iii) Retrieve and label the local maxima:

    # Import the module peak_local_max from skimage.feature

    # Local maximum detection: use peak_local_max to detect the local max with optional arguments min_distance=10 and indices=False (remember to use the documentation to understand the function and its arguments). (Suggested name for variable 'green_max')

    # Label the connected components. (Use the same variable name, 'green_max')


# (iv) Visualise the output by showing maxima as masked overlay - ie overlay the maxima (the seeds) on the original input image (or smoothed version):

    # Note that the variable that stores the seeds, or maxima, itself is an array, where every entry, whether a maxima or not, has a value of intensity. If you want to overlay the seeds, you only want to plot the pixels of this array that make up the seeds and not any of the other entries of the array. To achieve that, you can use a `mask the array', using the command ma.array of numpy.

    # What do you observe? (The seeding should probably be improved.)


# (v)
# Dilating (maximum filter) of distance transform improves the result of seeding:

    # Dilation adds pixels to the boundaries of objects in an image. 
    
    # Use ndi.filters.maximum_filter. Read the documentation to understand  how/if/where the se can be defined with this function and the filter applied. (Since you are improving the distance transform, use the same variable name as above, 'green_dt')


# (vi) Visualise the output.


# (vii) Retrieve and label the local maxima again:

    # Local maximum detection (as above, use peak_local_max to detect the local max). (Suggested name for variable 'green_max')

    # Label the connected components (use the same variable name, 'green_max')


# (viii) Visualise the output by showing maxima as masked overlay - ie overlay the maxima (the seeds) on the original input image (or smoothed version), as above.



# -------
# Part II
# -------

# Watershedding is a relatively simple but powerful algorithm for expanding seeds. The image intensity is considered as a topographical map (with high  intensities being "mountains" and low intensities "valleys") and water is  poured into the valleys from each of the seeds. The water first labels the lowest intensity pixels around the seeds, then continues to fill up. The cell boundaries (discontinuities or significant transitions in an image) are where the "waterfronts" between different seeds touch.

# EXERCISE

# Watershed:
    # Get the watershed function from skimage.morphology. 
    
    # Run it and allocate the result to a new variable. (Suggested name for variable 'green_ws')
    
    # Show result as transparent overlay, as we did before. Note that you can use the optional argument alpha of plt.imshow to specify opacity.
    

# OBSERVATION
# Note that the previously connected cells are now mostly separated and the 
# membranes are partitioned to their respective cells. However, we now see a 
# few cases of oversegmentation. This is a typical example of the trade-offs 
# one has to face in any computational classification task.     



#%%
#------------------------------------------------------------------------------
# SECTION 6 - SEGMENTATION OF CELL EDGES

# Finding cell edges is useful for many purposes. In our example, edge  intensities are a measure of membrane intensities, which may be a desired  readout. The length of the edge (relative to cell size) is also an informative feature about the cell shape. Finally, showing colored edges is a nice way of visualize segmentations.

# Here we need to use the function generic_filter again, but this time, instead of using the mean function, we define our own function. If you can not remember how generic_filter works, go back to the explanation of section 3.


# EXERCISE

# (i) Define a function called edge_finder to pass to ndi.filters.generic_filter:
    # This function will 'detect an edge'. More specifically, this function should 
    # - take in as input a se representing a neighbourhood around a pixel, 
    # - check if all pixels are the same or not (function all() is helpful here)
    # - return 0 or 1, respectively.
    
    # Why this operation makes sense: For a se of the "correct size", if all the corresponding pixels in the input image have the same value (0 or 1), then the origin of the se is not at an edge, otherwise it is at an edge.

    # Remember that to define a new function the syntax is 
    # def function_name(input arguments): 
    #   """function documentation string"""
    #   function procedure
    #   return [expression]
    #
    
# (ii) Iterate the edge_finder function defined above over the segmentation green_edges using the ndi.filters.generic_filter function, with optional argument size=3. Allocate the output to a new variable (with suggested name 'green_edges')

# (iii) Label the detected edges:
     # ...based on the underlying cells and allocate the output to a new variable (with suggested name 'green_edges_labeled'). (Hint: use array multiplication - green_edges is a binary array (consists of 1s and 0s) and green_ws  is an array with connected components, corresponding to segmented cells, labeled.

# (iv) Visualise as an overlay:
     # Labeled edges only (not cells) over the original (or smoothed) image


#%%
#------------------------------------------------------------------------------
# SECTION 7 - POSTPROCESSING: REMOVING CELLS AT THE IMAGE BORDER

# Segmentation is never perfect and it often makes sense to remove artefacts
# afterwards. For example, one could filter out cells that are too
# big, have a strange shape, or strange intensity values. Similarly, supervised 
# machine learning can be used to identify cells of interest based on a 
# combination of various features. 

# Another example of cells that should be removed are those that are cut at the image boundary. This is what we will do now:


# EXERCISE

# (i) Create boundary mask:
    # Remember that a mask or filter, is a binary array, with a specific size and distribution of 1's and 0's for a specific purpose. Recall the  examples of dilation and and erotion above. We also touched upon  blurring using a Gaussian and adaptive thresholding using a mean filter. Recall also that the size of the mask needs to be thougth about.
    # So now, you need to create a mask for the image boundary pixels. If you want to make the perifery of the image stand out, how would you allocate 1's and 0's and what would be the size of the mask? Function np.ones_like and array indexing are helpful. (Suggested name for variable 'boundary_mask')


# (ii) 'Delete' the cells in the border:

    # - Iterate over all cells in the segmentation, using a for loop and function np.unique. (Hint: remember that green_ws has disconnected components (corresponding to different cells) labeled with different integer values)
    # - Identify the cells that have pixels at the image boundary, using the boundary_mask, array arithmetic, relational operations and an if statement
    # - "Delete" the cells at the boundary, by changing their intensity 
    # values to 0
    # - (Optionally, relabel the cells not at the boundary, to keep the numbering consistant.)


# (iii) Visualise result:
    # Show the result as transparent overlay of the watershed over the blurred original. Rememeber that here you want to only hshow the watershed cells that are not touching the boundary, so you need np.ma.array


#%%
#------------------------------------------------------------------------------
# SECTION 8 - MEASUREMENTS: SINGLE-CELL AND MEMBRANE READOUTS

# Now that the cells and membranes in the image are nicely segmented, we can quantify various readouts for every cell individually. Readouts can be based on the intensity in different channels in the original image, on the size and shape of the cells themselves, etc.

# For the purposes of this tutorial, we want:
# - the cell id
# - mean intensity of each cell, for each channel
# - mean intensity for the membrane of each cell, for each channel
# - the cell size, in terms of the number of pixels that make up the cell
# - the cell outline length, in terms of the number of pixels that make up the cell boundaries

# We will use a dictionary to collect all the information



# EXERCISE

# (i) Initialize a dictionary variable for the results listed above:
     # Choose descriptive keys names
     # (Suggested variable name 'results')


# (ii) Record the measurements:
    
    # Iterate over segmented cells using a for loop and the function np.unique

        # Mask the pixels of the current cell and allocate to variable name cell_mask

        # Get the current cell's values listed above and store in dict (using .append for the appropiate key)
    
    
    # Hint 1: the segmented cells with membrane is in variable grend_ws. Run the following commands and understand the output:
#cell_mask = green_ws==1
#plt.imshow(cell_mask)
#plt.show()
    
    # Hint 2: the segmented cells' membranes is in variable green_edges. Run the following commands and understand the output:
#edge_mask = np.logical_and(cell_mask,green_edges)
#plt.imshow(edge_mask)
#plt.show()
    
    # Hint 3: you need to use the original raw data for quantification!
    # Hint 4: Recall you can index an array with another (logical) array. Run the following commands and understand the output:
#cell1 = img[0,:,:][cell_mask]



#%%    
#------------------------------------------------------------------------------
# SECTION 9 - ANALYSIS AND EXPORTING OUTPUT OF PROGRAM

# Now that you have collected the readouts to a dictionary, you can analyse them, by printing out the results, mapping some of the readouts back to the image,  e.g. the cells' size or membranes, doing statistical analysis of the data, such as making boxplots, scatter plots and fitting a line to the data and analysing the line.


# EXERCISE

# (i) Print out the results you want to see. If you want to see all the results, you could iterate over the keys of the dictionary.


# (ii) Make box plots of the cell and membrane intensities, for both channels, in the same image. Use the function plt.boxplot and a for loop for the x argument; for the label argument use the corresponding key names. You also need plt.show.


# (iii) Make a scatter plot to show if there is a dependancy of the membrane intensity (green channel) on the cell size and add the linear fit to scatter plot, to see the correlation:

    # Import the module stats from scipy       

    # Use the function stats.linregress to do a linear fit of cell size vs membrane intensity. Make sure to read the documentation to understand the result of this function. (Suggested variable name 'linfit').
        
    # Make scatter plot of the cell_size vs the green_membrane_mean. Use the function plt.scatter. Remember to label the axes, with plt.xlabel and plt.ylabel. 

    # Define the equation of the line that fits the data:
         # Recall that the equation of a straight line in 2D is y = m * x + c, where m is the slope and c is the intersection of the y-axis. Where do you get m and c from? (Hint: see the output of stats.linregress). 
         # You need to create an function that takes in a value (x) and returns the fitted value (y). You can create a normally defined function or an anonymous function, using lambda. The example below shows the difference between a normal function and an anonymous function:
         #
         # # normal function definition
         # 	def f(x): 
         #      "Returns the square value"
         #      return x**2
         #
         # anonymous function
         # g = lambda x: x**2       # returns the square value

    # Get the fitted values:
        # Get graph limits for the x axis. Use plt.gca and get_xlim. Allocate to a variable with name x_lims

        # Use the function map to pass the x_lims to the equation of a line function (anonymous function above), thus obtaining the corresponding fitted values. Check the documentation of map. Allocate the output to a variable with name fit_vals.

    # Plot the line         
        # Plot the line using plt.plot (you need to plot the x_vals vs the fit_vals). You can specify the esthetics of the line, e.g. line width, color, etc.
        
    # Finally, use plt.show() to show the image.

    
    
# (iv) Print out results from the linear regression (again, check the documentation of stats.linregress):

    # Define a list with the (strings) names of the output of the linear regression (from stats.linregress)

    # Iterate over all values using a for loop and function enumerate to enumerate the entries of linfit

        # print the name and the corresponding value on the same line

    # (After the for loop), also calculate and print the r-squared value
    

# (v) Map the cell size and cell membrane back onto the image:

    # Scale the cells sizes to 8bit (since we want image values). 
    # (Hint: if the largest cell size should correspond to the value 255 in uint8, then the other cell sizes correspond to cell_size*255/largest_cell_size) (suggested variable name 'sizes_8bit')
    
    # Initialise a new image (to zeros, with the same dimensions as e.g. green_ws and data type uint8). (Suggested variable name 'size_map')
    
    # Iterrate over the segmented cells of green_ws.(You need a for loop, and the functions enumerate and np.unique. )
        # Assign to each pixel of size_map the cell size (in uint8) of the cell it corresponds to
    
    # Visualise the result as a colored overlay over the smoothed image green_smooth

        
# (vi)
# Do an analysis of statistical tests and the rest of the measurements. Think  about what the results actually mean and whether any inconsistancies can be traced back to a not sufficiently good segmentation, e.g. we  have filtered for some artifacts, such as incomplete cells (that touch the boundary), but not for others; can you think of some?.
        


#%%    
#------------------------------------------------------------------------------
# SECTION 10 - MEASUREMENTS: WRITING OUTPUT

# There are several ways of presenting the output of a program. Data can be 
# saved to files in a human-readable format (e.g. text files (e.g. to import 
# into Excel), images, etc), or written to language-specific files for future 
# use (i.e. instead of having to run the whole program again). Here you will 
# learn some of these possibilities.

# ECERCISE

# (i) Write an image to a tif (could be opened e.g. in Fiji):

    # Get the file handling function imsave from the package tifffile
    
    # Save desired array to tif using imsave



# (ii) Write a figure to a png or pdf:

    # Recreate the scatter plot from above (without the regression line). (Remember to label axes)
    
    # Save to png, using plt.savefig
    
    # Save to pdf, using plt.savefig



# ***For the next two exercises, refer to the python documentation for input and output.***

# (iii) Write a python file that can be reloaded in other Python programs:

    # Import the module json
    
    # Open an empty file object (using with open), name it with extension .json and specify the the mode to 'w' (write). 
    
    # Use the function json.dump to write the resutls.    
    
# (Note: This could be loaded again in this way:
# with open(filename+'_resultsDict.json', 'r') as fp:
#    results = json.load(fp)


# (iv) Write a text file of the numerical data gathered (could be opened e.g. in Excel):

    # Open an empty file object (using with open), name it and specify the file format to .txt and the mode to write.
    
    # Write the headers of the data, separated with tabs ('\t'). (You need .write() and .join())
    
    # Iterate over cells saved in your results variable. (Use a for loop and the function enumerate)
    
        # for each key, write the data to the text file, separated with tabs ('\t'). (you need .write(), .join() and a for loop to iterate over all the keys)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# THIS IS THE END OF THE TUTORIAL.
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------




