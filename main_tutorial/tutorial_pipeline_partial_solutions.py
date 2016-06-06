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

# 5. If you feel these partial solutions are not challenging enough, you can follow the same tutorial with no solutions whatsoever.
    
#%%
#------------------------------------------------------------------------------
# SECTION 1 - IMPORT MODULES AND PACKAGES

# Let's import the package NumPy, which enables the manipulation of numerical arrays:
# (Note: if you are not familiar with arrays and NumPy, we strongly recommend that you follow the accompanying tutorials on these topics before carrying on with this one.)
import numpy as np    

# Recall that once imported, we can use functions/modules from the package, for example to create an array:
a = np.array([1, 2, 3])

# Note that the package is imported under a variable name (here "np"). You can choose this name freely yourself. For example, it would be just as valid (but not as convenient) to write:

#import numpy as lovelyArrayTool
#a = lovelyArrayTool.array([1,2,3])


# EXERCISE
# We will be using a number of data/image manipulation tools throughout this pipeline, so it makes sense to import them at the beginning.

from __future__ import division    # Python 2.7 legacy
import --- as np                 # Array manipulation package
import matplotlib.pyplot --- plt    # Plotting package
--- scipy.ndimage --- ndi        # Image processing package


#%%
#------------------------------------------------------------------------------
# SECTION 2 - IMPORT AND PREPARE DATA

# Image processing essentially means carrying out mathematical operations on images. For this purpose, it is useful to represent image data in orderly data structures called "arrays" for which many mathematical operations are well defined. Arrays are grids with rows and columns that are filled with numbers; in the case of image data, those numbers correspond to the pixel values of the image. Arrays can have any number of dimensions (or "axes"). For example, a 2D array could represent the x and y axis of a normal image, a 3D array could contain a z-stack (xyz), a 4D array could also have multiple channels for each image (xyzc) and a 5D array could have time on top of that (xyzct).


# EXERCISE
# We will now proceed to import the image data, verifying we get what we expect and specifying the data we will work with. Before you start, it makes sense to have a quick look at the data in Fiji/imagej so you know what you are working with.

# (i) Specify the filename: 

    # create a string variable with the name of the file to be imported. (Suggested name for variable: 'filename')
filename = ---


# (ii) Import tif files:

    # import the image file manipulation module 'io' from the package 'skimage' (Suggested name for module: 'io')
--- skimage.io --- io
    
    # import the provided multi-color tif file ('example_cells_1.tif') with the function 'imread' of 'io'. (Suggested name for variable: 'img')
img = io.---(filename)


# (iii) Check that everything is in order:

    # Check that 'img' is a variable of type 'ndarray' - use Python's built-in function 'type'
print ---(img)

    # Print the shape of the array using the numpy-function 'shape'. Make sure you understand the output; recall that the image has 2 color channels and is 930 by 780 pixels. 
 print img.---
    
    # Check the datatype of the individual numbers in the array. You can use the array attribute 'dtype' to do so.
--- "Loaded array has shape", img.shape
    
    # Visualise one of the channels; use pyplot's functions plt.imshow and plt.show. Check the documentation for plt.imshow and note the parameters that can be specified, such as the color map (cmap) and interpolation. Since you are working with scientific data, interpolation is usually undesired, so you should set it to 'none'. The most common cmap for grayscale images is naturally 'gray'.
    # Choose one from the three options below and delete (or comment out) the others
# $(a)$
plt.imshow(img,interpolation='none',cmap='gray')   # Showing one of the channels 
plt.show()

# $(b)$
plt.imshow(img[:,0,:],interpolation='none',cmap='gray')   # Showing one of the channels

# $(c)$
plt.imshow(img[0,:,:],interpolation='none',cmap='gray')   # Showing one of the channels
plt.show()


# (iv) Allocate the green channel to a separate new variable:

    # For segmentation, we only work with the green channel, so we need to allocate it to a new variable. The green channel in this image is the first channel (or channel 0 in python). We can allocated it to a new variable by slicing the 'img' array. (Suggested name for new variable: 'green'). 
    # Hint: Recall that the image has three dimensions, two (rows and columns) defining the size of the image in terms of pixels, and one defining the number of channels. To slice the array, you need to index each dimension to specify what you want from it.
    # For example, array A below has two dimensions.
    # A = np.array([[1,2,3],[4,5,6]])
    # To obtain all entries in the first row, we would slice like this:
    # B = A[0,:]
    # You can slice the 2D green channel out of the 3D 'img' array in a similar fashion. 
green = img[---,---,---]


#%%
#------------------------------------------------------------------------------
# SECTION 3 - PREPROCESSING AND SIMPLE CELL SEGMENTATION:
#            (I) SMOOTHING AND (II) ADAPTIVE THRESHOLDING
#
# Part I - It is very common to smoothen images to reduce technical noise; this improves all subsequent image processing steps. 
# Part II - Adaptive thresholding allows the masking of foreground objects even if the background intensity varies across the image. In this case, the foreground objects we want are the cell membrane.


# ------
# Part I
# ------

# Gaussian smoothing:

# The Gaussian blur is a type of image-blurring filter that uses a Gaussian function (which also expresses the normal distribution in statistics) for calculating the transformation to apply to each pixel in the image. The smoothing factor for the Gaussian is simply the value of the standard deviation of the Gaussian distribution (denoted as 'sigma'). 

# Choice of sigma: What value of sigma to use must be decided specifically for the type of image you are working on (it depends on the microscope, pixel size, etc...) and for the type of analysis you are performing. In general, the chosen sigma should be large enough to blur out noise but small enough so the structure of interest does not get blurred too much. A first guess can be made based on what the image looks like in Fiji (here, the noise in the image is relatively small, mostly just a couple of pixels). Then, the optimal number is found by trial and error.

# EXERCISE

    # (i) Create a variable for the smoothing factor sigma, which should be an integer value. After implementing the Gaussian smoothing function below, you can modify this variable to find the ideal value of sigma. (Suggested name for variable: 'sigma')
--- = 3                                           
    
    # (ii) Perform the smoothing on 'green'. To do so, use the Gaussian filter function 'ndi.filters.gaussian_filter' (from the image processing package ndimage, which was imported at the start of the tutorial). Check out the documentation of scipy to see how to implement this function. Allocate the output to a new variable. (Suggested name for variable: 'green_smooth')
green_smooth = ndi.filters.gaussian_filter(---,sigma)   
    
    # (iii) Visualise the result using plt.imshow and plt.show; compare with the original image visualised in the step above. Does the output make sense? Is this what you expected? Can you optimize sigma such that the image looks smooth without blurring the membranes too much?
plt.figure()    
plt.imshow(---,---,cmap='gray')
plt.show()



# -------
# Part II
# -------

# Adaptive thresholding:

# To distinguish foreground objects (here: membranes) from the image background, we can threshold the image - in other words, we create an array where all foreground pixels are set to 1 and all background pixels to 0. However, just applying a fixed intensity threshold often gives relatively poor results due to varying background and foreground intensities across the image.

# Adaptive thresholding solves this problem by first creating a "background image", which can then be used as a locally variable threshold. The goal in generating this background image is to generate an image that always has higher intensities than the local background but lower intensities than the local foreground. This is often achieved by strong blurring/smoothing of the image. 


# EXERCISE

# Create an adaptive background and use it to segment the cell membranes by thresholding. 

# Two steps are involved:

# Step 1: Create the background by allocating to each pixel the average values of its neighbourhood (essentially a strong local blurring). In practice, this involves running a mean filter across the image. Which pixels should be considered as 'neighborhood' in this mean filter is defined by a structuring element (se), a small binary image.

# Step 2. Use the adaptive background for thresholding: Pixels with higher values in the original image than in the background should be given the value 1 and pixels with lower values in the original image than in the background should be given the value 0. The resulting binary image should represent the cell membranes.


# Step 1

    # (i) Create a disk-shaped structuring element and asign it to a new variable. Structuring elements are small binary images that indicate which pixels should be considered as 'neighborhood' of the central pixel. An example of a small disk-shaped structuring element would be this:
    # 0 0 1 0 0
    # 0 1 1 1 0
    # 1 1 1 1 1
    # 0 1 1 1 0
    # 0 0 1 0 0

    # The equation below creates such structuring elements. It is an elegant but complicated piece of code and at the moment it is not necessary for you to understand it in detail. Use it to create structuring elements of different sizes (by changing 'i') and find a way to visualize the result. Try to answer the following questions: Is the resulting se really circular? Can certain values of 'i' cause problems? If so, why? Why is a circular se better than a square one? What size should be used for the se? Note that, similar to the sigma in Gaussian smoothing, the size of the se is first estimated based on the images and by thinking about what would make sense. Later, it can be optimized by trial and error.

struct = np.mgrid[:i,:i][0] - np.floor(i/2)**2 + (np.mgrid[:i,:i][1] - np.floor(i/2))**2) <= np.floor(i/2)**2


# (ii) Create the background: 

    # Run a mean filter over the image using the disc structure and assign the output to a new variable (suggested name for variable: 'bg'). Use the function 'skimage.filters.rank.mean' (you first need to import the 'skimage.filters.rank' module). Think about why a mean filter is used and if a different function (e.g. minimum, maximum or median) would work equally well.

from skimage.filters import rank           # Import module containing mean filter function
bg = rank.mean(-----, selem=-----)         # Run a mean filter over the image using the disc



# (iii) Visualise the resulting background image. Compare it to the images generated above. Does the outcome make sense?
plt.imshow(---,---,cmap='gray')
---  


# Step 2  

# (iv) Threshold the image 'green_smooth' using created background 'bg' to obtain the cell membrane segmentation:

    # Set pixels with higher values in the original than in the bg to 1 and pixels with lower values to 0. You can use a relational operator to do this, since numpy arrays will automatically perform element-wise comparisons when compared to other arrays of the same shape. Remember to allocate the output to a new variable. (Suggested name for variable: 'green_mem')
green_mem = --- >= ---


# (v) Visualise and understand the output. What do you observe? Are you happy with this result as a membrane segmentation? 
plt.imshow(---,---='none',cmap=---)
plt.show---


# (vi) Further clean the output using binary morphological operations.

    # You can try out dilation, erosion, opening and closing (all available in ndimage, for example 'ndi.binary_closing') and see what is better in terms of reducing false positives (spots of foreground inside the cells) and false negatives (gaps in membranes). 

    # A less common but very useful morphological operation to get rid of false positive spots is binary hole filling, which removes objects that are not connected to the image boundaries. The corresponding ndimage function is 'ndi.binary_fill_holes'. Assign your final result to a new variable (suggested name for variable: 'green_mem').
    
    # Pick one from the two options below:
#(a)
green_mem = ndi.binary_fill_holes(np.logical_not(green_mem))

#(b)
green_mem = ndi.binary_fill_holes(green_mem)


# (vii) Visualize the final result, then go back and tune the size of the se and the morphological cleaning operations until you are happy with the resulting membrane segmentation.
plt.---(---,---='none',cmap=---)
---



#%%
#------------------------------------------------------------------------------
# SECTION 4 - CONNECTED COMPONENTS LABELING (OR: "WE COULD BE DONE NOW")

# If the data is clean and we just want a very quick cell or membrane segmentation, we could be done now. All we would still need to do is to label the individual cells - in other words, to give each separate "connected component" an individual number.


# EXERCISE

# (i) Label connected components:

     # Collections of consecutive pixels that have the same value are considered as connected regions and, if the segmentation has been done correctly, they correspond to specific objects of the image, for example different cells. We can label each of them with a different integer value using the function 'ndi.label'. This enables us to make quantitative measurements for each cell individually. (Suggested name for variable: 'green_components'). Pick one of the three options below:
#(a)
green_components = ndi.label(green_mem)[0] 
#(b)
green_components = ndi.label(green_mem) 
#(c)
green_components = ndi.label(green_mem)[1] 

# (ii) Visualise the output.
plt.imshow(---,---,---)    
---


# OBSERVATION
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

# We can find seeds based on a membrane segmentation using a distance transform, which essentially maps the distance of each pixel in the cells from the nearest membrane pixel. Maxima (or 'peaks') in the resulting distance 'map' can be considered seeds. Unlike the labeling we do above, this approach can generate more than one seed even if two cells are connected by a gap in the membrane. Thus, it allows us to split up cells that would be wrongly fused by connected component labeling.


# EXERCISE

# (i) Distance transform on thresholded membranes:

    # The distance transform assigns each foreground pixel a value corresponding to its distance from the closest background pixel. To apply a distance transform, use the function 'ndi.distance_transform_edt'. (Suggested name for output variable: 'green_dt'). 

green_dt= ---(---)


# (ii) Visualise the output and understand what you are seeing.
plt.---(green_dt,interpolation='none')
---


# (iii) Retrieve the local maxima (the 'peaks' in the distance transform 'map')

    # Import the function 'peak_local_max' from the module 'skimage.feature'
--- skimage.feature --- peak_local_max

    # Detect local maxima: use peak_local_max to detect the local maxima. Use the optional arguments min_distance=10 and indices=False. Remember to use the documentation to understand the function and its arguments. (Suggested name for variable: 'green_max')
green_max = peak_local_max(---,indices=---,min_distance=---) 

    # Label the connected components to give each seed an individual number. (Suggested name for variable: 'green_seeds')
green_max = ndi.label(---)[0]  # Understand why [0]    


    
# (iv) Visualise the output by showing the seeds as a masked overlay. In other words, you should overlay the maxima (the seeds) on the original input image (or on smoothed version):

    # To do this, it is important to first understand a key point about how the pyplot module works: every plotting command is slapped on top of the previous plotting commands, until everything is ultimately shown when 'plt.show' is called. Hence, you can first plot the input (or smoothed) image and then plot the seeds on top of it before showing both with 'plt.show'.

    # As you can see if you try this, you will not get the desired result because the zero values in 'green_seeds' are painted in black over the image you want in the background. To solve this problem, you need to mask these zero values before plotting 'green_seeds'. You can do this using the function 'np.ma.array' to create a masked array.

    # What do you observe?

plt.imshow(green_smooth,---,---)
plt.imshow(np.---(---,---=green_max==0),interpolation=---) 
plt.show()

    # What do you observe? (The seeding should probably be improved.)


# (v) Dilating (maximum filter) of distance transform:

    # Ideally, there should be only one seed per cell. However, irregularities in the membrane shape can cause multiple distance transform peaks in each cell. This can be improved by cleaning the distance transform using a dilation before detecting the maxima. 
    
    # Use ndi.filters.maximum_filter to dilate the distance transform. Read the documentation to remind yourself how and where the structuring element can be defined with this function. You can try different structuring element sizes and shapes, if you like. 
green_dt = ndi.filters.---(---,---) 


# (vi) Visualise the output.
plt.imshow(---,---,---)
---


# (vii) Retrieve and label the local maxima again:

    # Local maximum detection (as above, use the 'peak_local_max' function). (Suggested name for variable: 'green_max')
green_max = peak_local_max(---,indices=---,min_distance=---) 

    # Label the connected components (Suggested name for variable: 'green_seeds')
green_max = ndi.---(---)[0]  


# (viii) Visualise the output by showing maxima as masked overlay again. Did the dilation improve the seeding? If you're not happy yet, you can try to optimize the seeding by playing with the parameters of the dilation (in other words, with the se size and shape). 
plt.imshow(green_smooth,---,---)
plt.imshow(np.---(---,---=green_max==0),interpolation=---) 
plt.show()



# -------
# Part II
# -------

# Watershedding is a relatively simple but powerful algorithm for expanding seeds. The image intensity is considered as a topographical map (with high  intensities being "mountains" and low intensities "valleys") and water is  poured into the valleys from each of the seeds. The water first labels the lowest intensity pixels around the seeds, then continues to fill up. The cell boundaries (discontinuities or significant transitions in an image) are where the "waterfronts" between different seeds touch.

# EXERCISE

# Watershed:

    # Get the 'watershed' function from 'skimage.morphology'. 
--- skimage.morphology --- ---
    
    # Tun the watershed, using green_smooth as the image topography and green_seeds as seeds. You can also try using the (inverted) distance transform as image topograpy if you're curious to see how the result differs from using green_smooth. Think about the advantages and disadvantages of either approach. (Suggested name for output variable: 'green_ws')
green_ws = ---(---,---)
    
    # Show the result as transparent overlay over the smoothed input image. This can be done similar to the masked overlay of the seeds, but now you don't need to mask the background in the overlayed image - instead, you need to make the overlayed image semi-transparent. This can be achieved using the optional argument 'alpha' of the 'plt.imshow' function to specify the opacity.
plt.imshow(---,interpolation=---,alpha=---) 
---

# OBSERVATION
# Note that the previously connected cells are now mostly separated and the membranes are partitioned to their respective cells. Depending on the quality of the seeding, however, there may now be some cases of oversegmentation (a single cell split into multiple segmentation objects). This is a typical example of the trade-off between specificity and sensitivity one always has to face in computational classification tasks. As an advanced task, you can try to think of ways to fuse the wrongly oversegmented cells back together.   


#%%
#------------------------------------------------------------------------------
# SECTION 6 - IDENTIFICATION OF CELL EDGES

# Now that we have a full cell segmentation, we can retrieve the cell edges, that is the pixels bordering neighboring cells. This is useful for many purposes; in our case, for example, edge intensities are a good measure of membrane intensity, which may be a desired readout. The length of the edge (relative to cell size) is also an informative feature about the cell shape. Finally, showing colored edges is a nice way of visualizing cell segmentations.

# There are many ways of identifying edge pixels in a fully labeled segmentation. It can be done using erosion or dilation, for example, or it can be done in an extremely fast and fully vectorized way (for this, see "Vectorization" in the optional advanced content). Here, we use a slow but intuitive method that also serves to showcase the 'generic_filter' function in ndimage.

# 'ndi.filters.generic_filter' is a powerful way of quickly iterating any function over numpy arrays (including functions that use a structuring element). 'generic_filter' iterates a structure element over all the values in an array and passes the corresponding values to a user-defined function. The result returned by this function is then allocated to the pixel in the image that corresponds to the origin of the se. Check the documentation to find out more about the arguments for 'generic_filter'.


# EXERCISE

# (i) Define a function called 'edge_finder' that you can pass to 'ndi.filters.generic_filter':
    # Your 'edge_finder' function should detect if the central pixel in a 3x3 array is at an edge. This is true whenever the central pixel is not identical to all other pixels in the 3x3 array. More specifically, your function should...
    # ...take in as input the values from a 3x3 se (representing a neighbourhood around a pixel), 
    # ...check if all pixels are the same or not (function 'np.all' is helpful here)
    # ...return 1 or 0, respectively.

    # Remember that to define a new function the syntax is as follows:
    #
    # def function_name(input arguments): 
    #   """function documentation string"""
    #   function procedure
    #   return [expression]

--- edge_finder(footprint_values):
    --- (footprint_values == footprint_values[0]).all():
        return ---
    else:
        ---
   
   
# (ii) Iterate your 'edge_finder' function over the segmentation ('green_ws') using the 'ndi.filters.generic_filter' function, (with optional argument size=3 to get a 3x3 kernel). Allocate the output to a new variable (suggested name: 'green_edges')
--- = ---(green_ws,---,size=3)


# (iii) Label the detected edges:

     # Based on the underlying cell, each edge pixel can be labeled coorrectly. (Suggested output name: 'green_edges_labeled').
    # Hint: Use array multiplication - 'green_edges' is a binary array (consists of 1s and 0s) and 'green_ws' contains the cells nicely numbered.
green_edges_labeled = green_edges * green_ws


# (iv) Visualize as a masked overlay of labaled edges over the smoothed original image.
---(green_smooth,---,---)
plt.imshow(---(---,---green_edges_labeled==0),---) 
---




#%%
#------------------------------------------------------------------------------
# SECTION 7 - POSTPROCESSING: REMOVING CELLS AT THE IMAGE BORDER

# Segmentation is never perfect and it often makes sense to remove artefacts afterwards. For example, one could filter out objects that are too small, have a very strange shape, or very strange intensity values. Note that this is equivalent to the removal of outliers in data analysis and should only be done for good reason and with caution.

# As an example of postprocessing, we will now filter out a particular group of problematic cells: those that are cut off at the image border.


# EXERCISE

# (i) Create image border mask:

    # Remember that a mask is a binary array with a specific size and distribution of ones and zeros. To identify the cells that border the image and should thus be excluded from the analysis, you first need to create a mask where the pixels at the image boundary are set to 1 and everything else is set to 0. (Suggested name for output variable: 'boundary_mask')
    # Hint: There are different ways of achieving this, for example by erosion or by array indexing. The function 'np.ones_like' may be useful. 
--- = np.ones_like(---) 
boundary_mask[---] = ---             



# (ii) 'Delete' the cells at the border:

    # Iterate over all cells in the segmentation. This is easily done using a 'for' loop and the function 'np.unique'. Remember that 'green_ws' has each cell labaled with a different integer.

    # For each cell, identify if it has pixels touching the image boundary or not. There is an easy solution to this that relies on relational operations and some basic arithmetic.

current_label = 1
for cell_id --- np.unique(green_ws):
    
    # "Delete" the cells at the boundary, by changing their intensity values to 0.

    --- np.sum((green_ws==cell_id)*boundary_mask) != 0:
        green_ws[green_ws==---] = 0
        
    # (Optionally, relabel the cells not at the boundary to keep the numbering consistant from 1 to N (with 0 as background).
    else---
        green_ws[green_ws==cell_id] = current_label
        current_label = ---
        

# (iii) Visualise result:
    # Show the result as transparent overlay of the watershed over the blurred original. You now have to combine both alpha (to show cells transparently) and 'np.ma.array' (to hide empty space where the border cells were deleted).
---(green_smooth,cmap='gray',interpolation='none')
plt.imshow(np.ma.array(---,mask=green_ws==0),---,alpha=---) 
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


# EXERCISE

# (i) Create a dictionary that contains a key-value pairing for each of the measurements mentioned above. The key should be a string describing the type of measurement (e.g. 'green_intensity_mean') and the value should be an empty list. We will fill these empty lists with the results of our measurements and the dictionary will make it easy to retrieve this data.
results = ---"cell_id"---[], "green_mean"---[], "red_mean":[],"green_membrane_mean":[], 
           "red_membrane_mean":[],"cell_size":[],"cell_outline":[]---    


# (ii) Record the measurements:
    
    # Iterate over segmented cells using a for loop and the 'np.unique' function (you've done this before in exercise (ii) of section 7)
for cell_id in np.unique(green_ws)[1:]:
    
        # Mask away everything in the image except the current cell
        cell_mask = green_ws --- cell_id  
        edge_mask = np.logical_and(---,green_edges)
        
        # Use the masked image to acquire the measurements listed above. Add them to the appropriate list in the result dictionary (you can use the list method 'list.append' to add elements to a list). Note that you should use the original raw data for quantification. 
        results["cell_id"].append(cell_id)
        results["green_mean"].append(np.mean(img[0,:,:][---]))
        results["red_mean"].append(np.mean(img[---,:,:][cell_mask]))    
        results["green_membrane_mean"].append(np.mean(---[0,:,:][edge_mask]))    
        results["red_membrane_mean"].append(np.mean(img[1,:,:][---]))    
        ---["cell_size"].append(np.sum(cell_mask))
        results["cell_outline"].---(np.sum(edge_mask))
    
# Hint 1: If you have saved the final segmentation in 'green_ws', you can run the following code to see what is meant by "masking away" everything except the current cell. For membrane measurements you can use 'green_edges' instead.
#cell_mask = green_ws==1
#plt.imshow(cell_mask)
#plt.show()
    
# Hint 2: Recall that you can index an array with another (logical) array. This makes the acquisition of measurements for each cell much easier. This is a simple example (note that the raw data is used):
#cell1 = img[0,:,:][cell_mask]
#print cell1
#cell1_mean = np.mean(cell1)
#print cell1_mean



#%%    
#------------------------------------------------------------------------------
# SECTION 9 - SIMPLE ANALYSIS AND VISUALIZATION

# Now that you have collected the readouts to a dictionary you can analyse them in any way you wish. This section shows how to do basic plotting and analysis of the results, including mapping the data back onto the image (as a 'heatmap') and producing boxplots, scatterplots and a linear fit. A more in-depth example of how to couple image analysis into advanced data analysis can be found in 'data_analysis' in the 'optional_advanced_material' directory.


# EXERCISE

# (i) First get yourself an overview of what you're working with by printing the results. You can print all results by iterating over the dictionary. Make sure you fully understand the data structure (the dict and the lists contained within) before you proceed.
for key --- results.keys()---
---print "\n" + key
    --- results[key]
    

# (ii) Create a box plot showing the mean cell and mean membrane intensities for both channels. You first need to retrieve the data you need from the dictionary and structure it properly so you can pass it to 'plt.boxplot'. Use the 'label' keyword of 'plt.boxplot' to label the x axis with the corresponding key names. 
plt.---( [results[key] for key in results.keys()][2:---], labels=results.keys()[---:-1])
plt.show---


# (iii) Create a scatter plot of green membrane intensity over cell size. Add a linear fit to the plot to viszualize a potential correlation.

    # Import the module 'stats' from 'scipy'       
from --- import ---

    # Use the function 'stats.linregress' to do a linear fit of membrane intensity vs cell size. Be sure to read the documentation to understand the output of this function.
linfit = ---(results[---],results[---]) 
        
    # Create a scatter plot of the mean green membrane intensity vs cell size. Use the function 'plt.scatter' to do so. Remember to label the axes, using the functions 'plt.xlabel' and 'plt.ylabel'. 
---(results[---],results[---])
plt.xlabel(---)
---("red_membrane_mean")

    # Define the equation of the line that fits the data. Recall that the equation of a straight line in 2D is y = m * x + c, where m is the slope and c is the intersection of the y-axis. You need to create a function that takes in a value (x) and returns the fitted value (y). You can define your function normally (using the 'def' keyword') or if you are feeling fancy you can create an anonymous function using 'lambda'. The example below shows the difference between a normal function and an anonymous function:

         # # normal function definition
         # 	def f(x): 
         #      """Returns the square value"""
         #      return x**2
         #
         # anonymous function
         # f = lambda x: x**2       # returns the square value
fit = --- x: linfit[---] * x + linfit[---]

    # Next, you need to get the actual fitted values to plot your line. You only need to evaluate your function at two points at the limits of your scatter plot so you can draw the line across the entire plot. You can get the limits for the x axis using 'plt.gca' and then 'get_xlim'. Once you have these x values, you can get the corresponding fitted y values using your line function (if you used 'lambda' to define it, use 'map' to get the result).
--- = plt.gca()        
x_lims = ax.---
fit_vals = map(---,x_lims)

    # Plot the line         
        # Plot the line using 'plt.plot'. You can specify the aesthetics of the line, e.g. line width, color, and so forth - see the documentation.
plt.plot(---,---,'r-',lw=2)
        
    # Finally, use 'plt.show' to show the plot.
plt.show()


# (iv) Print out results from the linear regression (again, check the documentation of 'stats.linregress'):

    # Define a list with the names of the output of the linear regression (e.g. 'slope',...)
linnames = ["slope","intercept",---,---,---]   

    # (optional) Header 
print "\nLinear fit of cell size to red membrane intensity"              

    # Iterate over all values in the 'stats.linregress' output.
for index,value in ---(linfit):  
        # Print the output value with the corresponding name. To simultaneously iterate over the output of 'stats.linregress' and over your list of output names, the function 'enumerate' is very helpful. You can check the docs or the web to find out how to use 'enumerate' in for-loops.
        print "  " + linnames[index] + "\t\t" + str(value)  
    
    # (After the for loop), also calculate and print the r-squared value
print "  r-squared\t\t" + str(linfit[2]**2)   

    # Note that this seems to return a highly significant p-value but a very low correlation coefficient (r-value). We also would not expect this correlation to be present in our data. This should prompt several considerations:
        # 1) What does this p-value actually mean? See help(stats.linregress)
        # 2) Could there be artefacts in our segmentation that could bias this analysis?
        # 3) We're now working with a lot of datapoints - this can skew statistical analyses. We can (and should!) accommodate this by multiple testing correction and by comparison with randomized datasets.
    # In general, it's always good to be very careful when doing data analysis. Make sure you understand the functions you are using and always check for possible errors or biases in your analysis!
    

# (v) Map the cell size back onto the image as a 'heatmap':

    # Scale the cells sizes to 8bit (since we need them as pixel intensity values). 
    # Hint: if the largest cell size should correspond to the value 255 in uint8, then the other cell sizes correspond to cell_size*255/largest_cell_size.
sizes_8bit = (255 * ---)/---  
    
    # Initialize a new image; all values should be zeros, the size should be identical to e.g. 'green_ws' and the dtype should be uint8. (Suggested name for variable: 'size_img')
 size_map = ---(green_ws,dtype=---)    
    
    # Iterate over the segmented cells of 'green_ws'.(You need a for-loop, and the functions 'enumerate' and 'np.unique'.)
for ---,--- in enumerate(np.unique(green_ws)[1:]):          
        # Assign to each pixel of 'size_img' the cell size (in uint8) of the cell it corresponds to in 'green_ws'
    size_map[green_ws==---] = sizes_8bit[---]               
    
    # Visualize the result as a colored semi-transparent overlay over the smoothed input image 'green_smooth'.
---
plt.imshow(np.ma.--(---,mask=size_map==---),interpolation=---,alpha=---)  
---

# (vi)
# Do an analysis of statistical tests and the rest of the measurements. Think about what the results actually mean and whether any inconsistancies can be traced back to a not sufficiently good segmentation, e.g. we  have filtered for some artifacts, such as incomplete cells (that touch the boundary), but not for others; can you think of some?.



#%%    
#------------------------------------------------------------------------------
# SECTION 10 - WRITING OUTPUT TO FILES

# There are several ways of saving the output of a program. Data can be saved to files in a human-readable format such as text files (e.g. to import into Excel), in a format readable for other programs such as tif-images (e.g. to visualize in Fiji) or in a language-specific file that makes it easy to reload the data into python in the future (e.g. for further analysis).

# EXERCISE

# (i) Write an image to a tif (could be opened e.g. in Fiji):

    # Get the file handling function 'imsave' from the package 'tifffile'
from --- import ---                                          
    
    # Save one of the images you've generated using 'imsave' and open it in Fiji
imsave(filename, data)  


 
# (ii) Write a figure to a png or pdf:

    # Recreate the scatter plot from above (without the regression line). Remember to label axes.
    
    # Save the figure to png using 'plt.savefig'
plt.savefig(filename)  
    
    # Save the figure to pdf using 'plt.savefig'. This creates a vector graphic that can be imported to illustrator etc.
plt.savefig(filename)  




# ***For the next two exercises, refer to the python documentation for input and output.***

# (iii) Write a python file that can be reloaded in other Python programs:

    # Import the module 'json'
--- json    
    
    # Open an empty file object using 'open' in write ('w') mode. Use a name that contains the extension '.json'. You should use the 'with'-statement (context manager) to make sure that the file object will be closed automatically when you are done with it.
with open(filename, 'w') as resout:  
    
    # Use the function 'json.dump' to write the results.    
        ---(results, ---)

# Note: This file could be re-loaded again as follows:
#with open(filename+'_resultsDict.json', 'r') as fp:
#   results = json.load(fp)


# (iv) Write a text file of the numerical data gathered (could be opened e.g. in Excel):

# Open an empty file object using 'open' in write ('w') mode. Use a name that contains the extension '.txt'. You should use the 'with'-statement (context manager) to make sure that the file object will be closed automatically when you are done with it.
--- ---(filename+"_output.txt","w") --- txt_out:  
    
    # Write the headers of the data (the result dictionary keys), separated with tabs ('\t'). You will need the function 'file.write' to write strings to the file. It makes sense to first generate a complete string of all the headers with a loop and then write it to the file.
    txt_out.---(''.---(key+'\t' for key in results.keys()) + '\n')                  

    # Iterate over all cells saved in your results variable using a for-loop and the function 'enumerate'.
    --- index,value in ---(results["cell_id"]):                                        
        
        # For each key in the dict, write the data to the text file, separated with tabs ('\t').
        txt_out.---(''.join(str(results[key][index])+'\t' for key in results.keys()) + '\n')   # Write cell data
        
        # After writing the data, have a look at the output file in a text editor.        
            
 #------------------------------------------------------------------------------
 #------------------------------------------------------------------------------
 # THIS IS THE END OF THE TUTORIAL.
 #------------------------------------------------------------------------------
 #------------------------------------------------------------------------------

        
        
