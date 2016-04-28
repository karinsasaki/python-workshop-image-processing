# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 16:28:01 2016

@author:    Jonas Hartmann @ Gilmour Group @ EMBL Heidelberg

@descript:  An example of how vectorization can speed up data analysis code.
            Vectorization refers to the removal of iterations in favor of array
            operations. These generally compute much faster since they are 
            implicitly parallel; if the same operation is done for all elements
            of an array independently - i.e. without a loop - multiple such 
            operations can be done in parallel (or at least in quick succession
            without any overhead) on the CPU.
            
            This example is based on the cell segmentation generated in the 
            main tutorial script, which is loaded from a npy file. The example
            shows a vectorized version of the edge pixel filter.

@moreInfo:  We previously used scipy.ndimage.generic_filter as a means of 
            iterating over our segmentation and detecting the edges of each 
            cell. Whilst generic_filter provides a means of fast array 
            iteration, it would be much faster to use a vectorized approach,
            i.e. one that does not rely on iteration at all.
            
            One way to find cell border pixels without iterating is to generate 
            all four possible "shifted-by-1" versions of the array. The edge 
            pixels are those that do not have the same value in one of the 
            shifted arrays as compared to the original. Since array comparison 
            does not require iteration, this approach is bound to be much 
            faster, especially for large arrays.
            
            However, there is a trade-off: generating 4 shifted copies of the 
            image array requires a lot of memory, which can be a problem for 
            big data. Such trade-offs between memory and speed are a common 
            concern in code optimization.
            
            Note that vectorization is actually quite easy for a lot of common
            operations in programs; it just takes a bit of thinking and often
            some knowledge of linear algebra. However, there are also cases 
            where the solution is not obvious or easily derived (this example
            is probably one such case - at least it was for me). In those
            cases, searching the internet for solutions to the problem (or a 
            similar problem) is usually worth a try.
            
@speed:     This version takes ~0.016s to run on my machine, versus ~5.318s for 
            the iteration-based implementation in the main tutorial.    

@requires:  Python 2.7
            NumPy 1.9, scikit-image 0.11.3, matplotlib 1.5.1
"""


# PREPARATION

# Module imports
from __future__ import division    # Python 2.7 legacy
import numpy as np                 # Array manipulation package
import matplotlib.pyplot as plt    # Plotting package

# Data import (segmentation from main tutorial)
filename = 'example_cells_1' 
seg = np.load(filename+'_segmented.npy')

# Begin timing
from time import time
before = time()


### EXECUTION

# Padding adds values around the original array (here just 1 line of pixels)
seg_pad = np.pad(seg,1,mode='reflect')

# This generates a list of shifted-by-1 arrays by slicing sub-blocks out of the
# padded original.
seg_shifts = [seg_pad[:-2,:-2],seg_pad[:-2,2:],seg_pad[2:,:-2],seg_pad[2:,2:]]

# Now it's just a matter of checking which pixels are different in a shifted
# array compared to the original
edges = np.zeros_like(seg)
for shift in seg_shifts:
    edges[shift!=seg] = 1
    
# Label the detected edges based on the underlying cells (as in the main tutorial)
edges = edges * seg


### DOWNSTREAM HANDLING

# End timing
after = time()
print after - before

# Show result as masked overlay (as in the main tutorial)
import skimage.io as io
img = io.imread(filename+'.tif')
plt.imshow(img[0,:,:],cmap='gray',interpolation='none')
plt.imshow(np.ma.array(edges,mask=edges==0),interpolation='none') 
plt.show()



    