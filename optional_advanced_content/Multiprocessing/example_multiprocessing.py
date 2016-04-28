# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 20:59:1 2016

@author:   Jonas Hartmann @ Gilmour Group @ EMBL Heidelberg

@descript: Multiprocessing is a simple way of increasing the speed of code if 
           it is impossible, insufficient or otherwise undesirable to do so by
           vectorization. Essentially, multiprocessing simply means running 
           different independent parts of a program (for example an function
           that is run again and again on different data) at the same time
           instead of sequentially in a loop. This is an example of using 
           multiprocessing to run the batch pipeline from the main tutorial on
           multiple different images at the same time, instead of processing 
           them one by one.
           
           In Python, multiprocessing is handled in the multiprocessing module. 
           The easiest way of using it is to initialize a pool of "worker" 
           processes, which are then available to run the functions passed to
           them (or "mapped onto them"). Although this is relatively easy to 
           do, multiprocessing has some quirks that need to be payed attention:

           1) Functions passed to worker processes can take at most one object
              as input and return at most one object as output. If multiple
              parameters need to be passed, they must be packaged into a single
              object first (and then unpacked at the beginning of the function).
              
           2) If functions write to files, print output or display graphs, 
              great care is advised during multiprocessing, as the different 
              subprocesses may try to do these things at the same time, which 
              may result in a garbled chaos or even a crash.
 
           3) Every worker process will start out by automatically trying to 
              set up the same "environment" as the main process. This 
              effectively means that each subprocess tries to execute the main
              script again at the start, which could obviously have catastrophic 
              consequences. To prevent this, the main script must be "protected". 
              This is done through the built-in variable __name__, which has the 
              value "__main__" if the script is called from the main process and 
              a different value if it's called by a worker process. This can be
              exploited to make sure that the main script is not completely 
              re-run by each subprocess (see the beginning of this script).   

           The following describes an example of how to run the batch pipeline 
           using N parallel processes. It requires batch_multiprocessing.py, 
           which is a "cleaned" version of the batch pipeline that accommodates 
           the three quirks mentioned above. Note that all code outside of the
           actual pipeline function has been deleted to avoid a similar problem 
           to (3) during the import of the function (Python executes all
           non-protected code blocks in a module when that module is imported!).

           Execution of the following example for 4 copies of the same image 
           takes ~73s on my machine (2 available cores). Running the 4 copies 
           without multiprocessing would take ~144s.

@requires: Python 2.7
           NumPy 1.9, SciPy 0.15, matplotlib 1.5.1, scikit-image 0.11.3

"""

# IMPORT BASIC MODULES

from __future__ import division    # Python 2.7 legacy
import numpy as np                 # Array manipulation package
import matplotlib.pyplot as plt    # Plotting package
import scipy.ndimage as ndi        # Image processing package


#------------------------------------------------------------------------------
   
# PROTECTION OF THIS SCRIPT FOR MULTIPROCESSING
# When subprocesses are initialized, they will first try to run this main 
# script again (this is done to set up the environment/name space properly).
# Since we do not want the following to be run again and again, we have to 
# protect it. 
# The built-in variable __name__ is automatically set to "__main__" in the main
# process but has other values in the subprocesses, which means those processes
# will ignore the code block within the following if-statement:

if __name__ == '__main__':
    
    
    #--------------------------------------------------------------------------
    
    # PREPARATION
    
    # Begin timing
    from time import time
    before = time()    
    
    # Generate a list of image filenames (just as in main tutorial)
    from os import listdir, getcwd
    filelist = listdir(getcwd())
    tiflist = [fname for fname in filelist if fname[-4:]=='.tif']
    
    # Prepare for multiprocessing
    N = 4                         # Maximum number of processes used
    import multiprocessing.pool   # Import multiprocessing class
    currentPool = multiprocessing.Pool(processes=N)    # Create a pool of worker processes
    from batch_multiprocessing import pipeline         # Import cleaned pipeline function
    
    
    # EXECUTION
    
    # Here, the function pipeline is executed by the current pool of worker 
    # processes for each parameter (filename) in the tiflist and the output is
    # written into the output_list.
    output_list = currentPool.map(pipeline,tiflist)
    
    # This is necessary clean-up to make sure that all worker subprocesses are
    # properly terminated. It's more of a "safety" thing, since things can
    # *really* go wrong in multiprocessing...
    currentPool.close()
    currentPool.join()
    
    # Reorganize the output into the same shape as in the batch tutorial
    all_results = [output[1] for output in output_list if output != "ERROR"]
    all_segmentations = [output[0] for output in output_list if output != "ERROR"]
    
    
    # DOWNSTREAM HANDLING
    
    # End Timing    
    after = time()
    print after - before
    
    # See if it worked by printing the short summary  
    print "\nSuccessfully analyzed", len(all_results), "of", len(tiflist), "images"
    print "Detected", sum([len(resultDict["cell_id"]) for resultDict in all_results]), "cells in total"
    
    # See if it worked by showing the scatterplot
    colors = plt.cm.jet(np.linspace(0,1,len(all_results))) # To give cells from different images different colors
    for image_id,resultDict in enumerate(all_results):
        plt.scatter(resultDict["cell_size"],resultDict["red_mem_mean"],color=colors[image_id])
    plt.xlabel("cell size")
    plt.ylabel("red_mem_mean")
    plt.show()
    
    
    #--------------------------------------------------------------------------



