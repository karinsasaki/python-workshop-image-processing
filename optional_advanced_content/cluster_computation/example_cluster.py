# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 00:12:38 2015

@author:    Jonas Hartmann @ Gilmour Group @ EMBL Heidelberg

@descript:  Sometimes a single computer just doesn't cut it; a computer cluster 
            is required. At EMBL, we have access to a high-performance
            computation (HPC) cluster with over 4000 CPUs, see:
                
                https://intranet.embl.de/it_services/services/computing/hpc_cluster/index.html

            The HPC cluster is used by submitting jobs from a (linux-based) 
            server, using a queuing system called "LSF". IT offers courses on
            how this is done, and since many people are using the cluster, it 
            is good to know what you are doing before trying it yourself, to 
            avoid causing problems for others.
            
            For those who already know about LSF (or plan to learn about it),
            this is an example of how cluster computation could be handled with
            Python, using the batch processing pipeline established in the main
            tutorial. However, instead of submitting to the cluster, this 
            script creates Python processes on the local machine, making it 
            more or less equivalent to multi-processing.
            
            In principle, cluster handling requires two things: job submission
            and result collection. Here, the analysis pipeline is submitted 
            with each image as a job and the resulting segmentations are
            collected when those jobs finish. Doing this on the clsuter would
            be slightly more complicated than doing it locally, but those who
            know about HPC/LSF should be able to figure it out.
            
            NOTE 1: This uses a "cleaned" version of the batch pipeline, which
            takes the input filename from a commandline argument and saves its
            output into a file.

            NOTE 2: This code is every so slightly dependent on the operating
            system used and on the paths of Python and the input files on the
            system. The current version is written for Windows and lines tagged 
            as #OS! have to be adjusted for linux (and in some cases also for
            Windows machines if the paths are different).

@requires:  Python 2.7
            NumPy 1.9, SciPy 0.15, scikit-image 0.11.3, matpliotlib 1.5.1
"""

# IMPORT BASIC MODULES

from __future__ import division    # Python 2.7 legacy
import numpy as np                 # Array manipulation package
import matplotlib.pyplot as plt    # Plotting package
import json                        # Writing and reading python objects


# PREPARATION

# Generate a list of image filenames (just as in the batch tutorial)
from os import listdir, getcwd
filelist = listdir(getcwd())
tiflist = [fname for fname in filelist if fname[-4:]=='.tif']


# SUBMISSION

# For each filename, use the commandline to execute the batch pipeline script
# with that filename as an input.
from os import system    # Function to run commandline commands
print "Submitting jobs..."
for fname in tiflist:
    
    system('python batch_cluster.py "'+fname+'"') #OS!
    
    # For cluster submission, it would look something like this. However, note
    # that this is pseudo-code and would have to be adjusted at least slightly!
    #system("bsub -o out.txt -e error.txt 'python batch_cluster.py "+fname+"'")


# RESULT COLLECTION

all_results = [] # Initialize result list
all_done = []    # This is used to check which images have already been processed
errors = 0       # This is used to count errors

# A while-loop to keep looking until all the output files have been retrieved.
# Note that unexpected errors within the pipeline may cause this to become an 
# infinite loop; it would be better to implement this in a clean fashion that
# handles exceptions in the pipelines properly (or at least stops automatically
# after a certain amount of waiting time).
while len(all_done) != len(tiflist):
    
    # Wait for 30 sec
    print "Waiting for results..."
    from time import sleep
    sleep(30)
    
    # Check for output files
    filelist = listdir(getcwd())
    outlist = [fname for fname in filelist if '_out.json' in fname]
    all_done = all_done + outlist
    
    # For each available output file...
    for fname in outlist:
        
        # Load the output data
        with open(fname, 'r') as fp:
            results = json.load(fp)
            
            # Make sure errors are caught...
            if type(results) == str:
                errors += 1
                
            # If all is well, add the result to all other results
            else:
                all_results.append(results)

        # Then just delete the file; we don't need it anymore
        system('del "'+fname+'"') #OS!
    
    # Report on progress, then wait for a minute, then try again
    print "Retrieved", len(all_done), "result files of", len(tiflist), "with a total of", errors, "errors!"


# DOWNSTREAM PROCESSING
  
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
    
