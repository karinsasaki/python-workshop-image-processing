# Advanced Content

### Code Optimization

Examples for how to speed up your code, relevant for anything that handles relatively large amounts of data (image analysis, data analysis, modeling, ...). There are scripts exemplifying three strategies:

- **Vectorization**

	- Using the edge finder from the main script as an example, this demonstrates the drastic 		 	   increase in speed that can often be achieved if operations are vectorized.
	
- **Multiprocessing**

	- This shows how Python's multiprocessing module can be used to simultaneously run the batch 	  		pipeline from the main tutorial on several images.
	
- **Cluster Processing**

	- An example of how to use a Python script to run another script multiple times with different 	   input data. If run locally, this is very similar to multiprocessing, but with a bit of 	 		  knowledge about high-performance cluster computing (see appropriate courses), this approach can 		be used to handle job submission and result collection on a computer cluster.

It should be noted that one of the key aspects of code optimization is finding out *which part* of the code costs the most time and could be optimized for the greatest gain in speed. This is called *profiling* and there are a number of options for how to do it, both in the form of Python modules as well as built into IDEs like Spyder. Profiling is not discussed here, but as a very simple example the `time` module is used to test how long the different versions take to run.


### Advanced Data Analysis

This tutorial illustrates how single-cell segmentation results can be piped into advanced data analysis. This is intended as a starting point for people to get into advanced data analysis with python. In particular, it shows off three important modules (scikit-learn, scipy.cluster and networkx) and illustrates a number of key concepts and methods (feature extraction, standardization/normalization, PCA and tSNE, clustering, graph representation). As a little bonus at the end, the xkcd-style plotting feature of matplotlib is shown. ;)

*Important note: This tutorial is a **BETA** - it may contain bugs and other errors!*