# Advanced Content

### Code Optimization

Examples for how to speed up your code, relevant for anything that handles relatively large amounts of data (image analysis, data analysis, modeling, ...). There are scripts exemplifying three strategies:

- **Vectorization**

	- Using the edge finder from the main script as an example, this demonstrates the drastic 		 	   increase in speed that can often be achieved if operations are vectorized.
	
- **Multiprocessing**

	- This shows how Python's multiprocessing module can be used to simultaneously run the batch 	  		pipeline from the main tutorial on several images.
	
- **Cluster Processing**

	- An example of how to use one Python script to run another script multiple times with different 	   input data. If run locally, this is very similar to multiprocessing, but with a bit of 	 		  knowledge about high-performance cluster computing (see appropriate courses), this approach can 		be used to handle job submission and result collection on a computer cluster.

It should be noted that one of the key aspects of code optimization is finding out *which part* of the code costs the most time and could be optimized for the greatest gain in speed. This is called *profiling* and there are a number of options for how to do it, both in the form of Python modules as well as built into IDEs like Spyder. Profiling is not discussed here, but as a simple example the time module is used to test how long the different versions take to run.

### Image/Cell Classification

> **To be added when we have better images (which actually contain features that can be meaningfully classified/clustered)!**