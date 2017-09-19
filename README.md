Python Workshop - Image Processing
===================================

**Please note that a new and improved version of the materials for this course is available [here](https://github.com/WhoIsJack/python-workshop-image-processing)!**


## Course Aims and Overview

This course teaches the basics of bio-image processing, segmentation and analysis in python. It is based on tutorials that integrate explanations and exercises, enabling participants to build their own image analysis pipeline step by step.

The `main_tutorial` uses single-cell segmentation of a confocal fluorescence microscopy image to illustrate key concepts from preprocessing to segmentation to data analysis. It includes a tutorial on how to apply such a pipeline to multiple images at once (batch processing).

The main tutorial is complemented by the `pre-tutorial`, which reviews some basic python concepts using as an example a rat fibroblast image, and by the `optional_advanced_content`, which features further examples and tutorials on the topics of vectorization, multiprocessing, cluster computation and advanced data analysis.

This course is aimed at people with basic to intermediate knowledge of python and basic knowledge of microscopy. For people with basic knowledge of image processing, the tutorials can be followed without attending the lectures.


## Instructions on following this course

- If you have only very basic knowledge of python or if you are feeling a little rusty, you should start with the `pre-tutorial`, which includes three tutorials: one on numpy arrays, one on python functions and one on the basics of interacting with image data in Python. If you are more experienced, you may want to skim or skip the pre-tutorial.
Note: The pre-tutorial is organized as an iPython notebook.

- In the `main_tutorial`, it is recommended to follow the `tutorial_pipeline` first. By following the exercises, you should be able to implement your own segmentation pipeline. If you run into trouble, you can use the provided solutions as inspiration - however, it is *highly* recommended to spend a lot of time figuring things out yourself, as this is an important part of any programming exercise. If you are having a lot of trouble, you may want to use the `partial_solutions`, which give you some help yet still demand that you think about it yourself. After completing the segmentation pipeline, you can follow the `tutorial_pipeline_batch` to learn how to run your program for several images and collect all the results.
Note: The main tutorial is organized simply as comments in an empty script. It is up to you to fill in the appropriate code.

- Finally, the `advanced_content` contains an introductory example to three important techniques for making making your scripts faster and operating on large datasets, namely *vectorization*, *multiprocessing* and *cluster processing*. The `data_analysis` tutorial (currently in *BETA*!) is an introduction to piping segmentation results into more advanced statistical data analysis, including *feature extraction*, *PCA*, *clustering* and *graph-based analysis*.


## Concepts discussed in course lectures

1. **Basic Python (KS)**
	* Importing packages and modules
	* Reading files
	* Data and variable types
	* Importing data
	* Storing data in variables
	* Defining and using functions
	* Arrays, indexing, slicing
	* Control flow
	* Plotting images
	* Debugging by printing
	* Output formatting and writing files
	* Using the documentation


2. **Basics of BioImage Processing (KM)**
	* Images as numbers
		* Bit/colour depth
		* Colour maps and look up tables 
	* Definition of Bio-image Analysis
		* Image Analysis definition for signal processing science 
		* Image Analysis definition for biology
		* Algorithms and Workflows
		* Typical workflows in biology
	* Convolution and Filtering
		* Why do we do filtering?
		* Convolution in 1D, 2D and 3D 
	* Pre-segmentation filtering
		* De-noising
		* Smoothing
		* Unsharp mask
	* Post-segmentation filtering
		* Tuning segmented structures
		* Mathematical morphology, erosion, dilation
			* Distance map 
			* Watershed

3. **Introduction to the Tutorial Pipeline (JH)**
	* Automated Single-Cell Segmentation
		* Why? (Advantages of single-cell approaches)
		* How? (Standard segmentation pipeline build)
			* Preprocessing (smoothing, background subtraction)
			* Presegmentation (thresholding, seed detection)
			* Segmentation (seed expansion; e.g. watershed)
			* Postprocessing (removing artefacts, refining segmentation)
			* Quantification and analysis
		* What? (for the main tutorial: 2D spinning disc confocal fluorescence microscopy images of Zebrafish embryonic cells)
		* Who? (YOU!)

3. **Advanced material**
	* CellProfiler to automate image analysis workflows and python plugin module **(VH)**
	* Code Optimisation (vectorisation, multiprocessing, cluster processing) & advanced data analysis **(JH)**

		
## Instructors

- Karin Sasaki
    - EMBL Centre for Biological Modelling
    - Organiser of course, practical materials preparation, tutor, TA
- Jonas Hartmann
    - Gilmour Lab, CBB, EMBL
    - Pipeline developer, practical materials preparation, tutor, TA
- Kora Miura
    - EMBL Centre for Molecular and Cellular Imaging
    - Tutor
- Volker Hilsenstein
    - Scientific officer at the ALMF
    - Tutor, TA (image processing)
- Toby Hodges
    - Bio-IT, EMBL
    - TA (python)
- Aliaksandr Halavatyi
    - Postdoc at the Pepperkik group
    - TA (programming/image processing)
- Imre Gaspar
    - Staff scientists at the Ephrussi group
    - TA (programming/image processing)


## Feedback 

We welcome any feedback on this course! 

Feel free to contact us at *karin.sasaki@embl.de* or *jonas.hartmann@embl.de*.
