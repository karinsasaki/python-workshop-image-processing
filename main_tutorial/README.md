## README on Main Tutorial

### DESCRIPTION
This is a tutorial to exemplify fundamental concepts of automated image processing and segmentation, using python.

This course assumes a basic knowledge of the Python Programming Language. For those at EMBL, this means that you have participated in *a* beginners course for programming, preferably a Python course.


### TASK
Segmentation of 2D spinning-disc confocal fluorescence microscopy images of a membrane marker in Zebrafish early embryonic cells.


### REQUIREMENTS
- Python 2.7 (we recommend the Anaconda distribution, which includes most of the required modules)
- Modules: NumPy, SciPy, scikit-image, tifffile
- A text/code editor (we recommend Spyder-Py2)


### HOW TO FOLLOW THIS TUTORIAL
- Files you should have:
	- `tutorial_pipeline.py`; the tutorial Python script
	- `tutorial_pipeline_batch.py`; the tutorial introducing batch processing
	- The corresponding solutions and if desired partial solutions
	- `example_cells_1.tif` and `example_cells_2.tif`; these images are dual-color spinning-disc confocal confocal micrographs (40x) of two membrane-localised proteins during zebrafish early embryonic  development (~10hpf). They have 2 colors and are 930 by 780 pixels.

- The tutorial is self explanatory, indicating the steps to be taken next, what you should aim at achieving after carrying out those steps and the commands that could be used (there is not a one correct answer; these are suggestions).

- With this exercise we want to encourage you to become a more independent programmer, so if there is a command you don't quite know how to use, make sure you read the documentation. To do so, most of the time it's enough to google for "python" and the name of the module or function you want to use.

- If you are following this tutorial in class, if you have any questions, raise your hand and someone will come to help you. Otherwise, feel free to send your query to one of these two email addresses:
*jonas.hartmann@embl.de*
*karin.sasaki@embl.de*

### IMAGE PROCESSING CONCEPTS DISCUSSED IN THIS TUTORIAL
- Loading and visualising images
- Images are arrays of numbers; they can be indexed, sliced, etc...
- Images contain 3 types of information: Intensity, Shape, Size (a good segmentation pipeline uses them all)
- Preprocessing: smoothing, background substraction
- Segmentation: adaptive thresholding, distance transformation, detection of maxima, watershed
- Filtering: Discarding undesired objects, e.g. cells at the image boundaries
- Analysis: Extracting measurements from segmentation
- Saving output (segmentation, data, graphs)
- Automation: How to apply a pipeline to all files in a directory (batch processing)

### PROGRAMMING CONCEPTS DISCUSSED/TRAINED IN THIS TUTORIAL
- Python scripts, functions
- Common variable types: numpy array, dictionaries
- Control flow
- Modules, packages, importing modules and packages and using them
- Importing data
- Using the documentation
- Arrays and manipulation (dimensions, indexing, slicing, arithmetic)
- Visualising images
- Debugging by printing relevant information and plotting images at appropriate stages
- Exporting data and writing files
- Good practice

