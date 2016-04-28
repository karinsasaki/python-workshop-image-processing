# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 18:11:13 2016

@author: Karin Sasaki @ CBM @ EMBL Heidelberg
         Jonas Hartmann @ Gilmour Group @ EMBL Heidelberg

@requires:  Python 2.7
            NumPy 1.9, SciPy 0.15
            scikit-image 0.11.2, tifffile 0.3.1
            
@content:             
1. Images -> numbers -> images
   	       |      |
 	       V      V
	   data types color maps/look up tables
       
2. Basic python reminder with short pipeline

3. How to use the documentation

@references:
            http://www.tutorialspoint.com/python/ 
            https://github.com/tobyhodges/ITPP
            https://github.com/cmci/HTManalysisCourse/blob/master/CentreCourseProtocol.md#workflow-python-primer
            http://cmci.embl.de/documents/ijcourses

"""


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 1. Basics Images with Python
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
 
# In this section we learn the numerical nature of digital images.


#%%
#------------------------------------------------------------------------------
# IMPORT PACKAGES AND MODULES

# Modules in Python are simply Python files with the .py extension, which  implement a set of functions. The purpose of writting modules is to group  related code together, to make it easier to understand and use, sort of as a  'black box'.

# Packages are name spaces that contain multiple packages and modules themselves. You can think of them as 'directories'.

# To be able to use the functions in a particular package or module, you need  to 'import' that module or package. To do so, you need to use the import command. For example, to import the package numpy, which enables the manipulation of arrays, you would do:

import numpy as np                 # Array manipulation package
import matplotlib.pyplot as plt    # Plotting package
import skimage.io as io            # Image file manipulation module

#%%
#------------------------------------------------------------------------------
# VARIABLE TYPE, BIT-DEPTH and DATA TYPE

# --------------------------
# Variable type
# --------------------------

# Standard types of objects built-in python include (or variable types): 

# boolean
a = 1<2
type(a)

# numeric (int, float, complex), 
b = 5.1
type(b)

# sequence (list, tuple, range), 
c = ['karin', 'sasaki']
type(c)

# text (str), 
type(c[0])

# The type of a NumPy array is numpy.ndarray. 
d = np.array([1,2,3,4,5])
type(d)
# note this is different form 
e = [1,2,3,4,5]
type(e)


# --------------------------
# Binary numbers
# --------------------------

# Computers code information with binary numbers.

# A binary number is a value represented in the binary numerical system, which represents numeric values using two symbols, 1 and 0. (Decimal (denary) numbers are represented with symbols 0,...,9.)

# We are most used to thinking in terms of base 10 values, so we can ask: how many differnet decimal values can be represented in binary?

# Well, if you only have one position, then you can only represent 2 decimal numbers, or 2^1, namely 0 and 1.

# If you have 2 positions, then you have the following possible combinations of 1s and 0s:
# binary    decimal
# 00          0   
# 01          1               2^0
# 10          2         2^1
# 11          3         2^1 + 2^0

# So you can represent 4 numbers, or 2^2.

# If you have three positions, you can represent 2^3 = 8 decimal numbers.
# binary    decimal
# 000           0       
# 001           1               2^0
# 010           2           2^1
# 011           3           2^1+2^0
# 100           4       2^2
# 101           5       2^2+    2^0
# 110           6       2^2+2^1
# 111           7       2^2+2^1+2^0

# If you have n positions, then you can represent 2^n decimal numbers.

# --------------------------
# Bit (short for ``binary value'') and bytes
# --------------------------

# A bit is the smallest unit of data in a computer and it has a single binary value, either 0 or 1. Bytes are multiples of bits.

# If a number in a computer is stored in an 8-bit byte, then that number will be one of 2^8 (+1, including 0) = 256 values.
# For example:
# 8-bit			    decimal
# 00000000          0
# 00000001			1 = 2^0
# 00000010			2 = 2^1
# 00000011			3 = 2^1 + 2^0
# 00001111			15 = 2^3 + 2^2 + 2^1 + 2^0
# 11111111			225 = 2^8 + 2^7 + 2^6 + 2^5 + 2^4 + 2^3 + 2^2 + 2^1 + 2^0

# The number of bits in a byte is what is refered to as the ``depth''.

# Note that increasing bit-depth comes hand in hand with increasing memory use.

# To find the depth of your data, you need to find the data type, using the command dtype.


#%%
#------------------------------------------------------------------------------
# DIGITAL IMAGE IS A MATRIX OF NUMBERS, VARIABLE TYPE AND DATA TYPE

# A digital image we display on a computer screen is made up of pixels, which have specific values (intensities).

# When you import an image into python, using the function io.imread from the package skimage, the image is a variable of type np.array and of data type uint8. In particular, it is a collection of numbers, arranged in an ordered pattern.
img = io.imread('nuclei.png')        # import  -> check documentation!!
img                                  # print
type(img)                            # type
plt.imshow(img)                      # plot
plt.show()
img.dtype                            # data type
img.shape                            # dimensions of array, in termsl of pixels

# Each pixel has brightness, or intensity (or more strictly, amplitude) somewhere between black and white represented as a number. 

# We usually do not see these values, or numbers, in the image displayed on monitor, but we could access these numbers in the image file by converting the image file to a text file.

np.savetxt('nuclei.txt', img[:,:,0])   # save to txt file; sliced array

# On the other hand, you can take such a collection of numbers that are ordered in a parttern, as visualise as an image - the reason being that those numbers can be considered as pixel values.

#img2=np.random.rand(20,10) # random array
#np.savetxt('randimg.txt', img2) # save as text

dataimg = np.loadtxt("randimg.txt", dtype='uint8')   # load txt file -> check documentation!
plt.imshow(dataimg)                                  # visulaise as image
plt.show()



#

# We can see individual pixels by zooming up the image using the magnifying tool. 

# At the moment it does not look great as we can not see the individual pixels. To amend that, include the argument interpolation='none' when plotting, as follows:
plt.imshow(data, interpolation='none') 
plt.show()
 
# Width and height of the image are defined by the number of pixels in x and y directions. 
print img.shape     # print array shape/dimensions
print data.shape     # print array shape/dimensions

# Key concepts: 
# - images are arrays of numbers
# - variable type: np.array
# - data type: uint8
# - arrays have dimensions that can be sliced


#%%
#------------------------------------------------------------------------------
# IMAGE BIT-DEPTH

# --------------------------
# Grayness resolution
# --------------------------

# Now that we understand that: 
# 1) images are arrays of numbers that represent pixel intensity, 
# 2) that those intensities (the grayscale) range from black to white and 
# 3) that such objects are stored in the computer memory at a specific depth,
# we can apply the concept of bit depth to images.

# 1-bit means that the grayscale of the image has 2^1 = 2 steps, so each pixel can be one of only two colors, black (0) or white (1).

# 8-bit means that the gray-scale of the image has 2^8 = 256 steps: in other words, the grayness between black and white is divided into 256 steps. 

# In the same way 16-bit translates to 216 = 65536 steps, hence for 16-bit images one can assign gray-level to pixels in a much more precise way; in other words "grayness resolution" is higher.

# Without getting into the technicalities of how microscope images are generated, the sensor of the microscope takes in the light and converts it into electrical signals, which is then converted into voltage and then finally into pixel values, for the corresponding position within the image. 
# The signal intensity is continuous (e.g. recall that there are an infinitelly many (decimal) numbers between 0 and 1). 
# However, as you have seen above, the digitalisation process makes the data discrete (as is represented with integer values), so we loose the ability to encode infinite steps and, for example, 0.45 might be converted to 0, whilst 0.56 to 1.

# Larger bit-depth enables more detailed conversion (higher resolution) of signal intensity (continuous value) to pixel values (discrete).

# It is important to keep this in mind when choosing a bit depth. This choice is ultimately a balance amongst the imaging conditions and the type of analysis required. 


# --------------------------
# Image arithmetic and unexpected errors
# --------------------------

# A digital image is a matrix of integers so you can do array arithmetic on the as normal.

data_plot=plt.imshow(dataimg,cmap="gray")
plt.show()
data_plot=plt.imshow(dataimg+100,cmap="gray")
plt.show()

# values are calculated modulo 256
data_plot
data_plot + 100

# floats get rounded
data3 = data_plot / 3
data
data3


# So keep this in mind when working with digital images. (example of what can go wrong: when you subtract a background, but you don't have an appropriate threshold, you might get negative numbers, but in the machine, it might loop around and something that was supposed to be black becomes white).



# --------------------------
# RGB format and colour depth
# --------------------------

# Let's discuss briefly color images in RGB format.

# Every color in a digital image is made up of some combination of the three primary colors of light - red, green and blue. 

# If your data type is 1-bit, then the most you could create would be eight different colors, including white if you mixed all three together and black if all of them are absent.
# BGR
# 001 - red
# 010 - green
# 100 - blue
# 011 - yellow
# 101 - magneta
# 110 - cyan
# 111 - white
# 000 - black

# To achieve all the beautiful colors that you can see in most images now a days, it is necessary to use multiple shades of red, green and blue and then indicate which shade of red, green and blue to include. By choosing a bit depth bigger than 1, you effectivly get 2^bit shades of each color,  giving you a total of (2^bit)^3 different colors. 

# Note that a single RGB image thus has three different channels. In other words, three layers of different images are overlaid in a single RGB image. In general, each channel (layer) has a bit depth of 8-bit. So a single RGB image is 24-bit image. 


# --------------------------
# Color maps and look up tables (LUT) (or how the matrix of numbers is converted to an image)
# --------------------------

# visualise with different color map
plt.imshow(img[:,:,0],interpolation='none',cmap='gray')        
plt.show()
plt.imshow(img[:,:,0],interpolation='none',cmap='Blues')        
plt.show()

# For the moment, let's stick to and RGB in 8-bit depth format. A look up table is a table with indices, 0 to 255 and corresponding values (between 0 and 255) for each of R, G and B.

# extract the color contained in a specific colormap
from matplotlib import cm           

# define function that prints the LUT
def printLUT(colorMap):
    """
    Description: Prints the LUT
    Requirements: cm from matplotlib
    Input argument(s): colorMap=cm.mapName
    Example: printLUT(cm.gray)
    """
    for i in range(256):
        print i, colorMap(i)
        
printLUT(cm.gray)    
printLUT(cm.Blues)        
       
# For the correct shade of a pixel to be displayed, the LUT is employed as follows: the pixel value is read, is compared with the list of indices in the table and the corresponding shade values for each of R, G and B is allocated.

# Note, if you choose a LUT that is a grayscale, the three colors have the same values at the same positions. 

# You can change the default color map with the comands below:
# import matplotlib as mpl
# mpl.rcParams['image.cmap'] = 'gray'


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# 2. Short pipeline example
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


# These images show:
# This cells called RAT2 (rat fibroblasts); The aim of this labeling is 
# colocalization of two protein; checking if they locate in the same places.
# Green - p150glued ( motor protein; the biggest subunit of dynactin promotes 
# the initiation of dynein-driven cargo motility from the microtubule plus-end); 
# Red - AP2 beta subunit (Adaptor protein complexes function in protein 
# transport via transport vesicles in different membrane traffic pathways)
# Blue - DAPI (nucleus)

# Aim: Measure co-localisation of motor proteins and adaptor protein in rat fibroblasts.

# import multi-color image
img = io.imread('nuclei.png') 

# use the documentation to understand how to use this function. Google python skimage io imread doc

# variable type
print type(img)

# data type
print img.dtype    

# print array shape/dimensions
print img.shape  

# visualise imahe in python
plt.imshow(img, interpolate='none',cmap='gray')
plt.show() 

# I don't want to type all of this every time I want to plot, so I make a function. 
# A function is a block of organized, reusable code that is used to perform a single, related action. Functions provide better modularity for your application and a high degree of code reusing.
# As you already know, Python gives you many built-in functions like print(), etc. but you can also create your own functions. These functions are called user-defined functions.
# Remember that the syntax is as follows:
#def functionname( parameters ):
#   "function_docstring"
#   function_suite
#   return [expression]

def myplt(img):
    """This function displays the input image with interpolation='none' and cmap='gray'. """
    plt.imshow(img,interpolation='none',cmap='gray')
    plt.show()
    
# to get help on this function, you need to run the command help(myplt)    

# looking at each slice of dimension 4
for i in range(img.shape[-1]): 
    myplt(img[:,:,i])
    
# slice array to get each channel
adaptor = img[:,:,0]
motor = img[:,:,1]
nuclei = img[:,:,2]


# The fact that this is a co-localization experiment with green and red offers a simple "biologically relevant" question: As a very simple measure for co-localization, we could check how often pixels that are above average intensity in red are also above average intensity in green.


# get all pixels in red above average
mean_adaptor = np.mean(adaptor)   

adaptor[adaptor<2*mean_adaptor] = 0
adaptor[adaptor>=2*mean_adaptor] = 1
        
myplt(adaptor)

# get all pixels in green above average
mean_motor = np.mean(motor)   

motor[motor<2*mean_motor] = 0
motor[motor>=2*mean_motor] = 1
        
myplt(motor)


# get all colocalised pixels

    # need to make 0 values distinct
img.shape    
for i in range(img.shape[0]): 
    for j in range(img.shape[1]): 
        if adaptor[i,j] == 0:
            adaptor[i,j] = 3
        elif motor[i,j]==0:
            motor[i,j] = 2
            
myplt(adaptor)
myplt(motor)                        
    
    # if colocalised, allocate 1, else, 0    
colocalised = adaptor == motor

    # visualise
myplt(colocalised) 
        
    
# store results        
results = {'protein':[], 'intensity':[], 'number':[]}        
results['protein'] = ['AP2', 'p150glued']
results['intensity'] = [mean_adaptor, mean_motor]
results['number'] = [sum(sum(adaptor)), sum(sum(motor)), sum(sum(colocalised))]

print results


# save image to tiff file

    # Get filehandling function
from tifffile import imsave
    
    # save image
imsave("ext_nuc_AP2_beta_subunit.tif",adaptor,bigtiff=True) 


# save quantitative data to json file
import json
json.dump(results, open("results.txt", 'w'))



#%%
#------------------------------------------------------------------------------
# FUNCTIONS

# -----------------------
# Function
# -----------------------

# More on functions

# -----------------------
# Defining a Function
# -----------------------
# You can define functions to provide the required functionality. Here are simple rules to define a function in Python.

#   Function blocks begin with the keyword def followed by the function name and parentheses ( ( ) ).

#   Any input parameters or arguments should be placed within these parentheses. You can also define parameters inside these parentheses.

#   The first statement of a function can be an optional statement - the documentation string of the function or docstring.

#   The code block within every function starts with a colon (:) and is indented.

#   The statement return [expression] exits a function, optionally passing back an expression to the caller. A return statement with no arguments is the same as return None.

# -----------------------
# Syntax
# -----------------------
def functionname( parameters ):
   "function_docstring"
   function_suite
   return [expression]

# By default, parameters have a positional behavior and you need to inform them in the same order that they were defined.

# -----------------------
# Example
# -----------------------
# The following function takes a string as input parameter and prints it on standard screen.

def printme( str ):
   "This prints a passed string into this function"
   print str
   return
   
def addme( a, b ):
   "This adds passed arguments."
   print a+b
   return a+b
# or just return ??   
   
   

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------   
   
# -----------------------   
# Calling a Function
# -----------------------

# Defining a function only gives it a name, specifies the parameters that are to be included in the function and structures the blocks of code.

# Once the basic structure of a function is finalized, you can execute it by calling it from another function or directly from the Python prompt. Following is the example to call printme() function −  

printme("I'm first call to user defined function!")
printme("Again second call to the same function") 
addme(1,2)

# When the above code is executed, it produces the following result −

#I'm first call to user defined function!
#Again second call to the same function
#3


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

# -----------------------
# Function Arguments
# -----------------------

# You can call a function by using the following types of formal arguments:

# ************************
# Required arguments
# ************************
# Required arguments are the arguments passed to a function in correct positional order. Here, the number of arguments in the function call should match exactly with the function definition.

# To call the function printme(), you definitely need to pass one argument, otherwise it gives a syntax error.

#Traceback (most recent call last):
#  File "test.py", line 11, in <module>
#    printme();
#TypeError: printme() takes exactly 1 argument (0 given)

# Similarly, for addme() you need to pass two arguments.

# ************************
# Keyword arguments
# ************************
# Keyword arguments are related to the function calls. When you use keyword arguments in a function call, the caller identifies the arguments by the parameter name.

# This allows you to skip arguments or place them out of order because the Python interpreter is able to use the keywords provided to match the values with parameters. 

# For example, the following code
printme( str = "My string")

# The following example gives more clear picture. Note that the order of parameters does not matter.

# Function definition is here
def printinfo( name, age ):
   "This prints a passed info into this function"
   print "Name: ", name
   print "Age: ", age
   return

# Now you can call printinfo function
printinfo( age=50, name="miki" )

# When the above code is executed, it produces the following result −

#Name:  miki
#Age:  50

# ************************
# Default arguments
# ************************
#A default argument is an argument that assumes a default value if a value is not provided in the function call for that argument. The following example gives an idea on default arguments, it prints default age if it is not passed −

# Function definition is here
def printinfo( name, age = 35 ):
   "This prints a passed info into this function"
   print "Name: ", name
   print "Age ", age
   return;

# Now you can call printinfo function
printinfo( age=50, name="miki" )
printinfo( name="miki" )

# When the above code is executed, it produces the following result −

#Name:  miki
#Age:  50
#Name:  miki
#Age:  35

# ************************
# Variable-length arguments
# ************************
# You may need to process a function for more arguments than you specified while defining the function. These arguments are called variable-length arguments and are not named in the function definition, unlike required and default arguments.

# Syntax for a function with non-keyword variable arguments is this −

#def functionname([formal_args,] *var_args_tuple ):
#   "function_docstring"
#   function_suite
#   return [expression]

# An asterisk (*) is placed before the variable name that holds the values of all nonkeyword variable arguments. This tuple remains empty if no additional arguments are specified during the function call. Following is a simple example −

# Function definition is here
def printinfo( arg1, *vartuple ):
   "This prints a variable passed arguments"
   print "Output is: "
   print arg1
   for var in vartuple:
      print var
   return;

# Now you can call printinfo function
printinfo( 10 )
printinfo( 70, 60, 50 )

# When the above code is executed, it produces the following result −

#Output is:
#10
#Output is:
#70
#60
#50


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

# -----------------------
# The Anonymous Functions
# -----------------------
# These functions are called anonymous because they are not declared in the standard manner by using the def keyword. You can use the lambda keyword to create small anonymous functions.

# Lambda forms can take any number of arguments but return just one value in the form of an expression. They cannot contain commands or multiple expressions.

# An anonymous function cannot be a direct call to print because lambda requires an expression

# Lambda functions have their own local namespace and cannot access variables other than those in their parameter list and those in the global namespace.

# Although it appears that lambda's are a one-line version of a function, they are not equivalent to inline statements in C or C++, whose purpose is by passing function stack allocation during invocation for performance reasons.

# -----------------------
# Syntax
# -----------------------
# The syntax of lambda functions contains only a single statement, which is as follows −

#lambda [arg1 [,arg2,.....argn]]:expression

# Following is the example to show how lambda form of function works −

# Function definition is here
sum = lambda arg1, arg2: arg1 + arg2;

# Now you can call sum as a function
print "Value of total : ", sum( 10, 20 )
print "Value of total : ", sum( 20, 20 )

# When the above code is executed, it produces the following result −

#Value of total :  30
#Value of total :  40


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

# -----------------------
# The return Statement
# -----------------------
# The statement return [expression] exits a function, optionally passing back an expression to the caller. A return statement with no arguments is the same as return None.

# All the above examples are not returning any value. You can return a value from a function as follows −

# Function definition is here
def substractme( arg1, arg2 ):
   # Substracts the second parameter from the first and returns the result."
   total = arg1 - arg2
   return total;

# Now you can call sum function
total = substractme( 10, 20 );
print total 

# When the above code is executed, it produces the following result −
#30

# The retuned arguments are also order-specific. So if you have a function:
def arithmetic( a, b ):
    sumab = a+b
    substractab = a-b
    multiplyab = a*b
    return sumab, substractab, multiplyab
    
# This
c, d, e = arithmetic(1,2)
print c
print d
print e

# does not alocate the same values to the varialbes c, d, e as this
c, e, d = arithmetic(1,2)
print c
print d
print e

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

# -----------------------
# Scope of Variables
# -----------------------
# All variables in a program may not be accessible at all locations in that program. This depends on where you have declared a variable.

# The scope of a variable determines the portion of the program where you can access a particular identifier. There are two basic scopes of variables in Python −

# Global variables

# Local variables



# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

# -----------------------
# Global vs. Local variables
# -----------------------
# Variables that are defined inside a function body have a local scope, and those defined outside have a global scope.

# This means that local variables can be accessed only inside the function in which they are declared, whereas global variables can be accessed throughout the program body by all functions. When you call a function, the variables declared inside it are brought into scope. Following is a simple example −

total = 0; # This is global variable.

# Function definition is here
def substractme( arg1, arg2 ):
   # Substratcs the second parameter from the first and return them."
   total = arg1 - arg2; # Here total is a local variable.
   print "Inside the function (local) total: " total 
   return total;

# Now you can call sum function
substractme( 10, 20 );
print "Outside the function (global) total : ", total 

# When the above code is executed, it produces the following result −

#Inside the function local total :  -10
#Outside the function global total :  0


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

# -----------------------
# Python Modules
# -----------------------
# A module allows you to logically organise your Python code. Grouping related code into a module makes the code easier to understand and use. A module is a Python object with arbitrarily named attributes that you can bind and reference.

# Simply, a module is a file consisting of Python code. A module can define functions, classes and variables. A module can also include runnable code.

# For example, in a new file, copy and past the following, making sure you undertand the code. Name the file module_example.py:

#def print_func( par ):
#   print "Hello : ", par
#   return

# In this example, the module we are intersted in ix modelu_example.py.


# -----------------------
# The import Statement
# -----------------------
# You can use any Python source file as a module by executing an import statement in some other Python source file. The import has the following syntax:

#import module1[, module2[,... moduleN]

# When the interpreter encounters an import statement, it imports the module if the module is present in the search path. A search path is a list of directories that the interpreter searches before importing a module. For example, to import the module module_example.py, you need to put the following command at the top of the script −

# Import module support
import module_example as modex

# Now you can call defined functions of the module, as follows
modex.print_func("Zara")

# When the above code is executed, it produces the following result −
#Hello : Zara

# A module is loaded only once, regardless of the number of times it is imported. This prevents the module execution from happening over and over again if multiple imports occur.

# -----------------------
# The from...import Statement and The from...import * Statement:
# -----------------------

# Python's from statement lets you import specific attributes from a module into the current namespace. The from...import has the following syntax −

#from modname import name1[, name2[, ... nameN]]

# For example, to import the function fibonacci from the module fib, use the following statement −

from scipy import sum as s

# This statement does not import the entire module/package scipy into the current namespace; it just introduces the item sum from the module scipy into the global symbol table of the importing module, and renames it to s.

# It is also possible to import all names from a module into the current namespace by using the following import statement −

#from modname import *
