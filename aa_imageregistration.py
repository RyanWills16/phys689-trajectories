"""
:file: aa_imageregistration.py
:author: clayton pruitt
:date: 4/2/2020
:desc: Uses the astroalign package (https://arxiv.org/abs/1909.02946) to align sky images.
"""


import astroalign as aa
import errno
import numpy as np
import os, os.path
import sys

from datetime import datetime
from PIL import Image


##################################
# GLOBALS
##################################


# valid extensions for images
valid_extensions = [".jpg", ".png"]

# extension for outputted images
output_extension = ".png"

# valid file names (w/o extension) for target images
valid_target_names = ["reference", "target"]


##################################
# METHODS
##################################


"""
Helper method for main(). Returns a file_path and all files in the path.
@param argv: arguments from command line
@return: 2-tuple containing the file_path and a list of all files in the path
"""
def __parse_input(argv):
    # get directory
    if (len(argv) == 1):
        file_path = input("directory > ")
    else:
        file_path = argv[1]

    # clean input
    if (file_path[:2] != "./"):
        file_path = "./" + file_path

    if (file_path[-1] != "/"):
        file_path = file_path + "/"

    # get file path
    try:
        files_in_path = os.listdir(file_path)
    except FileNotFoundError:
        print("Invalid directory given.")
        print("Terminating program.")
        sys.exit()

    return file_path, files_in_path


"""
Helper method for main(). Checks if a file is valid by checking its extension.
@param file: file to check if valid
@return: True if valid; False otherwise
"""
def __file_is_valid(file):
    # get extension
    file_extension = os.path.splitext(file)[1].lower()
    # check if valid
    if file_extension in valid_extensions:
        return True
    return False


"""
Helper method for main(). Checks if a file is the target image for registration.
@param file: file to check if target
@return: True if target; False otherwise
"""
def __file_is_target(file):
    # get file name
    file_name = os.path.splitext(file)[0].lower()
    # check if target
    if file_name in valid_target_names:
        return True
    return False


"""
Helper method for __open_image_into_array(). Opens an image from the given file.
@param file: path to an image file with a valid extension
@param grayscale: option parameter to open the image in grayscale format
@return: PIL Image object
"""
def __open_image(file, grayscale = False):
    if grayscale:
        return Image.open(file).convert('L')
    return Image.open(file)
    

"""
Helper method for __open_image_into_array(). Converts the given image into a numpy array.
@param image: PIL Image object
@return: numpy array
"""
def __convert_image_to_numpy_array(image):
    return np.asarray(image)


"""
Helper method for main(). Opens an image from the given file into a numpy array.
@param file: path to an image file with a valid extension
@return: numpy array containing image data
"""
def __open_image_into_array(file):

    # open image as PIL image
    image = __open_image(file, grayscale = True)

    # convert image to numpy array
    np_array = __convert_image_to_numpy_array(image)

    # return
    return np_array


"""
Transforms the source to the target.
@param source: a numpy 2D array of the source image
@param target: a numpy 2D array of the target image
@return: a numpy 2D array of the transformed source
"""
def __transform_image(source, target):
    return aa.register(source, target)[0]
    

"""
Helper method for main. Saves an image with the given file name the given directory.
@param image: numpy array representing an image
@param directory: directory to save image to; defaults to current working directory
"""
def __save_image(image, filename, directory = os.getcwd()):

    # if directory does not exist, make it exist
    filepath = directory + "/" + filename
    os.makedirs(os.path.dirname(filepath), exist_ok = True)

    # convert image back to RGB before saving for better compatibility
    opened_image = Image.fromarray(image)
    if opened_image.mode != 'RGB':
        opened_image = opened_image.convert('RGB')

    # output the image
    opened_image.save(directory + "/" + filename)
    

"""
Main method.
"""
def main():

    # get all images from a directory given by sys args
    
    file_path, files_in_path = __parse_input(sys.argv)

    target = None # target image
    sources = [] # images to transform relative to the target image

    for file in files_in_path:
        print("Reading " + file)
        # only open valid files
        if __file_is_valid(file):
            print("Valid.")
            # open the file
            image = __open_image_into_array(file_path + file)
            # determine if file is a target image or source image
            if __file_is_target(file):
                target = image
            else:
                sources.append(image)

    # register all non-target images in relation to the target
    transformed_images = []

    for source in sources:
        transformed_image = __transform_image(source, target)
        transformed_images.append(transformed_image)
                
    # save the registered images
    output_folder = datetime.now().strftime("output_%Y%m%d%H%M%S")
    output_directory = "./" + output_folder
    for i, image in zip(range(len(transformed_images)), transformed_images):
        filename = "registered" +  str(i) + output_extension
        __save_image(image, filename, output_directory)
        print("Outputted image to " + output_directory + "/" + filename)
        
        
if __name__ == "__main__":
    main()
