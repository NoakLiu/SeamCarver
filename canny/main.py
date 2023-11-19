from seam_carving import SeamCarver

import os

def image_resize_without_mask(filename_input, filename_output, new_height, new_width):
    obj = SeamCarver(filename_input, new_height, new_width)
    obj.save_result(filename_output)

if __name__ == '__main__':
    """
    Put image in in/images folder and protect or object mask in in/masks folder
    Ouput image will be saved to out/images folder with filename_output
    """

    folder_in = '../data/in'
    folder_out = '../data/out'

    filename_input = 'image.jpg'
    filename_output = 'image_result.png'
    new_height = 300
    new_width = 600

    print("it is cany method")

    input_image = os.path.join(folder_in, "images", filename_input)
    output_image = os.path.join(folder_out, "cany", filename_output)

    image_resize_without_mask(input_image, output_image, new_height, new_width)
