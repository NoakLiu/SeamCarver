from seam_carving import SeamCarver

import os

def image_resize_without_mask(filename_input, filename_output, new_height, new_width):
    obj = SeamCarver(filename_input, new_height, new_width)
    obj.save_result(filename_output)


def image_resize_with_mask(filename_input, filename_output, new_height, new_width, filename_mask):
    obj = SeamCarver(filename_input, new_height, new_width, protect_mask=filename_mask)
    obj.save_result(filename_output)


def object_removal(filename_input, filename_output, filename_mask):
    obj = SeamCarver(filename_input, 0, 0, object_mask=filename_mask)
    obj.save_result(filename_output)

if __name__ == '__main__':
    """
    Put image in in/images folder and protect or object mask in in/masks folder
    Ouput image will be saved to out/images folder with filename_output
    """

    folder_in = '../data/in'
    folder_out = '../data/out'

    filename_input = 'image.jpg'
    filename_output = 'image_result4.png'
    new_height = 300
    new_width = 600


    print("it is dual gradient energy method")
    input_image = os.path.join(folder_in, "images", filename_input)
    output_image = os.path.join(folder_out, "dual_gradient_energy", filename_output)

    image_resize_without_mask(input_image, output_image, new_height, new_width)
