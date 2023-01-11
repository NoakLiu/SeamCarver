# Enhanced-Seam-Carving-Through-Some-Aspects

Seam carving is a quite famous method for resize the image. Seam Carving is a technique proposed by Avidan and Shamir1 for content-aware resizing of images. By using
pixel removing, the algorithm removes paths of low energy pixels (seams) from top to bottom
or from left to right which are not so important for the understanding of the image content.However, there are several shortcomings for the seam carving because the simple energy
map designation, brief energy dp function, disreguard some special characteristics in the image etc. Specifically, I optimize the model from the following aspects: apply bicubic interpolation when enlarging the image, use the LC algorithm to maintain the main characteristics of the image, apply the cany line to augment the outline of the image , use the Hough transformation to detect the most the most significant line of the image and then augment the image, apply absolute energy function to improve the performance of the seam carving, use dual energy function to implement another version of forward energy, use image classification experiments based on CNN to evaluate the performance of the seam carving.
