# Enhanced-Seam-Carving-Through-Some-Aspects

Seam carving is a quite famous method for resize the image. By using pixel removing, the basic idea of this algorithm is to remove the pixel curves with the lowest energy (seams) from top to bottom or from left to right to realize image resizing horizontally or vertically.

However, there are several shortcomings for the seam carving because the simple energy dynamic programming function on graph pixel modeling, it will miss some special characteristics in the image etc. 

I enhance the performance of the model from the following aspects: 

1. apply bicubic interpolation when enlarging the image
2. use the LC algorithm to maintain the main characteristics of the image
3. apply the cany line to augment the outline of the image 
4. use the Hough transformation to detect the most the most significant line of the image and then augment the image
5. apply absolute energy function to improve the performance of the seam carving
6. use dual energy function to implement another version of forward energy
7. use image classification experiments based on CNN to evaluate the performance of the seam carving.
