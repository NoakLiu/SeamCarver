import numpy as np
import time
import cv2

def diag_sym_matrix(k=256):
    base_matrix = np.zeros((k,k))
    base_line = np.array(range(k))
    base_matrix[0] = base_line
    for i in range(1,k):
        base_matrix[i] = np.roll(base_line,i)
    base_matrix_triu = np.triu(base_matrix)
    return base_matrix_triu + base_matrix_triu.T

def cal_dist(hist):
    Diag_sym = diag_sym_matrix(k=256)
    hist_reshape = hist.reshape(1,-1)
    hist_reshape = np.tile(hist_reshape, (256, 1))
    return np.sum(Diag_sym*hist_reshape,axis=1)

def LC_f(image_gray):
    image_height,image_width = image_gray.shape[:2]
    hist_array = cv2.calcHist([image_gray], [0], None, [256], [0.0, 256.0])
    gray_dist = cal_dist(hist_array)

    image_gray_value = image_gray.reshape(1,-1)[0]
    image_gray_copy = [(lambda x: gray_dist[x]) (x)  for x in image_gray_value]
    image_gray_copy = np.array(image_gray_copy).reshape(image_height,image_width)
    image_gray_copy = (image_gray_copy-np.min(image_gray_copy))/(np.max(image_gray_copy)-np.min(image_gray_copy))
    return image_gray_copy


if __name__ == '__main__':
    file = '../data/in/images/image.jpg'

    start = time.time()
    image_gray = cv2.imread(file)[:,:,2].astype(np.uint8)
    print(image_gray.dtype)
    print(image_gray.shape)
    print(image_gray)
    saliency_image = LC_f(image_gray)
    print(saliency_image.shape)
    cv2.imwrite("resultt.png",saliency_image*255)
    end = time.time()

    print("Duration: %.2f seconds." % (end - start))
    cv2.imshow("gray saliency image", saliency_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
