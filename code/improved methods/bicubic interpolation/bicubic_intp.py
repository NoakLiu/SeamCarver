import numpy as np
def S(x):
    x = np.abs(x)
    if 0 <= x < 1:
        return 1 - 2 * x * x + x * x * x
    if 1 <= x < 2:
        return 4 - 8 * x + 5 * x * x - x * x * x
    else:
        return 0

def adjust(value):
    if value > 255:
        value = 255
    elif value < 0:
        value = 0
    return value

def doublecubic(img,i,j,ch):
    height, width, channels = img.shape
    sh , sw = 1.0, 1.0
    #if(dir == 1):
    m = height+1
    n = width
    sh = 1.0 + 1.0 / height
    #else:
    #    m = height
    #    n = width+1
    #    sw = 1.0 + 1.0 / width
    x=i/sh
    y=j/sw

    p = (i + 0.0) / sh - x  # 非常得注意加和方式
    q = (j + 0.0) / sw - y
    x = int(x) - 2
    y = int(y) - 2
    A = np.array([
        [S(1 + p), S(p), S(1 - p), S(2 - p)]
    ])
    if x >= 1 and x <= (m - 3) and y >= 1 and y <= (n - 3):
        B = np.array([
            [img[x - 1, y - 1], img[x - 1, y],
             img[x - 1, y + 1],
             img[x - 1, y + 1]],
            [img[x, y - 1], img[x, y],
             img[x, y + 1], img[x, y + 2]],
            [img[x + 1, y - 1], img[x + 1, y],
             img[x + 1, y + 1], img[x + 1, y + 2]],
            [img[x + 2, y - 1], img[x + 2, y],
             img[x + 2, y + 1], img[x + 2, y + 1]],

        ])
        C = np.array([[S(1 + q)], [S(q)], [S(1 - q)], [S(2 - q)]])
        blue = np.dot(np.dot(A, B[:, :, 0]), C)[0, 0]
        green = np.dot(np.dot(A, B[:, :, 1]), C)[0, 0]
        red = np.dot(np.dot(A, B[:, :, 2]), C)[0, 0]

        blue = adjust(blue)
        green = adjust(green)
        red = adjust(red)
        return np.array([blue, green, red], dtype=np.uint8)[ch]  # 填补某一个处的空缺
    else:
        return 0