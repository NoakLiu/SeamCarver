import numpy as np
# 霍夫变换实现检测图像中的前10条主要的直线
def Hough_Line(edge, img):
    ## Voting
    def voting(edge):
        H, W = edge.shape

        drho = 1
        dtheta = 1

        # get rho max length
        rho_max = np.ceil(np.sqrt(H ** 2 + W ** 2)).astype(np.int)

        # hough table
        hough = np.zeros((rho_max, 180), dtype=np.int)

        # get index of edge
        # ind[0] 是 符合条件的纵坐标，ind[1]是符合条件的横坐标
        ind = np.where(edge == 255)

        ## hough transformation
        # zip函数返回元组
        for y, x in zip(ind[0], ind[1]):
                for theta in range(0, 180, dtheta):
                        # get polar coordinat4s
                        t = np.pi / 180 * theta
                        rho = int(x * np.cos(t) + y * np.sin(t))

                        # vote
                        hough[rho, theta] += 1

        out = hough.astype(np.uint8)

        return out

    # non maximum suppression
    def non_maximum_suppression(hough):
        rho_max, _ = hough.shape

        ## non maximum suppression
        for y in range(rho_max):
            for x in range(180):
                # get 8 nearest neighbor
                x1 = max(x-1, 0)
                x2 = min(x+2, 180)
                y1 = max(y-1, 0)
                y2 = min(y+2, rho_max-1)
                if np.max(hough[y1:y2, x1:x2]) == hough[y,x] and hough[y, x] != 0:
                    pass
                    #hough[y,x] = 255
                else:
                    hough[y,x] = 0

        return hough

    def inverse_hough(hough, img):
        H, W, _= img.shape
        rho_max, _ = hough.shape

        out = img.copy()

        # get x, y index of hough table
        # np.ravel 将多维数组降为1维
        # argsort  将数组元素从小到大排序，返回索引
        # [::-1]   反序->从大到小
        # [:20]    前20个
        ind_x = np.argsort(hough.ravel())[::-1][:10]
        ind_y = ind_x.copy()
        thetas = ind_x % 180
        rhos = ind_y // 180

        # each theta and rho
        for theta, rho in zip(thetas, rhos):
            # theta[radian] -> angle[degree]
            t = np.pi / 180. * theta

            # hough -> (x,y)
            for x in range(W):
                if np.sin(t) != 0:
                    y = - (np.cos(t) / np.sin(t)) * x + (rho) / np.sin(t)
                    y = int(y)
                    if y >= H or y < 0:
                        continue
                    out[y, x] = [0,255,255]
            for y in range(H):
                if np.cos(t) != 0:
                    x = - (np.sin(t) / np.cos(t)) * y + (rho) / np.cos(t)
                    x = int(x)
                    if x >= W or x < 0:
                        continue
                    out[y, x] = [0,0,255]

        out = out.astype(np.uint8)

        return out


    # voting
    hough = voting(edge)

    # non maximum suppression
    hough = non_maximum_suppression(hough)

    # inverse hough
    out = inverse_hough(hough, img)

    return out
