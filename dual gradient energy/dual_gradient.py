import numpy as np
def cal_dual_der(matrix):
    m = len(matrix)
    n = len(matrix[0])
    dx = np.zeros([m,n])
    dy = np.zeros([m,n])
    energy = np.zeros([m,n])
    for i in range(2,m):
        for j in range(2,n):
            dx[i][j] = -3 * matrix[i][j] + 4 * matrix[i - 1][j] - matrix[i - 2][j]
            dy[i][j] = -3 * matrix[i][j] + 4 * matrix[i][j - 1] - matrix[i][j - 2]
    for i in range(0,2):
        for j in range(0,2):
            dx[i][j] = -3 * matrix[i][j] + 4 * matrix[i + 1][j] - matrix[i + 2][j]
            dy[i][j] = -3 * matrix[i][j] + 4 * matrix[i][j + 1] - matrix[i][j + 2]
    for i in range(0,2):
        for j in range(2,n-2):
            dx[i][j] = -3 * matrix[i][j] + 4 * matrix[i + 1][j] - matrix[i + 2][j]
            dy[i][j] = -3 * matrix[i][j] + 4 * matrix[i][j + 1] - matrix[i][j + 2]
    for j in range(0,2):
        for i in range(2,m-2):
            dx[i][j] = -3 * matrix[i][j] + 4 * matrix[i + 1][j] - matrix[i + 2][j]
            dy[i][j] = -3 * matrix[i][j] + 4 * matrix[i][j + 1] - matrix[i][j + 2]
    for i in range(0,2):
        for j in range(n-2,n):
            dx[i][j] = dx[i][n-2]
            dy[i][j] = dx[i][n-2]
    for i in range(m-2,m):
        for j in range(0,2):
            dx[i][j] = dx[m-2][j]
            dy[i][j] = dy[m-2][j]
    for i in range(0,m):
        for j in range(0,n):
            energy[i][j]=dx[i][j]**2+dy[i][j]**2
    return energy