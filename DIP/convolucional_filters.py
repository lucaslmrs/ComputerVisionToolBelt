import numpy as np
import cv2

def convolucional_filter(img, kernel):
    img = cv2.filter2D(img, -1, kernel)
    return img

def laplacian_kernel():
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    return kernel

def laplacian_kernel_2():
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    return kernel

def laplacian_kernel_horizontal():
    kernel = np.array([[-1, -1, -1],
                       [2, 2, 2],
                       [-1, -1, -1]])
    return kernel

def laplacian_kernel_vertical():
    kernel = np.array([[-1, 2, -1],
                       [-1, 2, -1],
                       [-1, 2, -1]])
    return kernel

def laplacian_kernel_diagonal():
    # laplacian kernel focused on identify 45 degrees edges
    kernel = np.array([[2, -1, -1],
                       [-1, 2, -1],
                       [-1, -1, 2]])
    return kernel

def laplacian_kernel_diagonal_2():
    # laplacian kernel focused on identify -45 degrees edges
    kernel = np.array([[-1, -1, 2],
                       [-1, 2, -1],
                       [2, -1, -1]])
    return kernel

def gaussian_kernel(size, sigma):
    kernel = cv2.getGaussianKernel(size, sigma)
    return kernel

def median_filter(ndarray, size):
    img = ndarray.copy()
    img = cv2.medianBlur(img, size)
    return img

def bilateral_filter(ndarray, size, sigma_color, sigma_space):
    img = ndarray.copy()
    img = cv2.bilateralFilter(img, size, sigma_color, sigma_space)
    return img


    




























