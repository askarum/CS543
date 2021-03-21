import numpy as np
from scipy import signal
import cv2
from scipy.ndimage import gaussian_filter
import scipy
from scipy.ndimage import maximum_filter


def harris(Ixx, Iyy, Ixy, window, alpha, sigma):
  h,w = Ixx.shape
  window //=2
  response = np.zeros((h,w))
  kernel = cv2.getGaussianKernel(window*2, sigma)
  kernel = kernel*kernel.T

  Mxx = signal.convolve2d(Ixx, kernel, mode = 'same', boundary = 'symm')
  Mxy = signal.convolve2d(Ixy, kernel, mode = 'same', boundary = 'symm')
  Myy = signal.convolve2d(Iyy, kernel, mode = 'same', boundary = 'symm')

  for i in range(h):
    for j in range(w):
      
      det = Mxx[i,j]*Myy[i,j] - Mxy[i,j]**2
      trace = Mxx[i,j] + Myy[i,j]

      R = det - alpha*(trace**2)

      if R>0:
        response[i,j] = R

      # response[x,y] = R
  return response


def nms(img,window):
  imgc = img.copy()
  h,w = img.shape
  m = maximum_filter(img, size = (window, window))

  for i in range(h):
    for j in range(w):
      if imgc[i,j]!=m[i,j]:
        imgc[i,j] = 0
  
  return imgc



def compute_corners(I):

  I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

  kernel = cv2.getGaussianKernel(5, 2)
  kernel = kernel*kernel.T

  
  I = I.astype(np.float32)/255

  # I = I + signal.convolve2d(I, kernel, mode= 'same', boundary = 'symm')

  dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same', boundary='symm')
  dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same', boundary = 'symm')

  # dx = dx-dx.mean()
  # dy = dy-dy.mean()

  Ixx, Iyy, Ixy = dx**2, dy**2, dx*dy

  response = harris(Ixx,Iyy,Ixy, 5, 0.05, 1.6)

  response = (response - response.min())/(response.max() - response.min())

  # response[response<.01] = 0


  corners = nms(response, 5)

  # h = corners.max()*.01
  # corners[corners > h] = 1


  corners = corners * 255.
  corners = np.clip(corners, 0, 255)
  corners = corners.astype(np.uint8)
  
  response = response * 255.
  response = np.clip(response, 0, 255)
  response = response.astype(np.uint8)
  
  return response, corners
