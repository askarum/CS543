import numpy as np
from scipy import signal
import cv2
from scipy.ndimage import gaussian_filter
import scipy

def gaussian_kernel(size, sigma):
  size//=2
  x = np.arange(-size,size+1)
  z = np.exp(-x**2/(2*sigma**2))
  normal = 1/(np.sqrt(2*np.pi)*sigma)
  z = np.array([z*normal])

  c = z*z.T

  return c

def harris(Ixx, Iyy, Ixy, window, alpha, sigma):
  h,w = Ixx.shape
  window //=2
  response = np.zeros((h,w))
  kernel = gaussian_kernel(window*2, sigma)


  for x in range(window, h - window):
    for y in range(window, w - window):
      wxx = Ixx[x-window:x+window+1, y-window:y+window+1]*kernel
      wyy = Iyy[x-window:x+window+1, y-window:y+window+1]*kernel
      wxy = Ixy[x-window:x+window+1, y-window:y+window+1]*kernel

      Mxx = wxx.sum()
      Myy = wyy.sum()
      Mxy = wxy.sum()

      det = Mxx*Myy - Mxy**2
      trace = Mxx + Myy

      R = det - alpha*(trace**2)

      if R>0:
        response[x,y] = R

      # response[x,y] = R
  return response

# def harris(Ixx, Iyy, Ixy, window, alpha, sigma):
#   h,w = Ixx.shape
#   window //=2
#   response = np.zeros((h,w))
#   kernel = gaussian_kernel(window*2, sigma)

#   Ixx = np.c_[np.zeros((Ixx.shape[0], window)), Ixx, np.zeros((Ixx.shape[0], window))]
#   Ixx = np.r_[np.zeros((window, Ixx.shape[1])), Ixx, np.zeros((window, Ixx.shape[1]))]

#   Iyy = np.c_[np.zeros((Iyy.shape[0], window)), Iyy, np.zeros((Iyy.shape[0], window))]
#   Iyy = np.r_[np.zeros((window, Iyy.shape[1])), Iyy, np.zeros((window, Iyy.shape[1]))]

#   Ixy = np.c_[np.zeros((Ixy.shape[0], window)), Ixy, np.zeros((Ixy.shape[0], window))]
#   Ixy = np.r_[np.zeros((window, Ixy.shape[1])), Ixy, np.zeros((window, Ixy.shape[1]))]

#   for x in range(window, h - window):
#     for y in range(window, w - window):
#       wxx = Ixx[x-window:x+window+1, y-window:y+window+1]*kernel
#       wyy = Iyy[x-window:x+window+1, y-window:y+window+1]*kernel
#       wxy = Ixy[x-window:x+window+1, y-window:y+window+1]*kernel

#       Mxx = wxx.sum()
      # Myy = wyy.sum()
      # Mxy = wxy.sum()

      # det = Mxx*Myy - Mxy**2
      # trace = Mxx + Myy

#       R = det - alpha*(trace**2)

#       if R>0:
#         response[x-window,y-window] = R

#       # response[x,y] = R
#   return response


def nms(img,window):
  h,w = img.shape
  imgc = img.copy()
  window //=2

  for x in range(window,h-window):
    for y in range(window, w - window):
      if img[x,y] != img[x-window:x+window+1, y-window:y+window+1].max():
        imgc[x,y] = 0
  
  return imgc

# def nms(img,window):
#   h,w = img.shape
#   imgc = img.copy()

#   window //=2
#   imgc = np.c_[np.zeros((imgc.shape[0], window)), imgc, np.zeros((imgc.shape[0], window))]
#   imgc = np.r_[np.zeros((window, imgc.shape[1])), imgc, np.zeros((window, imgc.shape[1]))]

#   imgc2 = imgc.copy()
  

#   for x in range(window,h-window):
#     for y in range(window, w - window):
#       if imgc[x,y] != imgc[x-window:x+window+1, y-window:y+window+1].max():
#         imgc2[x,y] = 0
  
#   return imgc2[window:-window, window:-window]

def compute_corners(I):

  I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

  I = I.astype(np.float32)/255

  dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same')
  dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same')

  Ixx, Iyy, Ixy = dx**2, dy**2, dx*dy

  response = harris(Ixx,Iyy,Ixy, 10, 0.06, .7)

  response = response/response.max()

  response[response<.007] = 0


  corners = nms(response, 7)

  h = corners.max()*.01
  corners[corners > h] = 1


  corners = corners * 255.
  corners = np.clip(corners, 0, 255)
  corners = corners.astype(np.uint8)
  
  response = response * 255.
  response = np.clip(response, 0, 255)
  response = response.astype(np.uint8)
  
  return response, corners
