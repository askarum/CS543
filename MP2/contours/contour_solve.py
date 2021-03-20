import numpy as np
from scipy import signal
import cv2
from scipy.ndimage import gaussian_filter





def nms(mag, tau, thresh):
  h,w = mag.shape
  Z = np.zeros((h,w), dtype = np.float32)

  theta = tau*180/np.pi
  theta[theta<0] +=180

  for i in range(1,h-1):
    for j in range(1,w-1):
      if mag[i, j] < thresh:
        Z[i, j] = 0

      else:

        if theta[i,j] <= 22.5 or theta[i,j] > 157.5:
          r,q = mag[i, j-1], mag[i, j+1]

        elif 22.5<theta[i,j]<=67.5:
          r,q = mag[i-1,j+1], mag[i+1, j-1]

        elif 67.5<theta[i,j]<=112.5:
          r,q = mag[i-1,j], mag[i+1, j]

        elif 112.5<theta[i,j]<=157.5:
          r,q = mag[i-1, j-1], mag[i+1, j+1]

        if mag[i,j] < r or mag[i,j]<q:
          Z[i,j] = 0

        else:
          Z[i,j] = mag[i,j]

      

  return Z


def hys(img, ht, lt):
  h,w = img.shape
  img_c = img.copy()
  for i in range(1,h-1):
    for j in range(1, w-1):
      if img[i,j] > lt:
        window = img[i-1:i+2, j-1:j+2]
        if window.max() >= ht:
          img_c[i,j] = window.max()
        else:
          img_c[i,j] = 0
  
  return img_c

def compute_edges_dxdy(I):
  """Returns the norm of dx and dy as the edge response function."""
  I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

  # I = gaussian_filter(I, sigma = 1.4)
  
  I = I.astype(np.float32)/255.
  

  
  dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same')
  dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same')
  mag = np.hypot(dx, dy)

  # mag = (mag - mag.min())/(mag.max() - mag.min())
  

  # angle = np.arctan2(-dy, dx)

  # highThresh = mag.max()*.09
  # lowThresh = highThresh*0.1

  # mag = nms(mag, angle, lowThresh)

  # mag = (mag - mag.min())/(mag.max() - mag.min())

  # highThresh = mag.max()*.15
  # lowThresh = highThresh*0.1


  # mag[mag<lowThresh] = 0


  # mag = hys(mag, highThresh, lowThresh)

  # highThresh = mag.max()*.15
  # lowThresh = highThresh*0.1

  # mag[mag<highThresh] = 0
  
  mag/1.5
  
  mag = mag * 255.
  mag = np.clip(mag, 0, 255)
  
  mag = mag.astype(np.uint8)
  return mag
