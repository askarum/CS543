import numpy as np
from scipy.ndimage import gaussian_filter


def blend(im1, im2, mask):
  mask = mask / 255.
  im1 = im1/255.
  im2 = im2/255.

  ga,gb,gm,la,lb, lr = [], [], [], [], [], []

  ga.append(im1)
  gb.append(im2)
  gm.append(mask)

  n = 10
  for i in range(n):
    ga.append(gaussian_filter(ga[-1], 2**i))
    gb.append(gaussian_filter(gb[-1], 2**i))
    gm.append(gaussian_filter(gm[-1], 2**i))
    la.append(ga[-2] - ga[-1])
    lb.append(gb[-2] - gb[-1])
    lr.append(la[-1]*gm[-2] + lb[-1]*(1-gm[-2]))

  g = ga[-1]*gm[-1] + gb[-1]*(1-gm[-1])
  out = sum(lr) + g

  return out*255
