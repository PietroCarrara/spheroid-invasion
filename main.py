import cv2
import numpy as np

# FILE = 'test-data/ATP_ARC0023.png'
# FILE = 'test-data/ATP0011.tif'
# FILE = 'test-data/siControl0047.png'
FILE = 'test-data/ATP_RGD0040.png'

# Contrast streching (enhances contrast)
def contrast(img):
  low = img.min()
  high = img.max()
  scale = high - low
  return np.uint8(((img - low)/scale)*255)

# Tests if a contour is circle-like
def isCircle(contour):
  _, r = cv2.minEnclosingCircle(contour)
  circleness = cv2.contourArea(contour) / (np.pi*r**2)
  return circleness >= 0.35 # Cover at least 35% of the circle's pixels to be a circle-like object

# Zoom in on a contour. Width and height are multiplied by the scale, affecting the zoom
def cropToRegion(contour, img, scale = 1):
  maxH, maxW = img.shape[:2]
  (x, y, w, h) = cv2.boundingRect(contour)

  # Zoom out according to scale but keep the center unchanged
  cx = x + w/2
  cy = y + h/2
  w *= scale
  h *= scale
  x = cx - w/2
  y = cy - h/2

  # Make sure we're still inside the image
  x = max(int(x), 0)
  y = max(int(y), 0)
  w = min(int(w), maxW)
  h = min(int(h), maxH)

  return img[y:y+h, x:x+w], (-x, -y)

orig = cv2.imread(FILE)

img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
img = contrast(img)

# Invert colors
img = 255 - img

# Kill 95% of the least bright pixels
_, top95 = cv2.threshold(img, np.percentile(img, 95), 255, cv2.THRESH_BINARY)

# Find the largest contour that is circular
contours, hierarchy = cv2.findContours(top95, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True) # Sort contours by area
sphere = next((c for c in contours if isCircle(c)), contours[0]) # Get the first that is circular

# Zoom in on the sphere
zoomImg, offset = cropToRegion(sphere, orig, 3)

cv2.imwrite('out.png', cv2.drawContours(zoomImg, [sphere], -1, (255, 0, 255), 3, offset=offset))