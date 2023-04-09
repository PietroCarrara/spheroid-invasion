import os
import cv2
import numpy as np
from datetime import datetime

class Logger:
  def __init__(self) -> None:
    self.dir = f"log {datetime.now()}"
    os.mkdir(self.dir)
    self.count = 0
    self.contourImage = None

  def log(self, img, title=""):
    self.count += 1

    if title != "":
      title = f" - {title}"

    cv2.imwrite(f"{self.dir}/{self.count}{title}.png", img)

class EmptyLogger:
  def log(self, img, title):
    return

def findJoinKill(img, iters = 1, logger = EmptyLogger()):
  # Find contours
  for i in range(iters):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    # Join very close contours
    joinedImg = np.zeros_like(img)
    cv2.drawContours(joinedImg, contours, -1, 255, 3)
    logger.log(joinedImg, "contours detected")
    joinedImg = cv2.morphologyEx(joinedImg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)));
    logger.log(joinedImg, "morphological closing")
    joined, _ = cv2.findContours(joinedImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    logger.log(cv2.drawContours(np.zeros_like(joinedImg), joined, -1, 255, 3), "joined contours")

    # Kill 99% of the smallest contours
    joined = sorted(joined, key=lambda c: cv2.contourArea(c), reverse=True) # Sort by perimeter
    joined = joined[:int(len(joined)*0.1)]
    img = cv2.drawContours(np.zeros_like(joinedImg), joined, -1, 255, 3)
    logger.log(img, "contours after elimination step")
    img = cv2.fillPoly(np.zeros_like(joinedImg), joined, 255)
    logger.log(img, "contours filled-in")

  return joined

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

# Distance between two polygons
def contourDistance(a, b, aOffset = (0, 0)):
  # Make sure a is always smaller than b
  if len(a) > len(b):
    tmp = a
    a = b
    b = tmp
    aOffset = (-aOffset[0], -aOffset[1])

  point = (int(a[0, 0, 0])+aOffset[0], int(a[0, 0, 1])+aOffset[1])
  minDist = -cv2.pointPolygonTest(b, point, True)
  for p in a[1:]:
    point = (int(p[0, 0])+aOffset[0], int(p[0, 1])+aOffset[1])
    d = -cv2.pointPolygonTest(b, point, True)
    if d < minDist:
      d = minDist

  return minDist
