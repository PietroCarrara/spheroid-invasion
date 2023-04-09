import os
import cv2
import csv
import sys
import numpy as np
from utils import findJoinKill, contrast, isCircle, cropToRegion, contourDistance, Logger, EmptyLogger
from pathlib import Path

# Number of contours to use to compose the final invasion zone
FINAL_SEGMENTS = 1
# How many iterations to run with the find/join/kill algorithm
GROUPING_ITERS = 2

stats = []

for fname in sys.argv[1:]:
  INPUT = fname
  Path("out").mkdir(exist_ok=True)
  OUTPUT = "out/" + os.path.basename(fname)

  #############
  # MAIN CODE #
  #############

  # Enable/Disable logging
  # logger = Logger()
  logger = EmptyLogger()

  orig = cv2.imread(INPUT)
  HEIGHT, WIDTH = orig.shape[:2]
  logger.log(orig, "input")

  img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
  img = contrast(img)
  logger.log(img, "enhancing contrast")

  img = 255 - img
  logger.log(img, "inverting colors")

  # Kill 95% of the least bright pixels
  _, top95 = cv2.threshold(img, np.percentile(img, 95), 255, cv2.THRESH_BINARY)
  logger.log(top95, "thresholding")

  # Find the largest contour that is circular
  contours, _ = cv2.findContours(top95, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
  contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True) # Sort contours by area
  sphere = next((c for c in contours if isCircle(c)), contours[0]) # Get the first that is circular
  logger.log(cv2.drawContours(np.zeros_like(top95), contours, -1, 255, 3), "contour detection")
  logger.log(cv2.drawContours(np.zeros_like(top95), [sphere], -1, 255, 3), "largest circular contour")

  # Zoom in on the sphere
  region, offset = cropToRegion(sphere, img, 3)
  invOffset = (-offset[0], -offset[1])
  R_HEIGHT, R_WIDTH = region.shape[:2]
  logger.log(region, "zooming into spheroid")

  # Remove the spheroid and find other bright elements (hopefully, the invasion "rays"/"tentacles")
  cv2.fillPoly(region, [sphere], np.median(region), offset=offset)
  region = contrast(region)
  _, region = cv2.threshold(region, np.percentile(region, 95), 255, cv2.THRESH_BINARY)
  cv2.fillPoly(region, [sphere], 255, offset=offset)
  logger.log(region, "thresholding before detecting contour of rays")

  # Find the largest contours that are not too far from the spheroid
  joined = findJoinKill(region, logger=logger, iters=GROUPING_ITERS)
  joined = sorted(joined, key=lambda c: cv2.contourArea(c)/max(1, contourDistance(c, sphere, invOffset))**2, reverse=True)
  invasions = joined[:FINAL_SEGMENTS]

  cv2.drawContours(orig, [sphere], -1, (0, 0, 255), 3)
  cv2.drawContours(orig, joined[:FINAL_SEGMENTS], -1, (255, 0, 0), 3, offset=invOffset)
  cv2.imwrite(OUTPUT, orig)
  logger.log(orig, "output")


  # Count only invasion area
  spheroidMask = np.zeros_like(img)
  cv2.fillPoly(spheroidMask, [sphere], 255)
  spheroidArea = cv2.countNonZero(spheroidMask)

  invasionMask = np.zeros_like(img)
  cv2.fillPoly(invasionMask, joined[:FINAL_SEGMENTS], 255, offset=invOffset)
  cv2.fillPoly(invasionMask, [sphere], 0)
  invasionArea = cv2.countNonZero(invasionMask)

  stats.append([
    INPUT,
    spheroidArea+invasionArea, # Total area
    spheroidArea,              # Spheroid area
    invasionArea,              # Invasion area
    invasionArea/spheroidArea, # Invasion percentage
  ])

with open('stats.csv', 'w+') as f:
  w = csv.writer(f)
  w.writerow(['filename', 'Total area (in pixels)', 'Shperoid area (in pixels)', 'Invasion Area (in pixels)', 'Invasion %'])
  for s in stats:
    w.writerow(s)
