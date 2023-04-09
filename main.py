import cv2
import numpy as np
from utils import findJoinKill, contrast, isCircle, cropToRegion, contourDistance, Logger

# Number of contours to use to compose the final invasion zone
FINAL_SEGMENTS = 10

# FILE = 'ATP_ARC0023.png'
# FILE = 'ATP_RGD0035.png'
FILE = 'ATP_RGD0040.png'
# FILE = 'ATP0011.tif'
# FILE = 'siControl0047.png'

INPUT = 'test-data/'+FILE
OUTPUT = 'out/'+FILE

#############
# MAIN CODE #
#############

logger = Logger()

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
logger.log(cv2.drawContours(top95.copy(), contours, -1, 127, 3), "contour detection")
logger.log(cv2.drawContours(top95.copy(), [sphere], -1, 127, 3), "largest circular contour")

# Zoom in on the sphere
region, offset = cropToRegion(sphere, img, 3)
invOffset = (-offset[0], -offset[1])
R_HEIGHT, R_WIDTH = region.shape[:2]
logger.log(region, "cropping")

# Remove the spheroid and find other bright elements (hopefully, the invasion "rays"/"tentacles")
cv2.fillPoly(region, [sphere], np.median(region), offset=offset)
region = contrast(region)
_, region = cv2.threshold(region, np.percentile(region, 95), 255, cv2.THRESH_BINARY)
logger.log(region, "thresholding")

# Find the largest contours that are not too far from the spheroid
joined = findJoinKill(region, logger=logger, iters=2)
joined = sorted(joined, key=lambda c: cv2.contourArea(c)/max(1, contourDistance(c, sphere, invOffset))**2, reverse=True)

cv2.drawContours(orig, [sphere], -1, (0, 0, 255), 3)
cv2.drawContours(orig, joined[:FINAL_SEGMENTS], -1, (255, 0, 0), 3, offset=invOffset)
cv2.imwrite(OUTPUT, orig)
logger.log(orig, "output")