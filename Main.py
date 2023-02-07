import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

print(cv.__version__)

img = cv.imread('Wisconsin Autonomous/red.png')

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)


lower_red = np.array([0, 0, 170])
upper_red = np.array([100, 100, 255])

mask = cv.inRange(img, lower_red, upper_red)

# Merge channels
# im_thresh = cv.merge((im_thresh, im_thresh, im_thresh))
# Remove small noise
im_thick = cv.medianBlur(mask, 13)
# Connect components
# im_thick = cv.erode(im_thick, np.ones((31, 31)))
# Draw a white border around shape to avoid errors in blob finding
# cv.rectangle(im_thick, (0, 0), img.shape[:2], (255, 255, 255), 10)

# To find each blob and size of each
im_out = img.copy()
im_thick = ~cv.split(im_thick)[0]
cnts, _ = cv.findContours(im_thick, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cnts = list(cnts)


cnts.sort(
    key=lambda p: max(cv.boundingRect(p)[2], cv.boundingRect(p)[3]), reverse=True
)
for cnt in cnts:
    peri = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
    x, y, w, h = cv.boundingRect(approx)
    cv.rectangle(im_out, (x, y), (x + w, y + h), (0, 255, 0), 5)
    print('new one found')
    print(type(cnt))



# mask = cv.inRange(hsv, lower_blue, upper_blue)

# k = np.array(([0, 1, 0],
#              [1, -4, 1],
#              [0, 1, 0]), np.float32)
#
#
#

#
# output = cv.filter2D(result, -1, k)

# plt.imshow(result)
# plt.title('Plot with matplot')

# plt.imshow(output)
# plt.title('Result')
#
# plt.waitforbuttonpress(0)
#

result = cv.bitwise_and(img, img, mask=mask)
# cv.imshow('frame', img)
cv.imshow('mask', mask)
# cv.imshow('result', result)
cv.imshow('blob', im_thick)
# cv.imshow('out', im_out)

cv.waitKey(0)
cv.destroyAllWindows()
