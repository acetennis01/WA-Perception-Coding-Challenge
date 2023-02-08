import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Wisconsin Autonomous/red.png')


lower_red = np.array([0, 0, 170])
upper_red = np.array([100, 100, 255])

mask = cv.inRange(img, lower_red, upper_red)


# Remove small noise
im_thick = cv.medianBlur(mask, 17)
# Connect components

# To find each blob and size of each
im_out = img.copy()
im_thick = ~cv.split(im_thick)[0]
cnts, _ = cv.findContours(im_thick, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cnts = list(cnts)
# print(len(cnts))


cnts.sort(
    key=lambda p: max(cv.boundingRect(p)[2], cv.boundingRect(p)[3]), reverse=True
)

i = 0

del cnts[0]

points = []

for cnt in cnts:
    peri = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
    x, y, w, h = cv.boundingRect(approx)

    # cv.circle(im_out, ((int)(x + w/2), (int)(y + h/2)), 20, (i, i, i), 6)

    points.append(((int)(x + w/2), (int)(y + h/2)))

    i += 10

# for i in range(len(points)):
#    #print(i)
#    print(points[i])

points.sort()

# print(points)

# for i in range(len(points)):
#    print(i)
#    print(points[i])

left_cones = []
right_cones = []

for i in range(len(points) - 1):
    slope = (points[i + 1][1] - points[i][1])/(points[i + 1][0] - points[i][0])
    # slope = y2 - y1 / x2 - x1
    # cv.line(im_out, points[i], points[i + 1], (i * 10, i * 10, i * 10), 5)
    if (not (slope > -0.5 and slope < 0.5)):
        if (slope > 0):  # if slope is positive

            if not (points[i] in left_cones):
                left_cones.append(points[i])
            if not (points[i + 1] in left_cones):
                left_cones.append(points[i + 1])
        elif (slope < 0):  # if slope is negative
            if not (points[i] in right_cones):
                right_cones.append(points[i])
            if not (points[i + 1] in right_cones):
                right_cones.append(points[i + 1])

    # print(slope)

print(len(right_cones))
print(len(left_cones))

right_cones = np.array(right_cones)
left_cones = np.array(left_cones)

[vx, vy, x, y] = cv.fitLine(right_cones, cv.DIST_L2, 0, 0.01, 0.01)

h, w, channels = im_out.shape

cv.line(im_out, (int(x-vx*w), int(y-vy*w)),
        (int(x+vx*w), int(y+vy*w)), (0, 0, 255), 5)

[vx, vy, x, y] = cv.fitLine(left_cones, cv.DIST_L2, 0, 0.01, 0.01)

cv.line(im_out, (int(x-vx*w), int(y-vy*w)),
        (int(x+vx*w), int(y+vy*w)), (0, 0, 255), 5)

result = cv.bitwise_and(img, img, mask=mask)
# cv.imshow('frame', img)
# cv.imshow('mask', mask)
# cv.imshow('result', result)
# cv.imshow('blob', im_thick)

cv.imwrite('answer.png', im_out)

cv.imshow('out', im_out)

cv.waitKey(0)
cv.destroyAllWindows()
