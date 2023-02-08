# Wisconsin Autonomous Perception Coding Challenge

## answer.png

![](https://github.com/acetennis01/WA-Perception-Coding-Challenge/blob/main/answer.png)

## Methodology

Before detectiing the cones, I had to preprocess the input image to only show the cones. I noticed that all the cones had the same color, so I filtered the image to only show those colors and applied other operations to take out the disturbances in the background. Next, I found the contours of the blobs and extracted the location of those blobs. I then used the slopes between the locations of the blobs to figure out wether the blob was a right cone or a left cone. After that, I used the line of best fit to plot the lines for the right and left cones.

## What not worked

Before I used the contour method, I tried to use a triangular kernel to figure out if the shape was a cone after the image was perprocessed. I tried for a bit but got nowhere so I decided to find another solution.

## Libraries used

OpenCV2
Numpy
