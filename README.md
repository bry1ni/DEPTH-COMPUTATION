# DEPTH-COMPUTATION
## Depth computation using simple stereo

#### Whats simple stereo ?

as human being, our two eyes, right and left, afford us to compute depth, and be able to know how much any object is far from our vision, simple stereo is a computer vision technic that perform the same way human eye works but with cameras.

#### How it works ?

well to be able to know how far any object is from your camera, you need to calibate your camera first, then compute the depth by getting the ditance between the pinhole of the camera and a specific point of the object.

#### Whats camera calibration ?

camera calibration is a process to get the intrinsic parameters of your camera, by locating the feature points of an object with a known geometry ( rubik cube for example or chess board ) were every point's location is known for us. well know u are wondering how do we get this points ? the answer is SIFT DESCRIPTOR.
