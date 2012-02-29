# Option #1

_______

### Overview


### Attempt 1
![Left](https://github.com/KnownSubset/CSE559-Project2/raw/master/left.jpg "Left")
![Right](https://github.com/KnownSubset/CSE559-Project2/raw/master/right.jpg "Right")
![Mosaic](https://github.com/KnownSubset/CSE559-Project2/raw/master/left-right1.jpg "Planar Mosaic")

![Sift-Left](https://github.com/KnownSubset/CSE559-Project2/raw/master/sift-left.jpg "Sift-Left")
![Sift-Right](https://github.com/KnownSubset/CSE559-Project2/raw/master/sift-right.jpg "Sift-Right")


1. Ransac

2. % of inliers

3. Observations

### Attempt 2
1. Ransac

2. % of inliers

3. Observations

_______

# Extension #3 -> Implement and describe some method for smoothly blending together images that are geometrically aligned, but which have different lighting or gain or contrast. 

_______

#Feathering blend
Encoding transparency
I(x,y) = (aR, aG, aB, a) 
Iblend = Ileft + Iright

What is the Optimal Window?
To avoid seams
window >= size of largest prominent feature
To avoid ghosting
window <= 2*size of smallest prominent feature


![Red](https://github.com/KnownSubset/CSE559-Project2/raw/master/WP_000288.jpg "Red")
![White](https://github.com/KnownSubset/CSE559-Project2/raw/master/WP_000291.jpg "White")
![Blended](https://github.com/KnownSubset/CSE559-Project2/raw/master/feathered.jpg "Sharp Blend")
![Wider Window](https://github.com/KnownSubset/CSE559-Project2/raw/master/feathered-2.jpg "<< Sharp Blend")
![Even Widered Blend Window](https://github.com/KnownSubset/CSE559-Project2/raw/master/feathered-1.jpg "<<< Sharp Blend")

_______

#Lapcian blend

General Approach:
Build Laplacian pyramids LA and LB from images A and B
Build a Gaussian pyramid GR from selected region R
Form a combined pyramid LS from LA and LB using nodes of GR as weights:
LS(i,j) = GR(I,j,)*LA(I,j) + (1-GR(I,j))*LB(I,j)
Collapse the LS pyramid to get the final blended image


![Wider Window](https://github.com/KnownSubset/CSE559-Project2/raw/master/blend-failure.jpg "<< Sharp Blend")
![Even Widered Blend Window](https://github.com/KnownSubset/CSE559-Project2/raw/master/blended-candles1.jpg "<<< Sharp Blend")



