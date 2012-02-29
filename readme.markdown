1. Option #1

_______

2. Extension #3
Implement and describe some method for smoothly blending together images that are geometrically aligned, but which have different lighting or gain or contrast. 

_______

#Distance blend


_______

#Lapcian blend
What is the Optimal Window?
To avoid seams
window >= size of largest prominent feature
To avoid ghosting
window <= 2*size of smallest prominent feature

General Approach:
Build Laplacian pyramids LA and LB from images A and B
Build a Gaussian pyramid GR from selected region R
Form a combined pyramid LS from LA and LB using nodes of GR as weights:
LS(i,j) = GR(I,j,)*LA(I,j) + (1-GR(I,j))*LB(I,j)
Collapse the LS pyramid to get the final blended image


