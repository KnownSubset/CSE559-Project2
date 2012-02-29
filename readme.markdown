# Option #1

_______

### Overview

The following puesdocode is what I used to generate the homography that mapped two images geometeries together.  
I cannot really supply any good puesdocode for merging the images together, as I had a lot of difficulty with this aspect and had manually play around with the images until I was able to get it to work.

	```matlab
    %Image1 & Image2 are already available
    
	%%SIFT Feature discovery
	[F_1,D_1] = vl_sift(im2single(Im1));
	[F_2,D_2] = vl_sift(im2single(Im2));

	%Match keypoints (most similar features, compared to 2nd most similar)
	%by calculating the eucliandian distance between the every two point and take the closest  pair
	[matches, scores] = score_matches(D_1,D_2);

	%Compute Homography for a while
	for ih = 1:40
	    %Build up matrix of unknowns and known points, from 4 random matched points 
	    for ip = 1:4
	        A = cat(1, A, [-p1_X -p1_Y -1 0 0 0 p1_X*p2_X p1_Y*p2_X]);
	        A = cat(1, A, [ 0 0 0 -p1_X -p1_Y -1 p1_X*p2_Y p1_Y*p2_Y]);
	        b = cat(1, b, [-p2_X; -p2_Y]);	     
	    end
	    
	    %Solve for Homography
	    V = A\b;
	    
	    %Rearrage to 3x3 matrix and store
	    homographies(:,:,ih) = [V(1), V(2), V(3); V(4), V(5), V(6); V(7), V(8), 1];
	    X2_ = homographies(:,:,ih) * X1;
    	
    	%Calculate the number of points that once projected onto image are within a close proximity
	    du = X2_(1,:)./X2_(3,:) - X2(1,:)./X2(3,:) ;
	    dv = X2_(2,:)./X2_(3,:) - X2(2,:)./X2(3,:) ;
	    ok{ih} = (du.*du + dv.*dv) < PROXIMITY_LIMIT ;
	    
	    % score homography
	    hscores(ih) = sum(ok{ih}) ;
	end
	
	%Declare Success
    %Map images onto a surface
    ```
    
### Attempt 1
For my first attempt I used the two images from the slides as a way to help ensure that I was on the correct path to solving the problem.
![Left](https://github.com/KnownSubset/CSE559-Project2/raw/master/left.jpg "Left")
![Right](https://github.com/KnownSubset/CSE559-Project2/raw/master/right.jpg "Right")
![Mosaic](https://github.com/KnownSubset/CSE559-Project2/raw/master/left-right1.jpg "Planar Mosaic")

Here are the two images marked with the features that the _"interesting"_ features that vl_sift was able to discover.
![Sift-Left](https://github.com/KnownSubset/CSE559-Project2/raw/master/sift-left.jpg "Sift-Left")
![Sift-Right](https://github.com/KnownSubset/CSE559-Project2/raw/master/sift-right.jpg "Sift-Right")

Here are the two images with the matching points correlated aftering determing the best matches.
![Matches-Left](https://github.com/KnownSubset/CSE559-Project2/raw/master/matches-left.jpg "matches-Left")
![Matches-Right](https://github.com/KnownSubset/CSE559-Project2/raw/master/matches-right.jpg "matches-Right")



1. Ransac
 The algorithm was able to produce reasonable results from matching points.  

2. % of inliers

 76 points out of 118 matches were able to matched between the two images

3. Observations

### Attempt 2
1. Ransac

2. % of inliers
 82 points of 133.  This was unexpected to me, as the two images were vastly similar I was expecting the percentage of inliers to be much higher for the best homography.  This could be an indicator that I am a homography that is sufficient instead of correctly choosing the best possible homography.  I will have to revisit this at a later date for further investigation.
3. Observations

### Attempt 2 (Failure Example)
1. Ransac
It preformed as expected, since I expected Ransac would not be able to solve for these two images.  There was no overlap between the two images even though they shared some same characteristics, like the wood door frame and the proximity of a walls' corner.

![Failure-1](https://github.com/KnownSubset/CSE559-Project2/raw/master/WP_000292.jpg "failure-1")
![Failure-2](https://github.com/KnownSubset/CSE559-Project2/raw/master/WP_000293.jpg "failure-2")
2. % of inliers
0% were inliers, since 0 matches were produced.  If you examine the images below you will see that the points deemed interesting by vl_feat did not coorespond in both images.

![Failure-Interesting-1](https://github.com/KnownSubset/CSE559-Project2/raw/master/Failure-Pts1.jpg "failure-Interesting-1")
![Failure-Interesting-2](https://github.com/KnownSubset/CSE559-Project2/raw/master/Failure-Pts2.jpg "failure-Interesting-2")
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



