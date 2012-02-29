# Option #1

_______

### Overview

	```matlab
    %Image1 & Image2 are already available
    
	%%SIFT Feature discovery
	[F_1,D_1] = vl_sift(im2single(Im1));
	[F_2,D_2] = vl_sift(im2single(Im2));

	%Match keypoints (most similar features, compared to 2nd most similar)
	[matches, scores] = score_matches(D_1,D_2);

	%Compute Homography
	for ih = 1:40
	    points = randi(size(matches,2),1,4);
    	A = [];
	    b = [];
	    %Build up matrix of unknowns and known points, from 4 random matched points 
	    for ip = 1:4
	        point1 = matches(1,points(ip));
	        point2 = matches(2,points(ip));
	        p1_X = F_1(1,point1);
	        p1_Y = F_1(2,point1);
	        p2_X = F_2(1,point2);
	        p2_Y = F_2(2,point2);
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
    
    ```
    
### Attempt 1
![Left](https://github.com/KnownSubset/CSE559-Project2/raw/master/left.jpg "Left")
![Right](https://github.com/KnownSubset/CSE559-Project2/raw/master/right.jpg "Right")
![Mosaic](https://github.com/KnownSubset/CSE559-Project2/raw/master/left-right1.jpg "Planar Mosaic")

Here are the two images marked with the features that the _"interesting"_ features that vl_sift was able to discover;
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



