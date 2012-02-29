# Option #1

_______

### Overview

Ransac is a great technique for determining how to map multiple images together by generating a homography (Hp = p').  However since there are mutliple possible homographies for two images there is some work involved too determine the best homography.  We are able to use brute force to help generate a reasonable guess of the _"correct"_ homography.  I was able to find a few cases in which Ransac was not successful, or generated false positives.

For projecting on the same surface, I stuck with a planar mapping.  If I had some more advance knowledge, I would have attempted the cylinderical mapping.  As the cylinderical mapping would help to alleviate some of the distortion effects evident in the planar mapping, such as the sharp layering.

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
 The algorithm was able to produce reasonable results from matching points.  The best homography that was produced was able to map images coordinates systems together.

2. Percentage of inliers

 76 points out of 118 matches were able to matched between the two images.  As you can see from the images below, the inliers that were found map fairly well between the two images.  Some of the outliers were features that were in the reflection of the original image from the lake.   It would have been interesting to mirror the image horizontially and see how the images would have mapped between each other.  Other outliers were the faces of the various cliffs that were excluded once the best homography found that those two points would not line up.
 ![Inliers-1](https://github.com/KnownSubset/CSE559-Project2/raw/master/inliers1.jpg "inliers-1")

3. Observations

### Attempt 2

![Red](https://github.com/KnownSubset/CSE559-Project2/raw/master/WP_000288.jpg "Red")
![White](https://github.com/KnownSubset/CSE559-Project2/raw/master/WP_000291.jpg "White")

1. Ransac
 These two images were abled to be aligned, but for the wrong reason due to the vast similarity between the two images.  The homography that was produced by Ransac assumes that the one of the images should be mirror horizontially and laid almost directly on top of the other image.
 The causes for this could be that I was so sleep deprived while taking the pictures that the angles of camera were different enough by Ransac's computations that I was mirroring the image.

![Red](https://github.com/KnownSubset/CSE559-Project2/raw/master/red-pts.jpg "Red")
![White](https://github.com/KnownSubset/CSE559-Project2/raw/master/white-pts.jpg "White")

Here are the interesting points that were found by the vl_sift.

2. % of inliers
 
 This was unexpected to me, as the two images were vastly similar I was expecting the percentage of inliers to be much higher for the best homography.  This could be an indicator that I am a homography that is sufficient instead of correctly choosing the best possible homography.  I will have to revisit this at a later date for further investigation.
 When I was debugging my code and using the vl_ubcmatch fuction packaged with vl_feat, the percentage of inliers was much higher.   82 points of 133.  I don't know if I should be proud of or disheartened that my percentage of inliers lower considering the two images were greatly similiar but the homography said that they should be flipped.
 ![Inliers-2](https://github.com/KnownSubset/CSE559-Project2/raw/master/inliers2.jpg "inliers-2")
 
 Another point of investigation could be to reshoot the images using a tripod and view the results.

3. Observations

### Attempt 3 (Failure Example)
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

Feathering is a method that gradually fades two images together to help hide the differences.  Part of the method invovles determining the optimal window to apply the feathering within.

*What is the Optimal Window?
**To avoid seams
***window >= size of largest prominent feature
**To avoid ghosting
***window <= 2*size of smallest prominent feature

    ```matlab
    width = size(Image1,2);
    mask1 = zeros(size(Image1));
    window = max(size(largest_feature)) - min(size(smallest_feature));
    mask1(:,1:width/2 + window/2) = 1;
    mask2 = mask1 - 1;

    gaussian = fspecial('gauss',30,sharpnessFactor);
    mask1 = imfilter(mask1,gaussian,'replicate'); %Blur mask
    mask2 = imfilter(mask2,gaussian,'replicate'); %Blur mask
    featheredImage = mask1.*mosaic1+mask2.*mosaic2;
    ```


I got some clues on how to implement feathering from [this slide deck](http://www.seas.upenn.edu/~cse399b/Lectures/CSE399b-11-Blending.ppt) I wasn't able to determine the author, but they were adapted from Alexei Efros @ CMU.


![Red](https://github.com/KnownSubset/CSE559-Project2/raw/master/WP_000288.jpg "Red")
![White](https://github.com/KnownSubset/CSE559-Project2/raw/master/WP_000291.jpg "White")
![Blended](https://github.com/KnownSubset/CSE559-Project2/raw/master/feathered.jpg "Sharp Blend")
![Wider Window](https://github.com/KnownSubset/CSE559-Project2/raw/master/feathered-2.jpg "<< Sharp Blend")
![Even Widered Blend Window](https://github.com/KnownSubset/CSE559-Project2/raw/master/feathered-1.jpg "<<< Sharp Blend")

_______

#Laplacian blend

General Approach:

1.Build Laplacian pyramids LA and LB from images A and B
2.Build a Gaussian pyramid GR from selected region R
3.Form a combined pyramid LS from LA and LB using nodes of GR as weights:
4.LS(i,j) = GR(I,j,)*LA(I,j) + (1-GR(I,j))*LB(I,j)
5.Collapse the LS pyramid to get the final blended image

I gathered the method from [the same slide deck](http://www.seas.upenn.edu/~cse399b/Lectures/CSE399b-11-Blending.ppt)

Or as we can see in picuters 
1. Generate Laplacian from Guassians

![Guassians](http://pages.cs.wisc.edu/~csverma/CS766_09/ImageMosaic/figure1.jpg "Generate Lapicains from Guassians")

2. Combine Laplacians at each level

![Laplacian](http://pages.cs.wisc.edu/~csverma/CS766_09/ImageMosaic/figure2.jpg "Laplacian")

3. Reconstruct image from Laplacians

![Reconstruct](http://pages.cs.wisc.edu/~csverma/CS766_09/ImageMosaic/figure3.jpg "")

These images were taken from Chaman Singh Verma's  and Mon-Ju's page on [image blending](http://pages.cs.wisc.edu/~csverma/CS766_09/ImageMosaic/imagemosaic.html)

I attempted to calculate the window in which the blending would need to occur instead of attempting on the entire image as the math requires the matrixes to be of the same size.  I did not know of a reasonable way to get this working with cropping the images to be the same size.  When attempting this on the region of the images that would need to be blended, the results were quite disasterious.

![Lapcian Blend results](https://github.com/KnownSubset/CSE559-Project2/raw/master/blend-failure.jpg "Lapcian Blend")

![Lapcian Blend failure](https://github.com/KnownSubset/CSE559-Project2/raw/master/pyramid.jpg "Lapcian Blend failure")


As I found out with a little insight the correct way to use Laplacian blend would have to basically apply the feathering techniques on the entire image that was generated through Laplacian pyramids.

    ```matlab
    width = size(Image1,2);
    mask = zeros(size(Image1));
    mask(:,1:middle_of_blend_region) = 1;
    Image = Image * mask;

    for level = 1:6;
        imG = imfilter(Image,fspecial('Gaussian',[5 5],1));
        imL = Image - imG;  % Laplacian
        Image = imresize(Image,0.5);
        laplacians{level} = imL;
        gaussians{level} = imG;
    end
    %Reconstruct Image
    for level = 6:2;
       limgo{p-1} = limgo{p-1} + impyramid(limgo{p}, 'expand');
       laplacians{p-1} = laplacians{p-1} + impyramid(laplacians{p}, 'expand');
    end

    %Do same for right side

    %Then combine images
    Mosaic = LeftImage + RightImage

    ```

