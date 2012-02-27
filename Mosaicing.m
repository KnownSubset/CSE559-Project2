run('vlfeat-0.9.14/toolbox/vl_setup');
imBlur = imfilter(imread('/Users/nathan/Development/CSE559-Project2/left1.png'),fspecial('Gaussian',[5 5],1));
LeftImage = im2double(imBlur)/255;
LeftImage = rgb2gray(LeftImage);
imBlur = imfilter(imread('/Users/nathan/Development/CSE559-Project2/right1.png'),fspecial('Gaussian',[5 5],1));
RightImage = im2double(imBlur)/255;
RightImage = rgb2gray(RightImage);
[F_Left,D_Left] = vl_sift(im2single(LeftImage));
[F_Right,D_Right] = vl_sift(im2single(RightImage));

figure(1), imagesc(LeftImage), colormap gray, axis off, axis image;
hold on;
perm = randperm(size(F_Left,2)) ; 
sel = perm(1:50) ;
h1 = vl_plotframe(F_Left(:,sel)) ; 
h2 = vl_plotframe(F_Left(:,sel)) ; 
set(h1,'color','k','linewidth',3) ;
set(h2,'color','y','linewidth',2) ;

figure(2), imagesc(RightImage), colormap gray, axis off, axis image;
hold on;
perm = randperm(size(F_Right,2)) ; 
sel = perm(1:50) ;
h1 = vl_plotframe(F_Right(:,sel)) ; 
h2 = vl_plotframe(F_Right(:,sel)) ; 
set(h1,'color','k','linewidth',3) ;
set(h2,'color','y','linewidth',2) ;

[vectorLength,descriptorCount] = size(D_Left);

similarity = zeros(descriptorCount,descriptorCount);
maxSimilarity = zeros(1,descriptorCount);
for ix = 1:descriptorCount;
    for iy = 1:vectorLength;
        similarity(ix,iy) = similarity(ix,iy) + abs(D_Left(iy) - D_Right(iy));
    end;
    maxSimilarity(ix) = max(similarity(ix));
end