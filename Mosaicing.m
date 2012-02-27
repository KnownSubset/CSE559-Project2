run('vlfeat-0.9.14/toolbox/vl_setup');
imBlur = imfilter(imread('/Users/nathan/Development/CSE559-Project2/left1.png'),fspecial('Gaussian',[5 5],1));
LeftImage = im2double(imresize(imBlur,[400,525]))/255;
LeftImage = rgb2gray(LeftImage);
imBlur = imfilter(imread('/Users/nathan/Development/CSE559-Project2/right1.png'),fspecial('Gaussian',[5 5],1));
RightImage = im2double(imresize(imBlur,[400,525]))/255;
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
[vectorLength2,descriptorCount2] = size(D_Right);

similarity = zeros(descriptorCount,descriptorCount2);
maxSimilarity = zeros(1,descriptorCount);
for ix = 1:descriptorCount;
    dx = sum(D_Left(:,ix));
    for iy = 1:descriptorCount2;
        dy = sum(D_Right(:,iy));
        similarity(ix,iy) = similarity(ix,iy) + abs(dx - dy);
    end
    [rte, index] = min(similarity(ix,:));
    maxSimilarity(1,ix) = index;
end

[matches,scores] = vl_ubcmatch(D_Left, D_Right);

hscores = zeros(1,1:40);
for ih = 1:40
    points = randi(size(matches,2),1,4);
    A = [];
    b = [];
    for ip = 1:4
        point1 = matches(1,ip);
        point2 = matches(2,ip);
        p1_X = F_Left(1,point1);
        p1_Y = F_Left(2,point1);
        p2_X = F_Right(1,point2);
        p2_Y = F_Right(2,point2);
        A = cat(1, A, [-p1_X -p1_Y  1     0     0  0 -p1_X*p2_X -p1_Y*p2_Y]);
        A = cat(1, A, [    0     0  0 -p1_X -p1_Y  1 -p1_X*p2_X -p1_Y*p2_Y]);
        b = cat(1, b,[-p2_X; -p2_Y]);
     
    end    
    H = A/b;
    
    tLeft = H.*LeftImage;
    
    hscores(ih) = 0;
end