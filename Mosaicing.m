run('vlfeat-0.9.14/toolbox/vl_setup');
imBlur = imfilter(imread('/Users/nathan/Development/CSE559-Project2/left1.png'),fspecial('Gaussian',[5 5],1));
Image1 = im2double(imresize(imBlur,[400,525]))/255;
Im1 = rgb2gray(Image1);
imBlur = imfilter(imread('/Users/nathan/Development/CSE559-Project2/right1.png'),fspecial('Gaussian',[5 5],1));
Image2 = im2double(imresize(imBlur,[400,525]))/255;
Im2 = rgb2gray(Image2);
[F_1,D_1] = vl_sift(im2single(Im1));
[F_2,D_2] = vl_sift(im2single(Im2));

%figure(1), imagesc(Im1), colormap gray, axis off, axis image;
%hold on;
%perm = randperm(size(F_1,2)) ; 
%sel = perm(1:50) ;
%h1 = vl_plotframe(F_1(:,sel)) ; 
%h2 = vl_plotframe(F_1(:,sel)) ; 
%set(h1,'color','k','linewidth',3) ;
%set(h2,'color','y','linewidth',2) ;

%figure(2), imagesc(Im2), colormap gray, axis off, axis image;
%hold on;
%perm = randperm(size(F_2,2)) ; 
%sel = perm(1:50) ;
%h1 = vl_plotframe(F_2(:,sel)) ; 
%h2 = vl_plotframe(F_2(:,sel)) ; 
%set(h1,'color','k','linewidth',3) ;
%set(h2,'color','y','linewidth',2) ;

[vectorLength,descriptorCount] = size(D_1);
[vectorLength2,descriptorCount2] = size(D_2);

%use ubc for now, will replace once I have it working
[matches, scores] = vl_ubcmatch(D_1,D_2);

similarity = zeros(descriptorCount,descriptorCount2);
maxSimilarity = zeros(1,descriptorCount);
for ix = 1:descriptorCount;
    dx = sum(D_1(:,ix));
    for iy = 1:descriptorCount2;
        dy = sum(D_2(:,iy));
        similarity(ix,iy) = similarity(ix,iy) + abs(dx - dy);
    end
    [rte, index] = min(similarity(ix,:));
    maxSimilarity(1,ix) = index;
end

hscores = zeros(1,1:40);
% score homography
X1 = cat(1,F_1(1:2,matches(1,:)),ones(1,size(matches,2)));
X2 = cat(1,F_2(1:2,matches(2,:)),ones(1,size(matches,2)));
homographies = zeros(3,3,40);
for ih = 1:40
    points = randi(size(matches,2),1,4);
    A = [];
    b = [];
    for ip = 1:4
        point1 = matches(1,points(ip));
        point2 = matches(2,points(ip));
        p1_X = F_1(1,point1);
        p1_Y = F_1(2,point1);
        p2_X = F_2(1,point2);
        p2_Y = F_2(2,point2);
        A = cat(1, A, [-p1_X -p1_Y -1     0     0  0 p1_X*p2_X p1_Y*p2_X]);
        A = cat(1, A, [    0     0  0 -p1_X -p1_Y -1 p1_X*p2_Y p1_Y*p2_Y]);
        b = cat(1, b, [-p2_X; -p2_Y]);
     
    end    
    V = A\b;
    %[U,S,V] = svd(A);
    
    %Rearrage to 3x3 matrix and store
    homographies(:,:,ih) = [V(1), V(2), V(3); V(4), V(5), V(6); V(7), V(8), 1];    
    %homographies(:,:,ih) = reshape(V(:,9),3,3);
    % score homography
    X2_ = homographies(:,:,ih) * X1;
    
    du = X2_(1,:)./X2_(3,:) - X2(1,:)./X2(3,:) ;
    dv = X2_(2,:)./X2_(3,:) - X2(2,:)./X2(3,:) ;
    ok{ih} = (du.*du + dv.*dv) < 6*6 ;
    hscores(ih) = sum(ok{ih}) ;
end

[value, index]=max(hscores);

t = maketform('projective',homographies(:,:,index)');
[mosaic1 xdata1 ydata1] = imtransform(Im1,t);
imagesc(mosaic1), colormap gray;

t = maketform('projective',eye(3));
[mosaic2 xdata2 ydata2] = imtransform(Im2,t);
imagesc(mosaic2), colormap gray;
imagesc(max(mosaic1,mosaic2));

xMin = min(xdata1(1), xdata2(1));
xMax = max(xdata1(2), xdata2(2));
yMin = min(ydata1(1), ydata2(1));
yMax = max(ydata1(2), ydata2(2));

xRange = ceil(abs(xMin) + xMax);
yRange = ceil(abs(yMin) + yMax);


mosaic = zeros(yRange,xRange);
mosaic(1:size(mosaic1,1), 1:size(mosaic1,2)) = mosaic1;
mosaic(abs(floor(xMin)):size(mosaic2,1)+abs(floor(xMin)), 112:size(mosaic2,2)+112) = mosaic2;

mosaic(-round(yMin):size(mosaic2,1)-round(yMin)-1, round(-xMin):xRange-2) = mosaic2;