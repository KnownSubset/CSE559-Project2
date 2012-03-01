run('vlfeat-0.9.14/toolbox/vl_setup');
imBlur = imfilter(imread('/Users/nathan/Development/CSE559-Project2/WP_000288.jpg'),fspecial('Gaussian',[5 5],1));
Image1 = im2double(imresize(imBlur,[500,700]))/255;
Im1 = rgb2gray(Image1);
imBlur = imfilter(imread('/Users/nathan/Development/CSE559-Project2/WP_000291.jpg'),fspecial('Gaussian',[5 5],1));
Image2 = im2double(imresize(imBlur,[500,700]))/255;
Im2 = rgb2gray(Image2);


%%SIFT Feature discovery
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
[dmatches, dscores] = vl_ubcmatch(D_1,D_2); % using this only for debugging

matches = [];
scores = [];
% A descriptor D1 is matched to a descriptor D2 only if the
% distance d(D1,D2) multiplied by THRESH is not greater than the
% distance of D1 to all other descriptors. The default value of
% THRESH is 1.5.
for ix=1:size(D_1,2)
    
    d1 = D_1(:,ix);
    d2s = zeros(1,size(D_2,2));
    for iy=1:size(D_2,2)
        d2 = D_2(:,iy);
        distance = 0;
        for iz=1:128
            distance = distance + (double(d1(iz,1)) - double(d2(iz,1)))^2;
            
        end
        d2s(iy) = sqrt(distance)*1.5;
    end
    [score, match] = min(d2s);
    if (score < 200)
        matches = cat(2,matches,[ix; match]);
        scores = cat(2,scores,d2s(match));
    end
end

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

%% Ransac
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

inliers = value/size(scores,2) % percentage of inliers

% --------------------------------------------------------------------
%                                                         Show matches
% --------------------------------------------------------------------

dh1 = max(size(Im2,1)-size(Im1,1),0) ;
dh2 = max(size(Im1,1)-size(Im2,1),0) ;
ok = ok{index} ;
figure(1) ; clf ;
subplot(2,1,1) ;
imagesc([padarray(Im1,dh1,'post') padarray(Im2,dh2,'post')]) ;
o = size(Im1,2) ;
line([F_1(1,matches(1,:));F_2(1,matches(2,:))+o], ...
     [F_1(2,matches(1,:));F_2(2,matches(2,:))]) ;
title(sprintf('%d tentative matches', size(matches,2))) ;
axis image off ;

subplot(2,1,2) ;
imagesc([padarray(Im1,dh1,'post') padarray(Im2,dh2,'post')]) ;
o = size(Im1,2) ;
line([F_1(1,matches(1,ok));F_2(1,matches(2,ok))+o], ...
     [F_1(2,matches(1,ok));F_2(2,matches(2,ok))]) ;
title(sprintf('%d (%.2f%%) inliner matches out of %d', ...
              sum(ok), ...
              100*sum(ok)/size(matches,2), ...
              size(matches,2))) ;
axis image off ;

drawnow ;


%% Merge images together
t = maketform('projective',H');
[mosaic1 xdata1 ydata1] = imtransform(Im1,t);
imagesc(mosaic1), colormap gray;

t = maketform('projective',eye(3));
[mosaic2 xdata2 ydata2] = imtransform(Im2,t);
imagesc(mosaic2), colormap gray;

xMin = min(xdata1(1), xdata2(1));
xMax = max(xdata1(2), xdata2(2));
yMin = min(ydata1(1), ydata2(1));
yMax = max(ydata1(2), ydata2(2));

xRange = ceil(abs(xMin) + xMax);
yRange = ceil(abs(yMin) + yMax);


mosaic = zeros(yRange,xRange);
mosaic(1:size(mosaic1,1), 1:size(mosaic1,2)) = mosaic1;
%size(mosaic(-round(yMin)+5:size(mosaic2,1)-round(yMin)-2, round(-xMin)+1:xRange-2))
%size(mosaic2(2:size(mosaic2,1)-5,2:size(mosaic2,2)))


%need some work on feather the edges of the images
%mosaic(-round(yMin)+5:size(mosaic2,1)-round(yMin)-2, round(-xMin)+1:xRange-2) = mosaic2(2:size(mosaic2,1)-5,2:size(mosaic2,2));
mosaic(3:502,2:701) = mosaic2;
imagesc(mosaic);


%% Blending
%blendRegion1 = mosaic1(106:515,floor(abs(xdata1(1)))-5:size(mosaic1,2)); % 410x214
%blendRegion1 =  blendRegion1(:, 1+5:size(blendRegion1,2)+5);
%blendRegion2 = mosaic2(:,1:xdata1(2)+2);

blendRegion1 = mosaic1(:,xdata1(2)/2:xdata1(2));
blendRegion2 = mosaic2(5:500,1:xdata1(2)/2+1);

%% image pyramid
imBlur = blendRegion1;
ls1 = cell([1 6]);gs1 = cell([1 6]);

for ix = 1:6;
    imG = imfilter(imBlur,fspecial('Gaussian',[5 5],1));
    imL = imBlur - imG;  % Laplacian
    imBlur = imresize(imG,0.5);
    ls1{ix} = imL;
    gs1{ix} = imG;
end

imBlur = blendRegion2;
ls2 = cell([1 6]);gs2 = cell([1 6]);
for ix = 1:6;
    imG = imfilter(imBlur,fspecial('Gaussian',[5 5],1));
    imL = imBlur - imG;  % Laplacian
    imBlur = imresize(imG,0.5);
    ls2{ix} = imL;
    gs2{ix} = imG;
end

pyramid = cell(1,6); % the blended pyramid
lss = cell(1,6); % the blended pyramid
for p = 1:3
    
	[Mp Np ~] = size(ls1{p});
	ls = zeros(Mp, Np*2);
    ls(:,1:Np) = ls2{p};
    ls(:,Np+1:2*Np) = ls1{p};
    lss{p} = ls;
    maskap = gs1{p};
	maskbp = gs2{p};
	pyramid{p} = ls1{p}.*maskap + ls2{p}.*(1 - maskbp);  %Form a combined pyramid LS from LA and LB using nodes of GR as weights: LS(i,j) = GR(I,j,)*LA(I,j) + (1-GR(I,j))*LB(I,j)
end

imBlur = zeros(size(pyramid{6}));
for p = 3:2
    pyramid{p-1} = pyramid{p-1} + impyramid(pyramid{p}, 'expand'); 
    lss{p-1} = lss{p-1} + impyramid(lss{p}, 'expand'); 
end
xStart = floor(abs(xdata1(1)));
[lssX, lssY] = size(lss{1});
mosaic(4:499,xStart+1:xStart+size(lss{1},2)) = mosaic(4:499,xStart+1:xStart+size(lss{1},2)) - lss{1};
colormap gray;
figure(2), imagesc(mosaic), colormap gray;
figure(3), imagesc(lss{1})
