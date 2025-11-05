function SegmentationInfo = generateSegmentationMask(image2D, config)
% generateSegmentationMask: Generates a segmentation mask for a 2D image of
% fission yeast cells.
%
%   Parameters used: sensitivity=0.6 for the wiener2 noise removal filter
%


%   Detailed explanation goes here

% We will employ thresholding and morphological methods to isolate the cell
% cytoplasm signal from the imaging background

% Begin with a wiener filter to reduce the local deviations in noise.
image2D = mat2gray(image2D);
[imWiener, ~] = wiener2(image2D, [3 3]);

% Now, we'll try to even out the intensities in the backgroun and within
% the cell. First, the background. This will be done by opening the image
% with a structural element followed by a reconstuction. Next, the
% cytoplasmic region will be evened out by closing the image followed by a
% reconstruction.

se = strel('disk', 7);
% Open by reconstruction(erosion expands areas starting in the dark regions
% and then going to the bright regions). Reconstruction then uses the
% intensities of the imWiener to reconstruct the image.
Iobr = imreconstruct(imerode(imWiener, se), imWiener);

% Close by reconstruct( this involves a dilation followed by a
% reconstruction). Dilation spreads out the bright regions before the dark
% ones. A reconstruction is done to match the intensities to the original
% image.
Iobrcbr = imcomplement( imreconstruct( imcomplement( imdilate( Iobr, se) ), imcomplement( Iobr) ) );

% now use adaptive thresholding to threshold the background( which may be
% different in different regions of the image). 0.6 has been determined by
% trial and error. This number might be different if the levels of
% background and cytoplasm are changed.
T = adaptthresh(Iobrcbr, 0.6);
imAthresh = imbinarize( Iobrcbr, T);

% The next thing we'll do is to fill holes in the image
imFill = imfill(imAthresh, 'holes');

% Now we have a decently good mask. I think we can do better. Open the
% image with a structural element
se = strel('disk', 2);
imOpen = imopen(imFill, se);

% Remove any small regions that have somehow made it through
imAfilt = bwareafilt( imOpen, [500 Inf]);

% We won't be able to analyze any cells that are attached to the border of
% this field of view since their features may be cut off. We remove cells
% at the border
imCborder = imclearborder( imAfilt);

% We'll make each object convex in the mask. We do this because cells are
% convex in shape, just like baloons are convex. We do this by finding the
% connected components and replacing them by their convex hull.
cc = bwconncomp( imCborder);
stats = regionprops( cc, 'ConvexHull', 'Centroid', 'MinorAxisLength');
imConvex = zeros( size(image2D) );
imConvW = zeros( size(image2D) );
for jObj = 1 : cc.NumObjects
    imTemp = zeros( size(image2D) );
    colSub = round( stats( jObj).ConvexHull(:,1) );
    rowSub = round( stats( jObj).ConvexHull(:,2) );
    idx = sub2ind( size(imTemp), rowSub, colSub );
    if stats(jObj).MinorAxisLength < 50
        imTemp( idx ) = 1;
        imConvex = logical( imConvex + bwconvhull( imTemp) );
    else
        % the object is wierd. We'll pass it through some functions to
        % salvage cells.
        imTemp( cc.PixelIdxList{jObj} ) = 1;
        imC = bwconvhull( imTemp);
        [x, y] = find( imC == 1);
        im2 = imC; im2( min(x) : max(x), min(y): max(y) ) = 1;
        [~, ~, nms, ~]=steerableDetector(image2D, 4, 2);
        nms = nms .* imTemp;
        imVals = nms( nms > 0);
        threshVals = multithresh( imVals, 2);
        imNMS = nms; imNMS( imNMS < threshVals(1) ) = 0;
        imNMS = mat2gray(imNMS.*imTemp); % imNMS(imNMS > 0) = 1;
        imGauss = mat2gray(imgaussfilt( imNMS, 8) );
        T = adaptthresh(imGauss,0.9);
        imNew = imbinarize(imGauss.*image2D,0.02);
        imNew = bwareafilt( imNew, [100 Inf]);
        ccW = bwconncomp( imNew);
        statsW = regionprops( ccW, 'ConvexHull', 'MinorAxisLength');
        imConvWLocal = zeros( size(image2D) );
        for jW = 1 : ccW.NumObjects
            colSub = round( statsW( jW).ConvexHull(:,1) );
            rowSub = round( statsW( jW).ConvexHull(:,2) );
            idx = sub2ind( size(imTemp), rowSub, colSub );
            if statsW( jW).MinorAxisLength < 50
                imYes = zeros( size(image2D) );
                imYes( idx ) = 1;
                imConvWLocal = logical( imConvWLocal + bwconvhull( imYes) );
            end
        end
        % keep dilating image until any two objects merge
        ccW = bwconncomp( imConvWLocal); statsW = regionprops( ccW, 'MinorAxisLength');
        numObjLocal = ccW.NumObjects; currObj=numObjLocal; count = 0; imConvOld = imConvWLocal;
        while currObj == numObjLocal && count < 10 && all([statsW.MinorAxisLength] < 45)
            imConvOld = imConvWLocal;
            imConvWLocal = imdilate( imConvWLocal, strel('disk', 1) );
            ccWcurr= bwconncomp( imConvWLocal); statsW = regionprops( ccWcurr, 'MinorAxisLength');
            currObj = ccWcurr.NumObjects; count = count+1;
        end
        imConvWLocal = imConvOld;
        imConvW = logical( imConvW + imConvWLocal );

    end
end

% Now we want to add the wierd cells back into the convex image with known
% good cells. However, perhaps adding these wierd cells will ruin good
% cells if there is an overlap.
imConvexStorage = imConvex;
for jX = 2 : size(imConvex,1)-1
    for jY = 2 : size(imConvex,1)-1
        nhood = imConvexStorage( jX-1:jX+1, jY-1:jY+1);
        % If this pixVal is nonzero, than we'll check if the nhood contains
        % anything other than the pixValue or zero. If that is not the
        % case, we will set the pixel value to zero.
        if all(nhood(:) == 0)
            imConvex( jX, jY) = imConvW( jX, jY);
        end
    end
end



% dispImg({imTemp.*image2D, im2.*image2D, imGauss.*image2D, imNew.*image2D}, [2 2])
    


% Now, we have a convex image. During this process its possible that we
% have created regions where two possible cells are barely making contact.
% We will erode this region.
imCeroded = imerode(imConvex, strel('disk', 1));

% At this point, we have a number of possible cell objects. Each connected
% object in imMask will be treated as an individual cell. However, some of
% these objects will fail in certain criteria for cell shape and we'll
% eliminate them. This criteria includes: 
% 1) Cell Area - minimum 675 micronSquared (45*15=675 - 6 microns x 2 microns)
% 2) Cell Elliptical-ness - SemiMajorAxis/SemiMinorAxis > 1.5 and < 10
% 3) Cell Diameter - SemiMinorAxis > 15 pixels(2 microns) and < 35 pixels(4.3 microns) 

cc = bwconncomp( imCeroded);
stats = regionprops( cc, 'MajorAxisLength', 'MinorAxisLength', 'Area');
idxRm = [];
for jObj = 1 : cc.NumObjects
    
    area = stats( jObj).Area;
    ellip = stats( jObj).MajorAxisLength / stats( jObj).MinorAxisLength;
    diameter = stats( jObj).MinorAxisLength;
    if area < 675 || ellip < 1.5 || ellip > 15 || diameter < 15 || diameter > 50
        idxRm = [idxRm, jObj];
    end
    
end

% Now we'll set the non-cell regions to 0.
imLabel = bwlabel( imCeroded);
for jRm = 1 : length(idxRm)
    imLabel( imLabel == idxRm( jRm) ) = 0;
end
imLabel = bwlabel( logical( imLabel) );

% Finally we remember that we eroded the mask. So now, we dilate it once
% again to recover the lost edges.
imMask = imdilate( imLabel, se);

% As the last step, we'll ensure that no all objects are separated by
% zeros.
for jX = 2 : size(imMask,1)-1
    for jY = 2 : size(imMask,1)-1
        nhood = imMask( jX-1:jX+1, jY-1:jY+1);
        pixVal = imMask( jX, jY);
        % If this pixVal is nonzero, than we'll check if the nhood contains
        % anything other than the pixValue or zero. If that is not the
        % case, we will set the pixel value to zero.
        if pixVal ~= 0 && any(nhood(:) ~= pixVal & nhood(:) ~= 0)
            imMask( jX, jY) = 0;
        end
    end
end


% Store segmentation information for output
SegmentationInfo.MaskLabeled = imMask;
SegmentationInfo.MaskLogical = logical(imMask);
SegmentationInfo.MaskColored = label2rgb(imMask, 'jet', 'k');
SegmentationInfo.NumCells = max( imMask(:) );


end

