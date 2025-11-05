function SegmentationInfo = drawSegmentationMask(image2D, imageTime, config)
% drawSegmentationMask: User-drawn segmentation mask for a 2D image of
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

% Display 2D image and prompt user to draw ellipse ROIs on the image until the user stops the drawing process
h = figure; 
set(h,'WindowStyle','docked')
imagesc( mat2gray(imWiener)); axis equal; colormap gray; xlim([0 size(imWiener,2)]); ylim([0 size(imWiener,1)])
% nCells = 2;
% ellipses = cell(1,nCells);
% ellipses{1} = drawellipse()

ellipses = {}; masks = {};
currkey='1';
% do not move on until enter key is pressed
while currkey~='2'
    disp('Draw and edit an ellipse ROI.')
    disp('  Press 1 to confirm this and add another')
    disp('  Press 2 to confirm this and stop')
    ell = drawellipse();
    pause; % wait for a keypress
    ellipses = {ellipses{:}, ell};
    masks = {masks{:}, createMask(ell,image2D)};
    currkey = get(gcf,'CurrentKey');
    if currkey=='2'
        disp('Stopping (detected 2 press)')
        break
    end
end

for jj = 1 : length(ellipses)
    delete(ellipses{jj})
end
close(h);
imMaskSum = 0*image2D;
for j = 1 : length( masks)
    imMaskSum = imMaskSum + masks{j};
end
imMaskSum = logical( imMaskSum);

% Now, we have a convex image. During this process its possible that we
% have created regions where two possible cells are barely making contact.
% We will erode this region.
imCeroded = imerode(imMaskSum, strel('disk', 1));

% At this point, we have a number of possible cell objects. Each connected
% object in imMask will be treated as an individual cell. However, some of
% these objects will fail in certain criteria for cell shape and we'll
% eliminate them. This criteria includes: 
% 1) Cell Area - minimum 675 micronSquared (45*15=675 - 6 microns x 2 microns)
% 2) Cell Elliptical-ness - SemiMajorAxis/SemiMinorAxis > 1.5 and < 10
% 3) Cell Diameter - SemiMinorAxis > 15 pixels(2 microns) and < 35 pixels(4.3 microns) 

% % % cc = bwconncomp( imCeroded);
% % % stats = regionprops( cc, 'MajorAxisLength', 'MinorAxisLength', 'Area');
% % % idxRm = [];
% % % for jObj = 1 : cc.NumObjects
% % %     
% % %     area = stats( jObj).Area;
% % %     ellip = stats( jObj).MajorAxisLength / stats( jObj).MinorAxisLength;
% % %     diameter = stats( jObj).MinorAxisLength;
% % %     if area < 675 || ellip < 1.5 || ellip > 15 || diameter < 15 || diameter > 50
% % %         idxRm = [idxRm, jObj];
% % %     end
% % %     
% % % end

% Now we'll set the non-cell regions to 0.
imLabel = bwlabel( imCeroded);
% % % for jRm = 1 : length(idxRm)
% % %     imLabel( imLabel == idxRm( jRm) ) = 0;
% % % end
imLabel = bwlabel( logical( imLabel) );

% Finally we remember that we eroded the mask. So now, we dilate it once
% again to recover the lost edges.
imMask = imdilate( imLabel, strel('disk',1));

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

