function IsolatedCells = useSegmentationMask( matfile, imageMask, method, configData)
% useSegmentationMask : applies a segmentation mask on an image and returns
% an image of a specific cell/cells;
%
%   method - 'prompt' prompts the user to select cells
%          - [ 1, 5, 7, ...] selects the cells within this vector
%
%   imageMask can either be a logical Mask or a labeled Mask
%
%
%
%
%   Detailed explanation goes here

% create logical and labeled segmentation masks. Find number of cells in
% this mask
imLogic = logical( imageMask);
imLabel = bwlabel( imLogic);
numCells = max( imLabel(:) );

% if image2D is a 3D image with the third dimension being time, then add up
% the times
sizeImg = size( matfile,'imData'); 
if numel(sizeImg) < 5, nC = 1; else nC = sizeImg( 5); end
nT = sizeImg( 4);
nZ = sizeImg( 3);
nX = sizeImg( 2);
nY = sizeImg( 1);

% We can go one of two routes. If the method is 'prompt', then we'll
% display an image for the user and ask the user to select cells to
% isolate. Otherwise, we will just isolate the cells for the user that
% specified in the method vector.

 % Create the image with numbers on top of cells to prompt the user
% with. To do this, we need the centroid of each cell which which
% dictate the position the number will be placed on the image.
cc = bwconncomp( imLogic);
stats1 = regionprops( cc, 'Centroid');

% create background image for user
image2D = mean( matfile.imData( :,:,3,1:5:nT, 1), 4);

% dispImg( image2D);

imUser = image2D .* imLogic;
idx = (1:numCells)';
pos = zeros( numCells, 2);
for jObj = 1 : numCells
    pos(jObj, :) = round( stats1(jObj).Centroid );
end

imUser = mat2gray( imUser);
imUser( imUser > 0.3) = 0.3; imUser = mat2gray(imUser);
T = multithresh( imUser( imUser>0), 2);
% add numbers to the image
imUserNumbered = insertText( imUser, pos, idx, 'AnchorPoint', 'Center', 'BoxOpacity', 0, 'TextColor', 'red', 'FontSize', 20);
h = figure;
imagesc( imUserNumbered)
saveas(h, "MaskNumbered.png")

if strcmp( method, 'prompt')
       
    % display the image and prompt the user to select cells
    h = figure;
    imagesc( imUserNumbered)
    title('Cell Selection Window')
    set( h, 'WindowStyle', 'Docked');
    savefig(h, "MaskNumbered.png")

    % Prompt user to select cells, and check that specified cells are
    % possible.
    flagInput = 1; count = 0;
    while flagInput
        CellsSelected = input(['\n Please specify a single cell to select for analysis : \n'...
            '    Valid cell numbers are ' num2str( 1) '-' num2str( numCells ) ...
            '. \n    Selected Cells = ']);
        count = count+1;
        if all( ismember( CellsSelected, [1 : numCells] ) )
           flagInput = 0;
           close(h)
        elseif count > 5
            close(h)
            error('useSegmentationMask: Too many user tries, software aborting. \n Please run it again. \n')
        else
           fprintf('\n Incorrect Entry! Please try again. Please specify cells in the provided range and in this format [1, 5, 24, ...] \n') 
        end
    end

elseif isa( method, 'double') && size(method, 1)==1
    
    CellsSelected = method;
    
else
    error('useSegmentationMask: Incorrect argument, method must be either "prompt" or a vector specifying the cells to isolate')
end

% Sort the Cells selected so that they are in order of number
CellsSelected = sort(CellsSelected, 'ascend');

% Now we have selected some cells. For each cell we'll crop a 150*150 box
% centered at the cell.
boxSize = configData.cellSize;
numWanted = length( CellsSelected);

% We need the centroid of the cells to crop a region right around it.
% We will also ensure that the isolated cells are elliptical
padwid = ceil( boxSize)/2;
imLogicPad = padarray( imLogic, [padwid, padwid], 0);
cc = bwconncomp( imLogicPad);
stats = regionprops( cc, 'Centroid', 'MajorAxisLength', 'MinorAxisLength', 'Orientation');

% pad the original image
image3Dpad = zeros( nX+2*padwid, nY+2*padwid, nZ, nT, nC, 'uint16'); 
 
for jCell = 1 : numWanted 
    
    % Turn on only the correct cell label
    currCell = CellsSelected( jCell);
    imMask = imLabel;
    imMask( imMask ~= currCell ) = 0;
    imMask = logical( imMask);
%   dispImg( imMask); title('Mask for current cell'); drawnow

    % Create elliptical mask
    t = linspace(0,2*pi,1000);
    a = stats( currCell).MajorAxisLength/2;
    b = stats( currCell).MinorAxisLength/2;
    Xc = stats( currCell).Centroid(1);
    Yc = stats( currCell).Centroid(2);
    phi = deg2rad( -stats( currCell).Orientation);
    x = Xc + a*cos(t)*cos(phi) - b*sin(t)*sin(phi);
    y = Yc + a*cos(t)*sin(phi) + b*sin(t)*cos(phi);
    idx = sub2ind( size( imLogicPad), round(y), round(x) );
    imMaskPad = 0*imLogicPad;
    imMaskPad( idx) = 1; imMaskPad = imfill( imMaskPad, 'holes');
%   dispImg( imMaskPad); title('Mask 2'); drawnow
    
    % get region around cell;
    ub = ceil( stats( currCell).Centroid + padwid );
    
    lb = ceil( stats( currCell).Centroid - padwid );
 %   disp( sprintf('ub = %.1f ', ub)) 
 %    disp( sprintf('lb = %.1f ', lb)) 
    
    imMaskPad = imMaskPad(lb(2): ub(2), lb(1): ub(1) ); 
    imMaskPad = imMaskPad( 1: boxSize, 1 : boxSize); 
    % dilate it just a little to ensure any edge features are captured.
    imMaskPad = imdilate( imMaskPad, strel('disk', 2) );
    imMaskPad = logical( imMaskPad);

% dispImg( imMaskPad); title('mask'); drawnow

    % multiply with the original image to recover intensity information
    imMask3DPad = repmat( imMaskPad, 1,1,nZ, nT, nC );
    imCell3DPad = uint16( 0*imMask3DPad); 
    
	ubb = ceil( stats1( currCell).Centroid + padwid );
    lbb = ceil( stats1( currCell).Centroid - padwid );
    
    % disp( sprintf('ub2 =  %.1f ', ubb)) 
    % disp( sprintf('lb2 =  %.1f ', lbb)) 
	
    xl = 1; yl = 1; xr = 2*padwid; yr = xr;
    
    % disp( sprintf('xlxr = %.1f ', [xl, xr])) 
    % disp( sprintf('ylyr = %.1f ', [yl, yr])) 

	if lbb(1) < 1; xl = abs( lbb(1) )+1; end
        if ubb(1) > nX; xr = 2*padwid - abs( ubb(1)-nX ); end
        if lbb(2) < 1; yl = abs( lbb(2) )+1; end
        if ubb(2) > nY; yr = 2*padwid - abs( ubb(2)-nY ); end
	lbb( lbb < 1) = 1;
        ubb( ubb > nX) = nX;
	if ubb(2)-lbb(2) ~= yr-yl; ubb(2) = lbb(2) + (yr-yl); end
	if ubb(1)-lbb(1) ~= xr-xl; ubb(1) = lbb(1) + (xr-xl); end
	
%   imCell3DPad( yl:yr, xl:xr, :,:, :) = matfile.imData( lbb(2):ubb(2), lbb(1):ubb(1), :, :, :);
% 	imCellRaw = imCell3DPad;
%   imCell3DPad = imCell3DPad .* uint8( imMask3DPad);
    if nC == 1
        imCell3DPad( yl:yr, xl:xr, :,:) = matfile.imData( lbb(2):ubb(2), lbb(1):ubb(1), :, :);
    else
        imCell3DPad( yl:yr, xl:xr, :,:, :) = matfile.imData( lbb(2):ubb(2), lbb(1):ubb(1), :, :, :);
    end
    disp(class(imCell3DPad)); disp(class(imMask3DPad) )
    imCellRaw = imCell3DPad;
    imCell3DPad = imCell3DPad .* uint16( imMask3DPad);

    % obtain cropped regions for both the segmentation version and the
    % field of view region
    IsolatedCells( jCell).cell3D = imCell3DPad;
%    IsolatedCells( jCell).cellMIP = max( IsolatedCells( jCell).cell3D, [], 3);
    IsolatedCells( jCell).raw = imCellRaw;
%    IsolatedCells( jCell).rawMIP = mat2gray( max( IsolatedCells( jCell).raw, [], 3) );
    IsolatedCells( jCell).cellNumber = currCell;
    IsolatedCells( jCell).locations = imUserNumbered;
    IsolatedCells( jCell).planeTimes = matfile.planeTimes;
    IsolatedCells( jCell).metaData = matfile.metaData; 
end



end

