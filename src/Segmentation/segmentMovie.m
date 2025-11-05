function segmentMovie( mpath)
%{
Locate the movie:
    ensure path to movie is valid, or
    prompt user to select movie if path is not provided
    valid formats : nd2, tiff? 
%}

addpath( genpath( pwd) )

disp('Locating the movie...')
if nargin==0
    disp('  Please select a .nd2 file in the dialog box...')
    [mname, rpath] = uigetfile({'*.nd2'});
    if mname==0
        error( 'segmentMovie: file not selected in dialog box'), end
    mpath = [rpath, filesep, mname];
end

[rpath, mname, ext] = fileparts(mpath);
if ~strcmp(ext, '.nd2')
    error( 'segmentMovie: input path did not have a .nd2 extension.'), end

%{
Import the movie: nd2 or tiff (filetype-specific import functions)
    We save the movie, because holding it in memory leaves very little mem for computation
    Note: could be changed in a future iteration to hold data in memory
%}
disp('Importing the movie...')
importND2( mpath);
mpathmat = [rpath,filesep,mname,'.mat'];
configData = segmentMovie_config();

%{
Segment the movie based on a specific segmentation routine run on a specific channel
%}

disp('Segmenting the movie...')
varInfo = who('-file', mpathmat);
mov = matfile( mpathmat, 'Writable', true);
sizeData = size( mov, 'imData');
imXYZT = squeeze( mov.imData( :, :, :, 1:5:sizeData(4), configData.channelForSegmentation) );
imXY = mean( mean( imXYZT,4),3);	

disp('  Generating the segmentation mask for the movie...')
if configData.manualSeg
    maskInfo = drawSegmentationMask( imadjust(mat2gray(imXY)) , imXY, configData);
else
    maskInfo = generateSegmentationMask( imXY, configData);
end
mov.MaskLogical = maskInfo.MaskLogical;
mov.MaskColored = maskInfo.MaskColored;
mov.NumCells = maskInfo.NumCells;
if configData.manualSeg || strcmp( configData.cellsToSegment, 'all')
    cellsToSegment = 1: maskInfo.NumCells;
elseif strcmp( configData.cellsToSegment, 'prompt')
    cellsToSegment = 'prompt';
else
    error( 'segmentMovie: cellsToSegment config property is invalid')
end

disp('  Applying the segmentation mask to the movie...')
segCells = useSegmentationMask(mov, maskInfo.MaskLogical , cellsToSegment, configData);

disp('  Saving the segmented cells...')
cFoldPath = [rpath, filesep, mname,'_cells'];
mkdir( [rpath, filesep, mname,'_cells'])
cc = bwconncomp( mov.MaskLogical); st = regionprops( cc, 'Centroid');
cellCentroids = flip( cat( 1, st.Centroid), 2);
cellsToSegment = [segCells(:).cellNumber];
for jCell = 1: length(cellsToSegment)
    currCell = cellsToSegment( jCell);
    cellData = segCells(jCell);
    cellData.cellCentroids = cellCentroids;
    save( [cFoldPath, filesep, mname,'_', num2str( currCell)], 'cellData')
end

disp(['Segmentation complete. segmented cells stored in location: ', cFoldPath])

end
