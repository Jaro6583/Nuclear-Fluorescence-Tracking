function getAmplitudeAtCellLocations( movpath, cellFiles)
% importND2: loads the nd2 file using the bfmatlab package and stores the information in matlab variables.
   
    
    % Loop through files and extract metadata and image data
    reader = bfGetReader( movpath); % Creates a Bioformats reader objects
    metaData = reader.getMetadataStore();   % Use class function to acess metadata of image

    % Extract key parameters from  metadata
    % get number of voxels in all the dimensions
    Meta.numVoxelsX = metaData.getPixelsSizeX(0).getValue();
    Meta.numVoxelsY = metaData.getPixelsSizeY(0).getValue();
    Meta.numVoxelsZ = metaData.getPixelsSizeZ(0).getValue();
    Meta.numTimes = metaData.getPixelsSizeT(0).getValue();
    Meta.numChannels = metaData.getPixelsSizeC(0).getValue();

    % Initialize intensity array for data
    imData = zeros( Meta.numVoxelsX, Meta.numVoxelsY, Meta.numVoxelsZ, ...
        Meta.numTimes, Meta.numChannels, class(bfGetPlane( reader, 1)));

    for jChannel = 1 : Meta.numChannels
        for jTime = 1 : Meta.numTimes
            for jZ = 1 : Meta.numVoxelsZ
                % get index of plane with specific z, t, and c
                jPlane = reader.getIndex( jZ - 1, jChannel - 1, jTime - 1) + 1;
                % get image plane corresponding to the index above
                imData( :, :, jZ, jTime, jChannel) = bfGetPlane( reader, jPlane);
            end
        end
    end
    
    % Pad the array
    pSize = 100;
    impadded = padarray( imData, [pSize,pSize,0,0]);
    
    
    for jc = 1 : length(cellFiles)
        
        cellFile = cellFiles{jc};
        [ path, onlyName, ext] = fileparts( cellFile);
        saveFile = [path, filesep, onlyName, '_amp.mat'];

    
        % Load segmented cell data
        load(cellFile); % loads 'cellData'
        centroid = pSize+round( cellData.cellCentroids( cellData.cellNumber, :) );
        nX = size( cellData.cell3D,1);
        nY = size( cellData.cell3D,2);
        nZ = size( cellData.cell3D,3);

        % Get region in imData around centroid matching the size of the cell
        xy_init = round( centroid - [ nX, nY]/2);
        xy_final = xy_init + [nX,nY] - 1;
        newCell = impadded( xy_init(1): xy_final(1), ...
            xy_init(2): xy_final(2), :, :, :);

        % get mask (if possible) from cellData
        mask = mat2gray( cellData.cell3D(:,:,:,1) );
        mask( mask(:) > 0) = 1;

        maxAmp = zeros(1, size(newCell,4) );
        minAmp = maxAmp;

        for jt = 1 : size(newCell,4)
            imT = newCell(:,:,:,jt).*uint16(mask);
            minAmp(jt) = min( imT(imT ~=0));
            maxAmp(jt) = max( imT(:) );
        end
        fprintf(' med_maxAmp = %d\n  med_minAmp = %d\n',median(maxAmp), median(minAmp))
        
        % Save amplitude ranges
        save(saveFile, 'maxAmp', 'minAmp');
        
        % Save these also in the original cellData metaData
        cellData.metaData.ampMax = maxAmp;
        cellData.metaData.ampMin = minAmp;
        save( cellFile, 'cellData');
        
    end
    
end