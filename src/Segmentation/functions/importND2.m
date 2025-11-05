function importND2( filepath)
% importND2: loads the nd2 file using the bfmatlab package and stores the information in matlab variables.
   
    [ path, onlyName, ext] = fileparts( filepath);
    saveFile = [path, onlyName, '.mat'];

    % Loop through files and extract metadata and image data
    reader = bfGetReader( filepath); % Creates a Bioformats reader objects
    metaData = reader.getMetadataStore();   % Use class function to acess metadata of image

    % Extract key parameters from  metadata
    % get number of voxels in all the dimensions
    Meta.numVoxelsX = metaData.getPixelsSizeX(0).getValue();
    Meta.numVoxelsY = metaData.getPixelsSizeY(0).getValue();
    Meta.numVoxelsZ = metaData.getPixelsSizeZ(0).getValue();
    Meta.numTimes = metaData.getPixelsSizeT(0).getValue();
    Meta.numChannels = metaData.getPixelsSizeC(0).getValue();

    % get physical sizes of voxels
    Meta.sizeVoxelsX = double( metaData.getPixelsPhysicalSizeX(0).value() );
    Meta.sizeVoxelsY = double( metaData.getPixelsPhysicalSizeY(0).value() );
    Meta.sizeVoxelsZ = double( metaData.getPixelsPhysicalSizeZ(0).value() );

    % Get other useful information
    Meta.numPlanes = metaData.getPlaneCount(0);
    %Meta.wavelength = double(metaData.getChannelEmissionWavelength(0,0).value);
    %Meta.exposureTime = double(metaData.getPlaneExposureTime(0,0).value);
    Meta.fileName = onlyName;
    Meta.fileExtension = ext;

    % Pre-allocate 5D array for storing the image planes:
    % array dimensions are arranged as ( x, y, z, t, c)
    planeTimes = zeros( Meta.numVoxelsZ, Meta.numTimes, Meta.numChannels);

    % Add planes to imData for each channel, for every frame, for every z
    % plane.
%     for jChannel = 1 : Meta.numChannels
%         maxVal(jChannel) = uint16(0); minVal( jChannel) = uint16(2^16);
% 	for jTime = 1 :5:  Meta.numTimes
%             for jZ = 1 : 3: Meta.numVoxelsZ
%                 % get index of plane with specific z, t, and c
%                 jPlane = reader.getIndex( jZ - 1, jChannel - 1, jTime - 1) + 1;
%                 % get image plane corresponding to the index above
%                 im2 = bfGetPlane( reader, jPlane);
% 		if max( im2(:) ) > maxVal( jChannel) && ~all( im2(:) == max(im2(:) ) )
% 		maxVal(jChannel) = max( im2(:) ); ind = [jZ, jTime];
% 		end
% 		if min( im2(:) ) < minVal( jChannel) && ~all( im2(:) == min(im2(:) ) )
% 		minVal(jChannel) = min( im2(:) );
% 		end
%             end
%         end
%     end
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
                % get time correponding to this plane
                try; planeTimes( jZ, jTime, jChannel) = metaData.getPlaneDeltaT( 0, jPlane).value; end
            end
        end
    end

    metaData = Meta;
    save( saveFile, 'imData', 'planeTimes', 'metaData', '-v7.3')
    disp(saveFile)

end
