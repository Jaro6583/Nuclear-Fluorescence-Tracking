% This class contains all information about the voxel information of a cell 
classdef ImageData

    properties (Access = private)
        image % the image data
        numberOfChannels
        lifetime
        timeStep
        sizeVoxels
        numVoxels
        path
    end

    methods (Access = public)

        % ImageData {{{
        function obj = ImageData( image, varargin)
            % Initialize the Object
            % Input Arguments : 
            %   image : 5d array (X,Y,Z,T,C) 3-spatial dimensions, times and channels.
            %   Optional arguments : 
            %       'Lifetime' : 2-element vector specifying the start and end time for the existence of a cell in image. Entries must be > 1 and less than size(image, 4)
            %       'SizeVoxels' : 3-element double vector specifying the size of a voxel in each dimension in microns
            %       'TimeStep' : frame rate in seconds
            % e.g. ImageData( img, 'Lifetime', [ 2 15], 'SizeVoxels', [ 0.5 0.5 1.0], 'TimeStep', 1.4)
            
            % Parse Arguments
            opts = parseArgs( image, varargin{:});

            obj.image = image;
            obj.lifetime = opts.Lifetime;
            obj.timeStep = opts.TimeStep; 
            obj.sizeVoxels = opts.SizeVoxels; 
            obj.path = opts.Path;
            obj.numberOfChannels = size( image, 5);
            obj.numVoxels = size( image(:,:,:,1,1) );

            disp( 'Created ImageData Object')
            %ImageData.PrintInfo( obj);

            % parseArgs {{{
            function opts = parseArgs( image, varargin)
               
                % default lifetime : all times
                defaultLifetime = [1 , size(image, 4)];
                % default Size of voxels 
                defaultSizeVoxels = [1 1 1];
                % default timestep
                defaultTimeStep = 1;
                % default path
                defaultPath = '';

                % Input Parser
                p = inputParser;

                % Lifetime
                validLifetime = @(x) length(x) == 2 && all(isnumeric(x)) && all( x >= 1) && all( x <= size(image, 4) ) && all( round(x) == x);
                addParameter( p, 'Lifetime', defaultLifetime, validLifetime);

                % Size Voxels
                validSizeVoxels = @(x) length(x) == 3 && all(isnumeric(x))&& all( x > 0);
                addParameter( p, 'SizeVoxels', defaultSizeVoxels, validSizeVoxels);

                % TimeStep 
                validTimeStep = @(x) length(x) == 1 && isnumeric(x) && x >0; 
                addParameter( p, 'TimeStep', defaultTimeStep, validTimeStep);

                % Path
                validPath = @(x) ischar(x);
                addParameter( p, 'Path', defaultPath, validPath);

                parse( p, varargin{:} );
                opts = p.Results;

            end
            % }}}

        end
        % }}}

        % Set/Get methods {{{
        % Image : get and set 
        function image = GetImage(obj)
            image = obj.image;
        end
        function SetImage(obj, image)
            obj.image = image;
            obj.numberOfChannels = size( image, 5);
            obj.numVoxels = size( image(:,:,:,1,1) );
        end

        % Lifetime : Get and Set
        function lifetime = GetLifetime( obj)
            lifetime = obj.lifetime;
        end
        function SetLifetime(obj, lifetime)
            obj.lifetime = lifetime;
        end

        % TimeStep : Get and Set
        function timeStep = GetTimeStep( obj)
            timeStep = obj.timeStep;
        end
        function SetTimeStep(obj, timeStep)
            obj.timeStep = timeStep;
        end

        % SizeVoxels : Get and Set
        function sizeVoxels = GetSizeVoxels( obj)
            sizeVoxels = obj.sizeVoxels;
        end
        function SetSizeVoxels(obj, sizeVoxels)
            obj.sizeVoxels = sizeVoxels;
        end

        % Path : Get and Set
        function path = GetPath( obj)
            path = obj.path;
        end
        function SetPath(obj, path)
            obj.path = path;
        end
        % }}}

        % Play {{{
        function Play( obj, varargin)
            % Play a movie of the image stacks
            ImageData.playImageMovie( obj.image, varargin{:} );
        end
        % }}}
        
    end

    methods (Static = true)

        % InitializeFromCell {{{
        function imageData = InitializeFromCell( moviepath, varargin)
            % Imports a cell movie of segmented cell and uses it to initialize an ImageData object
            % Optional argument: 'Lifetime', [ starttime endtime]
            
            % Import Segmented Cell
            cellData = ImageData.importCell( moviepath);

            % Parse arguments
            opts = parseArgs( cellData.cell3D, varargin{:} );

            % Process timestep/framerate from imported cell
            % find mean time for all z-slices
            timesTC = squeeze(mean( cellData.planeTimes, 1) ); 
            timeSteps = diff(timesTC);
            % remove any nans and then use the median timeStep as the timeStep
            timeSteps( isnan( timeSteps) ) = 0;
            timeStep = median( timeSteps);
            timeStep = mean( timeStep);

            % Initialize ImageData object
            imageData = ImageData( cellData.cell3D, ...
                'Lifetime', opts.Lifetime, ...
                'SizeVoxels', [ cellData.metaData.sizeVoxelsX, cellData.metaData.sizeVoxelsY, cellData.metaData.sizeVoxelsZ], ...
                'TimeStep', timeStep, ...
                'Path', moviepath );

            % parseArgs {{{
            function opts = parseArgs( image, varargin)
               
                % default lifetime : all times
                defaultLifetime = [ 1 size(image, 4)];

                % Input Parser
                p = inputParser;
                validLifetime = @(x) length(x) == 2 && all(isnumeric(x)) && all( x >= 1) && all( x <= size(image, 4) ) && all( round(x) == x);
                addParameter( p, 'Lifetime', defaultLifetime, validLifetime);

                parse( p, varargin{:});
                opts = p.Results;

            end
            % }}}

        end
        % }}}

        % importCell {{{
        function cellData = importCell( moviepath)
            % Input: full path to movie of segmented cell
            % Output: cellData :  
            % CellData.raw (3D raw data)
            % CellData.cell3D (3D masked data) (optional)
            % CellData.planeTimes (average time values for all XY slices) (optional)
            % CellData.cellNumber (cell number in segmentations routine) (optional)
            % CellData.locations (image with cell numbers and locations shown) (optional)
            % CellData.cellCentroids (array with XY positions of cells masked in original segmentation routine) (optional)
            % CellData.metaData (metaData)

            % Acceptable formats
            validFormats = {'.mat', '.tif'};
            [ rpath, fname, ext] = fileparts( moviepath);
            if ~any( strcmp( ext, validFormats) )
                error('importSingleCell : file extension is invalid'); 
            end

            % Import depending on file extension
            switch ext
                case '.mat'
                    load( moviepath);
                case '.tif'

                    reader = bfGetReader(moviepath);
                    metaData = reader.getMetadataStore();
                    Meta.numVoxelsX = metaData.getPixelsSizeX(0).getValue();
                    Meta.numVoxelsY = metaData.getPixelsSizeY(0).getValue();
                    Meta.numVoxelsZ = metaData.getPixelsSizeZ(0).getValue();
                    Meta.numTimes = metaData.getPixelsSizeT(0).getValue();
                    Meta.numChannels = metaData.getPixelsSizeC(0).getValue();

                    % get physical sizes of voxels
                    Meta.sizeVoxelsX = double( metaData.getPixelsPhysicalSizeX(0).value() );
                    Meta.sizeVoxelsY = double( metaData.getPixelsPhysicalSizeY(0).value() );
                    Meta.sizeVoxelsZ = double( metaData.getPixelsPhysicalSizeZ(0).value() );

                    % Get data type of the images
                    im_temp = bfGetPlane( reader, reader.getIndex( 0, 0, 0) + 1);

                    % Pre-allocate 5D array for storing the image planes:
                    % array dimensions are arranged as ( x, y, z, t, c)
                    planeTimes = zeros( Meta.numVoxelsZ, Meta.numTimes, Meta.numChannels);
                    imData = zeros( Meta.numVoxelsY, Meta.numVoxelsX, Meta.numVoxelsZ, Meta.numTimes, Meta.numChannels, class(im_temp) );

                    % Read plane from series iSeries at Z, C, T coordinates (iZ, iC, iT)
                    % All indices are expected to be 1-based
                    reader.setSeries(0);
                    for jC = 1 : Meta.numChannels
                        for jZ = 1 : Meta.numVoxelsZ
                            for jT = 1 : Meta.numTimes
                                iPlane = reader.getIndex(jZ-1, jC-1, jT-1) + 1;
                                imData(:,:,jZ, jT, jC) = bfGetPlane(reader, iPlane);
                                try
                                    planeTimes( jZ, jT, jC) = metaData.getPlaneDeltaT( 0, iPlane).value;
                                catch
                                    planeTimes(jZ,jT,jC) = jT;
                                end
                            end
                        end
                    end

                    cellData.metaData = Meta;
                    cellData.cell3D = imData;
                    cellData.planeTimes = planeTimes;

            end

        end
        % }}}

        % playImageMovie {{{
        function playImageMovie( image, varargin)
            % Plays the image movie of the Cell.
            % Channels can either be a scalar or a vector of upto 3 elements specifying the channels to play
            % Color can either be a string or a string array of upto 3 elements specifying the color of each of the channels. the size of cColors must match the size of cChannels
            % e.g. ImageData.playImageMovie( image, 'Channels', [ 1 2], 'Colors', {'R', 'B'} )

            % Number of Channels
            nChannels = size( image, 5);

            % parse Args
            opts = parseArgs( nChannels, varargin{:} );

            % Pre-process image
            img = max( image, [], 3);
            img = squeeze( img);
            img = permute( img, [ 1 2 4 3] ); % XYCT
            imStack = zeros( size(img, 1), size(img,2), 3, size(img, 4) );

            % Find Channel for red color
            cR = opts.Channels( find( strcmp( opts.Colors, 'R') ) );
            if ~isempty( cR)
                imStack(:,:,1,:) = mat2gray( img(:,:,cR, :) );
            end

            % Find cChannel for green color
            cG = opts.Channels( find( strcmp( opts.Colors, 'G') ) );
            if ~isempty( cG)
                imStack(:,:,2,:) = mat2gray( img(:,:,cG, :) );
            end
            
            % Find cChannel for blue color
            cB = opts.Channels( find( strcmp( opts.Colors, 'B') ) );
            if ~isempty( cB)
                imStack(:,:,3,:) = mat2gray( img(:,:,cB, :) );
            end

            fps = 10;
            implay( imStack, fps);

            % parseArgs {{{
            function opts = parseArgs( nChannels, varargin)
               
                % default channels. Max upto 3.
                defaultChannels = 1: nChannels;
                if nChannels > 3
                    defaultChannels = 1:3;
                end
                
                % default Color (RGB allowed)
                defaultColors = {'R', 'G', 'B'};
                defaultColors = defaultColors( defaultChannels);

                % Input Parser
                p = inputParser;
                validChannels = @(x) all( x>=1 & x <= 3 & x <= nChannels);
                addParameter( p, 'Channels', defaultChannels, validChannels);
                validColors = @(x) length( x) >= 1  && length(x) <= 3 && length(x) <= nChannels;
                addParameter( p, 'Colors', defaultColors, validColors);
                parse( p, varargin{:});
                opts = p.Results;

            end
            % }}}

        end
        % }}}
        
        % PrintInfo {{{
        function PrintInfo( obj)
            % Prints information about the cell

            fprintf( '\nImageData : \n')
            fprintf( ' Movie Size XYZTC : %d x %d x %d x %d x %d\n', size(obj.image, 1), size(obj.image, 2), size(obj.image, 3), size(obj.image, 4), size( obj.image, 5) ) 
            fprintf( ' Lifetime : %d - %d\n', obj.lifetime(1), obj.lifetime(2) );
            fprintf( ' timeStep : %d seconds\n', obj.timeStep)
            fprintf( ' SizeZvoxels : [ %d %d %d ]\n', obj.sizeVoxels)
            fprintf( ' Path : %s\n', obj.path)
        end
        % }}}
        
        % FilterGaussBandpass {{{
        function imfilter = FilterGaussBandpass( image, sigMax, sigMin)
        % uses a bandpass filter to filter the image

            if iscell(image)
                cellYes = 1;
                imfilter = cell( size(image) );
                image = cat( 3, image{:});
            else
                cellYes = 0; 
                imfilter = 0*image;
            end

            % for each slice, do a 2d bandpass
            for jT = 1 : size(image,3)

                maxV = max( max( image(:,:,jT) ) );
                minV = min( min( image(:,:,jT) ) );
                im_filter_highfreq = imgaussfilt( image(:,:,jT), sigMin);
                im_filter_lowfreq = imgaussfilt( image(:,:,jT), sigMax);
                imgFilt =  (maxV - minV)*mat2gray(im_filter_highfreq - im_filter_lowfreq) + minV;
                if cellYes, imfilter{jT} = imgFilt;  else, imfilter(:,:,jT) = imgFilt; end

            end

        end
        % }}}

    end

end
