function dispImg( varargin)
        
if length(varargin) == 1 && ( isa( varargin{1}, 'double') || isa( varargin{1}, 'logical') || isa( varargin{1}, 'uint8') || isa( varargin{1}, 'single') )
    % its a single image
    img = varargin{1};
    figure; imagesc(img); axis equal; 
    if size(img,3) == 1
        colormap gray
    end

    pos = get(gcf, 'position');
    set(gcf, 'pos', [-10000 pos(2:end)]);
    set(gcf, 'WindowState', 'maximized');
    set(gca, 'xlim', [1 size(img, 2)], 'ylim', [1 size(img, 1)], 'xtick',[], 'ytick',[] )
    
elseif length( varargin) > 1
    % there are multiple images for comparison. The display order is
    % specified in the last argument [nrows ncols]
    
    if size( varargin{end}, 2) ~= 2
        error('dispImg: the last argument must specify the order of plots [nRows nCols]')
    end
    
    nFig = length( varargin)-1;
    order = varargin{ end};
    nR = order(1); nC = order(2);
    img = varargin(1 : nFig);
    figure; 
    
    for jFig = 1 : nFig

        subplot( nR, nC, jFig); imagesc( img{jFig} ); axis equal; 
        if size( img{jFig}, 3) == 1
            colormap gray
        end
        set(gca, 'xlim', [1 size(img{jFig}, 2)], 'ylim', [1 size(img{jFig}, 1)], 'xtick',[], 'ytick',[] )
    end
    pos = get(gcf, 'position');
    set(gcf, 'pos', [-10000 pos(2:end)]);
    set(gcf, 'WindowState', 'maximized');
        
else
    error('Oops, something went wrong!')
end

end
