
function trackCellsInSegmentation()

	% generate 9 random points spread out in space (on a 3x3 grid)
	[x0, y0] = meshgrid( 1:3, 1:3);
	x0 = x0(:); y0 = y0(:);
	cells(1).x = x0; 
	cells(1).y = y0;
	cells(1).time = 1;

	% add random values to (x,y) positions to change them, mimicing cell crawling, segmentation errors. gaussian noise with mean 0 and std (0.1)
	for jTime = 2 : 10

		cells( jTime).x = cells(jTime-1).x + randn( length( cells(1).x) )/10;
		cells( jTime).y = cells(jTime-1).y + randn( length( cells(1).y) )/10;
		cells( jTime).time = jTime;

	end

	% Assign each cell a unique color
	colors = distinguishable_colors( 2*length( cells(1).x ), {'w', 'k'});
	for jCell = 1 : length( cells(1).x )
		cells(1).display( jCell).color = colors( jCell, :);
	end

	figure;
	scatter( cells(1).x, cells(1).y, 100, cat( 1, cells(1).display(:).color), 'filled')
	set( gca, 'xlim', [0 4], 'ylim', [0 4]); 


	
	






end
