function [movie3D_CH1, movie3D_CH2, voxel_size, time_step] = GetMovieInfo(data_path)

    %% Loading ImageObj and cut7/MT movies
    disp('Loading the movie...')
    ImageObj = ImageData.InitializeFromCell(data_path);
    movie5D = ImageObj.GetImage();   % Both channel = 5D
    % Getting spatial and temporal dimension of the data
    voxel_size = ImageObj.GetSizeVoxels;
    time_step = ImageObj.GetTimeStep;
    movie3D_CH1 = double(movie5D(:,:,:,:,1));
    movie3D_CH2 = double(movie5D(:,:,:,:,2));
    clear movie5D;

end