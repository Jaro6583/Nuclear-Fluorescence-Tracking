%% Add path to accessory functions
addpath(genpath('functions'))

%% Located and load the .ND2 Data (Segmented Cells.mat)
cell_path = "C:\Users\jared\Desktop\Research\Nuclear-Fluorescence-Tracking\src\MB1411\Segmented Cells\";
cell_type = "1411_200R_150G_q25s_25deg";
cell_number = "010_2";

data_path = cell_path + cell_type + "_" + cell_number + ".mat";
data_path = convertStringsToChars(data_path);

%% Loading ImageObj and cut7/MT movies
[movie3D_CH1, movie3D_CH2, voxel_size, time_step] = GetMovieInfo(data_path);

data_dim = size(movie3D_CH2);

%% Some quick test:
tt = [1,10,26,40,50,66,72];
t=26;
movie_t_7z = movie3D_CH2(:,:,:,t);

data_dim(3) = 6;
% Original slices: 
figure;
for z = 1:data_dim(3)
    subplot(2,3,z);
    imagesc(movie_t_7z(:,:,z)); colormap gray; colorbar;
    Z_info = "z = " + num2str(z);
    title(Z_info);
end

% Output raw data as a .mat file (MATLAB Data file)
save('raw_data.mat', 'movie_t_7z')
pause(1);

% Initial thresholding: 
c_min = min(movie_t_7z(movie_t_7z ~= 0), [], 'all', 'omitnan');
c_med = median(movie_t_7z(movie_t_7z ~= 0), 'omitnan');
c_max = max(movie_t_7z(movie_t_7z ~= 0), [], 'all', 'omitnan');
c_low = c_min + 0.5*(c_med-c_min);
c_upp = c_med + 0.05*(c_max-c_med);

% Display BW filtered image: 
movie_mod = movie_t_7z;
movie_mod(movie_mod < c_upp)=0;
movie_mod(movie_mod > c_upp)=1;
figure;
for z = 1:data_dim(3)
    subplot(2,3,z);
    imagesc(movie_mod(:,:,z)); colormap gray;
    Z_info = "z = " + num2str(z);
    title(Z_info);
end
pause(1);

% Apply a gaussian smooth to BW filtered: 
movie_smoothed = imgaussfilt3(movie_mod, 2);
figure;
for z = 1:data_dim(3)
    subplot(2,3,z);
    imagesc(movie_smoothed(:,:,z)); colormap gray;
    Z_info = "z = " + num2str(z);
    title(Z_info);
end
pause(1);

% Re-apply BW thresholding on smoothed: 
fc_min = min(movie_smoothed(movie_smoothed ~= 0), [], 'all', 'omitnan');
fc_med = median(movie_smoothed(movie_smoothed ~= 0), 'omitnan');
fc_max = max(movie_smoothed(movie_smoothed ~= 0), [], 'all', 'omitnan');
fc_low = fc_min + 0.5*(fc_med-fc_min);
fc_upp = fc_med + 0.3*(fc_max-fc_med);

movie_filtered = movie_smoothed;
movie_filtered(movie_filtered < fc_med)=0;
movie_filtered(movie_filtered > fc_upp)=1;

figure;
for z = 1:data_dim(3)
    subplot(2,3,z);
    imagesc(movie_filtered(:,:,z)); colormap gray;
    Z_info = "z = " + num2str(z);
    title(Z_info);
end
pause(1);

% Everything beneath this line was added by Jared Roth
%% Apply strict threshold
movie_filtered(movie_filtered < 1)=0;

% Display figure
figure;
for z = 1:data_dim(3)
    subplot(2,3,z);
    imagesc(movie_filtered(:,:,z)); colormap gray;
    Z_info = "z = " + num2str(z);
    title(Z_info);
end
pause(1);

% Output data as a .mat file (MATLAB Data file)
save('post-threshold_data.mat', 'movie_filtered')