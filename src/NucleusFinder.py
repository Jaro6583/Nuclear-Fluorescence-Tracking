import scipy.io as sio
import numpy as np
import os
from skimage.filters import threshold_multiotsu
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


class SegmentedCell:
    """
    This class represents a single segmented cell, loaded from a .mat file
    and will contain all the methods needed to run the analysis pipeline.
    """

    def __init__(self, mat_filepath):
        """
        Initializes the cell object.

        Args:
            mat_filepath (str): The full path to the .mat file.
        """

        if not os.path.exists(mat_filepath):
            print(f"Warning: File not found at {mat_filepath}")
        
        self.filepath = mat_filepath
        self.filename = os.path.basename(mat_filepath)

        # 4D data from the .mat file
        self.movie_ch1 = None
        self.movie_ch2 = None
        self.voxel_size = None
        self.time_step = None

        # 3D data for a single time point
        self.raw_data_t = None
        self.post_threshold_data_t = None
        self.processed_time_index = None

        print(f"SegmentedCell object created for {self.filename}")

    def load_mat_data(self):
        """
        Loads and parses the data from the .mat file.
        """
        print(f"Loading data from {self.filename}...")
        try:
            # Load the .mat file
            mat_contents = sio.loadmat(self.filepath)
            cell_data_struct = mat_contents['cellData']
            data = cell_data_struct[0, 0]
            self.field_names = cell_data_struct.dtype.names

            # Load movie data
            if 'cell3D' in self.field_names:
                self.movie_ch2 = data['cell3D']
            if 'raw' in self.field_names:
                self.movie_ch1 = data['raw']

            # Load metadata
            if 'metaData' in self.field_names:
                meta_data = data['metaData'][0, 0]
                meta_data_fields = meta_data.dtype.names

                if 'voxel_size' in meta_data_fields:
                    self.voxel_size = meta_data['voxel_size']
                elif 'sizeVoxelsX' in meta_data_fields:
                    self.voxel_size = [
                        meta_data['sizeVoxelsX'],
                        meta_data['sizeVoxelsY'],
                        meta_data['sizeVoxelsZ']
                    ]
                
                if 'time_step' in meta_data_fields:
                    self.time_step = meta_data['time_step']
            
            print(f"Successfully loaded data for {self.filename}")

        except Exception as e:
            print(f"An error occurred while loading {self.filename}: {e}")

    
    def run_thresholding(self, time_index=25, channel_index=1, smooth_sigma=1.0):
        """
        Takes the loaded 5D data, processes a single time point using
        multi-Otsu thresholding (3 classes), and saves the result.

        Args:
            time_index (int): The 0-based time index to analyze
                (ex: 25 for t=26)
            channel_index (int): The 0-based chanel index to analyze
            smooth_sigma (float): The sigma for the 3D Gaussian blur applied
                before thresholding.
        """
        if self.movie_ch2 is None:
            print("Error: No data loaded. Call .load_mat_data() first.")
            return
        
        print(f"Starting thresholding for time index {time_index}, channel {channel_index}...")
        self.processed_time_index = time_index

        # Get data for one time point
        try:
            movie_t_7z = self.movie_ch2[:, :, :, time_index, channel_index]
            # Save the raw slice for later
            self.raw_data_t = movie_t_7z
        except IndexError:
            print(f"Error: Time index {time_index} or channel index {channel_index} is out of bounds for data with shape {self.movie_ch2.shape}")
            return

        # Smooth the raw 3D image to remove noise
        if smooth_sigma > 0:
            movie_smoothed = gaussian_filter(self.raw_data_t,
                                             sigma=smooth_sigma,
                                             mode='reflect')
        else:
            movie_smoothed = self.raw_data_t

        # Calculate the best threshold using Otsu's method (ignoring pixels of zero)
        try:
            thresholds = threshold_multiotsu(movie_smoothed, classes=3)
            nucleus_threshold = thresholds[1]

        except ValueError:
            print("Warning: Otsu thresholding failed. Image may be all-zero.")
            self.post_threshold_data_t = np.zeros_like(self.raw_data_t, dtype=bool)
            return

        # Apply the threshold to create a binary mask
        binary_mask = movie_smoothed > nucleus_threshold

        # Save the final mask
        self.post_threshold_data_t = binary_mask

        print(f"Thresholding complete. Otsu thresholds found at: BLANK FOR NOW")


    def print_info(self):
        """
        A helper function to print the status of the loaded data.
        FIXME: Decide what the useful things to print are.
        """

        print(f"\n--- Info for {self.filename} ---")

        if self.movie_ch2 is not None:
            print(f"4D data shape ('cell3D'): {self.movie_ch2.shape}")
        else:
            print("4D Data: Not loaded. Call .load_mat_data() first.")
        
        if self.raw_data_t is not None:
            print(f"3D Raw Slice Shape: {self.raw_data_t.shape}")
        else:
            print(f"3D Raw Slice: Not processed.")
        
        if self.post_threshold_data_t is not None:
            print(f"3D Thresholding Slice Shape: {self.post_threshold_data_t.shape}")
        else:
            print(f"3D Thresholding Slice: Not processed.")
        
        print("-------------------------------------------\n")


    def _plot_3d_data(self, data_3d, title_prefix):
        """
        Private helper method to plot the first 6 Z-slices of 3D data
        (X, Y, Z) in a 2x3 grid.
        """
        if data_3d is None:
            print(f"Error: No data available to plot for '{title_prefix}'")
            return
        
        Z_INDEX = 2
        num_z_available = data_3d.shape[Z_INDEX]
        if num_z_available == 0:
            print(f"Error: Data has zero Z-slices.")
            return
        
        # Plot the first 6 slices in a 2x3 grid
        num_z_to_plot = min(num_z_available, 6)
        nrows = 2
        ncols = 3

        fig, axes = plt.subplots(nrows,
                                 ncols,
                                 figsize=(ncols * 4, nrows * 3.5),
                                 squeeze=False)
        axes = axes.flatten()

        # Check if data is binary mask or grayscale
        is_binary = data_3d.dtype == bool
        cmap = 'gray' if not is_binary else 'viridis'

        for i in range(num_z_to_plot):
            ax = axes[i]
            slice_data = data_3d[:, :, i]
            im = ax.imshow(slice_data, cmap=cmap, interpolation='none')
            ax.set_title(f"Z-Slice {i + 1}")
            ax.axis('off')
            if not is_binary:
                fig.colorbar(im, ax=ax, shrink=0.8)
        
        # Hide any unused subplots
        for j in range(num_z_to_plot, nrows * ncols):
            axes[j].axis['off']
        
        # Get the time index that was used for this data
        time_info = ""
        if self.processed_time_index is not None:
            time_info = f"(t = {self.processed_time_index + 1})"
        
        fig.suptitle(f"{title_prefix} {time_info}\n(File: {self.filename})", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3.0)
        plt.show()


    def plot_raw_data(self):
        """
        Generates a plot of the raw 3D data for the selected time point
        (first 6 z-slices).
        """
        print("Plotting raw data...")
        self._plot_3d_data(self.raw_data_t, "Raw Data (Pre-Thresholding)")


    def plot_thresholded_data(self):
        """
        Generates a plot of the final 3D thresholded mask for the selected
        time point (first 6 z-slices).
        """
        print("Plotting thresholded data...")
        self._plot_3d_data(self.post_threshold_data_t, "Post-Thresholding Mask")


if __name__ == "__main__":

    # Define cell path
    CELL_PATH = r"C:\Users\jared\Desktop\Research\Nuclear-Fluorescence-Tracking\src\data\MB1411\Segmented Cells"
    CELL_TYPE = "1411_200R_150G_q25s_25deg"
    CELL_NUMBER = "010_1"

    # Build the full filepath
    full_path = os.path.join(CELL_PATH, f"{CELL_TYPE}_{CELL_NUMBER}.mat")

    # Create an instance of the class
    my_cell = SegmentedCell(full_path)

    # Load data and print
    my_cell.load_mat_data()
    my_cell.print_info()

    # Run thresholding for time index 25
    my_cell.run_thresholding(time_index=25)
    my_cell.print_info()

    # Plot data
    my_cell.plot_raw_data()
    my_cell.plot_thresholded_data()

    # Now access full 4D data
    if my_cell.post_threshold_data_t is not None:
        print("\n--- Accessing data ---")
        print(f"Raw data shape (from t=25): {my_cell.raw_data_t.shape}")
        print(f"Thresholded data shape (from t=25):",
              f"{my_cell.post_threshold_data_t.shape}")