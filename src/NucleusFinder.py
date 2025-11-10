import scipy.io as sio
import numpy as np
import os
from skimage.filters import gaussian
from scipy.ndimage import gaussian_filter


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

    
    def run_thresholding(self, time_index=25, channel_index=1):
        """
        Takes the loaded 5D data, processes a single time point, and saves
        the results into object attributes.

        Args:
            time_index (int): The 0-based time index to analyze (ex: 25 for t=26)
            channel_index (int): The 0-based chanel index to analyze
        """
        if self.movie_ch2 is None:
            print("Error: No data loaded. Call .load_mat_data() first.")
            return
        
        print(f"Starting thresholding for time index {time_index}, channel {channel_index}...")

        # Get data for one time point
        try:
            movie_t_7z = self.movie_ch2[:, :, :, time_index, channel_index]
            # Save the raw slice for later
            self.raw_data_t = movie_t_7z
        except IndexError:
            print(f"Error: Time index {time_index} or channel index {channel_index} is out of bounds for data with shape {self.movie_ch2.shape}")
            return
        
        # Perform initial thresholding
        non_zero_data = movie_t_7z[movie_t_7z != 0]

        # Check if non_zero_data is empty after all
        if non_zero_data.size == 0:
            print(f"Warning: No non-zero data found at t={time_index}, channel={channel_index}.")
            self.post_threshold_data_t = np.zeros_like(movie_t_7z)
            return

        c_min = np.nanmin(non_zero_data)
        c_med = np.nanmedian(non_zero_data)
        c_max = np.nanmax(non_zero_data)

        c_upp = c_med + 0.05 * (c_max - c_med)

        movie_mod = np.zeros_like(movie_t_7z)
        movie_mod[movie_mod > c_upp] = 1

        # Apply gaussian smooth
        movie_smoothed = gaussian_filter(movie_mod, sigma=2, mode='reflect')

        # Re-apply BW thresholding
        non_zero_smoothed = movie_smoothed[movie_smoothed != 0]

        if non_zero_smoothed.size == 0:
            print(f"Warning: Thresholding (step 1) resulted in an all-zero image. Aborting.")
            self.post_threshold_data_t = np.zeros_like(movie_smoothed)
            return

        fc_med = np.nanmedian(non_zero_smoothed)
        fc_max = np.nanmax(non_zero_smoothed)

        fc_upp = fc_med + 0.3 * (fc_max - fc_med)

        movie_filtered = np.zeros_like(movie_smoothed)
        movie_filtered[movie_smoothed < fc_med] = 0
        movie_filtered[movie_smoothed > fc_upp] = 1  # Typo? Should be '> fc_med' ?

        # Apply strict threshold
        movie_filtered[movie_filtered < 1] = 0

        # Save result
        self.post_threshold_data_t = movie_filtered
        print("Thresholding complete.")

    
    def print_info(self):
        """
        A helper function to print the status of the loaded data.
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
        
        print("-------------------------------------------")


if __name__ == "__main__":

    # Define cell path
    CELL_PATH = r"C:\Users\jared\Desktop\Research\Nuclear-Fluorescence-Tracking\src\data\MB1411\Segmented Cells"
    CELL_TYPE = "1411_200R_150G_q25s_25deg"
    CELL_NUMBER = "010_2"

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

    # Now access full 4D data
    if my_cell.post_threshold_data_t is not None:
        print("\n--- Accessing data ---")
        print(f"Raw data shape (from t=25): {my_cell.raw_data_t.shape}")
        print(f"Thresholded data shape (from t=25):",
              f"{my_cell.post_threshold_data_t.shape}")