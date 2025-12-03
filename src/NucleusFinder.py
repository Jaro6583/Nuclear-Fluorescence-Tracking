import scipy.io as sio
import numpy as np
import os
from skimage.filters import threshold_multiotsu
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure
from skimage.morphology import skeletonize
from skimage.segmentation import active_contour
from scipy.spatial.distance import euclidean
from skimage.morphology import binary_opening, disk, remove_small_objects
from skimage.measure import find_contours


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

        # Skeleton attributes
        self.skeletons = {}
        self.skeletons_df = None

        # Snake attribute
        self.snakes_df = None

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

    def run_thresholding(self,
                         time_index=25,
                         channel_index=1,
                         smooth_sigma=1.0,
                         binary_smooth_size=1,
                         min_size=15,
                         max_size=800):
        """
        Takes the loaded 5D data, processes a single time point using
        multi-Otsu thresholding (3 classes) on each individual z-slice,
        then stacks them back together.

        Args:
            time_index (int): The 0-based time index to analyze
                (ex: 25 for t=26)
            channel_index (int): The 0-based chanel index to analyze
            smooth_sigma (float): The sigma for the 3D Gaussian blur applied
                before thresholding.
            binary_smooth_size (int): Radius of the disk used for binary
                opening. Higher values smooth the edge more.
            min_size (int): The smallest allowable object size (in pixels) for
                a single z-slice. Regions smaller than this are removed.
            max_size (int): Largest allowable object. Set to 'None' to disable
        """
        if self.movie_ch2 is None:
            print("\nError: No data loaded. Call .load_mat_data() first.")
            return

        print(f"\nStarting thresholding for time index {time_index},",
              f"channel {channel_index}...")
        self.processed_time_index = time_index

        # Get 3D data for one time point
        try:
            time_snip = self.movie_ch2[:, :, :, time_index, channel_index]
            # Save the raw slice for later
            self.raw_data_t = time_snip
        except IndexError:
            print(f"Error: Time index {time_index} or channel index",
                  f"{channel_index} is out of bounds for data with shape",
                  f"{self.movie_ch2.shape}")
            return

        # Loop through each Z-slice, apply 2D threshold, and collect results
        all_slice_masks = []
        num_z_slices = self.raw_data_t.shape[2]

        for i in range(num_z_slices):
            slice_data = self.raw_data_t[:, :, i]

            # Apply a 2D smooth
            if smooth_sigma > 0:
                slice_smoothed = gaussian_filter(slice_data,
                                                 sigma=smooth_sigma,
                                                 mode='reflect')
            else:
                slice_smoothed = slice_data

            # Calculate the best threshold using Otsu's multi method
            try:
                thresholds = threshold_multiotsu(slice_smoothed, classes=3)
                nucleus_threshold = thresholds[1]

                # Apply the upper threshold
                slice_mask = slice_smoothed > nucleus_threshold

                # Smooth the mask
                if binary_smooth_size > 0:
                    footprint = disk(binary_smooth_size)
                    slice_mask = binary_opening(slice_mask, footprint)
                
                # Remove small regions
                if min_size > 0:
                    slice_mask = remove_small_objects(slice_mask,
                                                      min_size=min_size)
                
                # Remove large regions
                # This is mostly just to catch if the entire window passed the threshold
                if max_size is not None:
                    # Label connected regions
                    labeled_slice = measure.label(slice_mask)

                    # Iterate through each region
                    for region in measure.regionprops(labeled_slice):
                        if region.area > max_size:
                            slice_mask[labeled_slice == region.label] = False

                all_slice_masks.append(slice_mask)

            except ValueError:
                print("Warning: Otsu thresholding failed."
                      "Image may be all-zero.")
                self.post_threshold_data_t = np.zeros_like(self.raw_data_t,
                                                           dtype=bool)
                return
            except Exception as e:
                print(f"Warning: Thresholding failed for z-slice {i+1}:",
                      f"{e}. Appending empty mask.")
                all_slice_masks.append(np.zeros_like(slice_data, dtype=bool))

        # Stack the list of 2D masks back into a 3D (X, Y, Z) array
        self.post_threshold_data_t = np.stack(all_slice_masks, axis=2)
        print(f"Thresholding complete.\n")

    def find_skeletons(self):
        """
        Finds the skeletons from the 3D thresholded mask.

        This method populates 'self.skeletons' with a dictionary:
            Keys: (z_slice, region_id)
            Values: (N, 3) numpy array of (X, Y, Z) coordinates
        """
        if self.post_threshold_data_t is None:
            print("Error: No thresholded data found.")
            print("Please run .run_thresholding() first")
            return

        print("\nFinding skeletons in thresholded data...")
        data = self.post_threshold_data_t
        self.skeletons = {}  # Clear any previous skeletons

        # Initialize a list for the DataFrame
        data_list = []

        # Loop through each z-slice
        Z_INDEX = 2
        num_z_slices = data.shape[Z_INDEX]
        for i in range(num_z_slices):
            z = i + 1
            image_slice = data[:, :, i]

            # Label the data (find disconnected regions)
            labeled_slice = measure.label(image_slice, connectivity=2)
            num_objects = labeled_slice.max()

            # Loop through each region in this slice
            for region_label_ID in range(1, num_objects + 1):

                # Isolate a single region
                region = (labeled_slice == region_label_ID)

                # Find skeleton
                skeleton = skeletonize(region)

                # Collect coordinates
                rows, cols = np.nonzero(skeleton)

                if rows.size > 0:
                    # Create (X, Y, Z) coordinates
                    z_coords = np.full(rows.shape, z)
                    coords = np.column_stack((cols, rows, z_coords))
                    self.skeletons[(z, region_label_ID)] = coords

                    # Add to DataFrame list
                    temp_df = pd.DataFrame(coords, columns=["X", "Y", "Z"])
                    temp_df['Region_id'] = region_label_ID
                    data_list.append(temp_df)
                else:
                    self.skeletons[(z, region_label_ID)] = np.array([])

        # Create the final DataFrame
        if data_list:
            self.skeletons_df = pd.concat(data_list, ignore_index=True)
        else:
            self.skeletons_df = pd.DataFrame(
                columns=["X", "Y", "Z", "Region_id"]
            )

        print(f"Skeleton finding complete.",
              f"Found {len(self.skeletons)} regions.\n")

    def find_boundaries(self):
        """
        Finds the outer perimeter (contours) of the thresholded mask.
        """
        if self.post_threshold_data_t is None:
            print("Error: No thresholded data found.")
            print("Please run .run_thresholding() first.")
            return
        
        print("\nFinding boundaries (contours) in thresholded data...")
        data = self.post_threshold_data_t
        self.skeletons = {}

        data_list = []

        # Loop through each z-slice
        num_z_slices = data.shape[2]

        for i in range(num_z_slices):
            z = i + 1
            image_slice = data[:, :, i]

            # Label the data to separate distinct nuclei
            labeled_slice = measure.label(image_slice, connectivity=2)
            num_objects = labeled_slice.max()

            for region_label_ID in range(1, num_objects + 1):
                # Isolate the specific object mask
                region_mask = (labeled_slice == region_label_ID)
                # Find contours. level=0.5 finds the boundary between 0 and 1.
                contours = find_contours(region_mask, level=0.5)

                if contours:
                    # If multiple contours found (e.g. holes), take the
                    # longest one.
                    longest_contour = max(contours, key=len)

                    # find_contours returns (row, col) -> (Y, X).
                    # We need to swap these to match convention.
                    coords = longest_contour[:, [1, 0]]

                    # Add z coordinate column
                    z_coords = np.full((coords.shape[0], 1), z)
                    full_coords = np.hstack((coords, z_coords)) # Now (X,Y,Z)

                    # Store in dict
                    self.skeletons[(z, region_label_ID)] = full_coords

                    # Store in DataFrame list
                    temp_df = pd.DataFrame(coords, columns=["X", "Y"])
                    temp_df["Z"] = z
                    temp_df['Region_id'] = region_label_ID
                    data_list.append(temp_df)
                else:
                    self.skeletons[(z, region_label_ID)] = np.array([])
        
        # Create the final dataframe
        if data_list:
            self.skeletons_df = pd.concat(data_list, ignore_index=True)
        else:
            self.skeletons_df = pd.DataFrame(
                columns=["X", "Y", "Z", "Region_id"]
            )
        
        print(f"Boundary finding complete. Found {len(self.skeletons)} regions.\n")

    def print_info(self):
        """
        A helper function to print the status of the loaded data.
        FIXME: Decide what the useful things to print are.
        """

        print(f"\n|--- Info for {self.filename} ---")

        if self.movie_ch2 is not None:
            print(f"|4D data shape ('cell3D'): {self.movie_ch2.shape}")
        else:
            print("|4D Data: Not loaded. Call .load_mat_data() first.")

        if self.raw_data_t is not None:
            print(f"|3D Raw Slice Shape: {self.raw_data_t.shape}")
        else:
            print(f"|3D Raw Slice: Not processed.")

        if self.post_threshold_data_t is not None:
            print(f"|3D Thresholding Slice Shape:",
                  f"{self.post_threshold_data_t.shape}")
        else:
            print(f"|3D Thresholding Slice: Not processed.")

        if self.skeletons:
            print(f"|Skeletons found for {len(self.skeletons)} regions.")
        else:
            print(f"|Skeletons: Not processed.")

        if self.snakes_df is not None:
            print(f"|Active Contours fit for",
                  f"{len(self.snakes_df["Region_id"].unique())} regions.")
        else:
            print(f'|Active Contours: Not processed.')

        print("|----------------------------------------------------\n")

    def _plot_3d_data(self, data_3d, title_prefix, save_path=False):
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
            ax.invert_yaxis()
            if not is_binary:
                fig.colorbar(im, ax=ax, shrink=0.8)

        # Hide any unused subplots
        for j in range(num_z_to_plot, nrows * ncols):
            axes[j].axis['off']

        # Get the time index that was used for this data
        time_info = "??"
        if self.processed_time_index is not None:
            time_info = f"(time index = {self.processed_time_index})"

        # Final formatting
        fig.suptitle(f"{title_prefix} {time_info}\n(File: {self.filename})",
                     fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3.0)

        # Save figure (if requested)
        if save_path:
            try:
                plt.savefig(save_path)
                print(f"Saved plot to {save_path}\n")
            except Exception as e:
                print(f"Failed to save plot: {e}\n")

        # Display plot
        plt.show()

    def plot_raw_data(self, save_path=False):
        """
        Generates a plot of the raw 3D data for the selected time point
        (first 6 z-slices).
        """
        print("Plotting raw data (see figure)")
        self._plot_3d_data(self.raw_data_t,
                           "Raw Data (Pre-Thresholding)",
                           save_path=save_path)

    def plot_thresholded_data(self, save_path=False):
        """
        Generates a plot of the final 3D thresholded mask for the selected
        time point (first 6 z-slices).
        """
        print("Plotting thresholded data (see figure)")
        self._plot_3d_data(self.post_threshold_data_t,
                           "Post-Thresholding Mask",
                           save_path=save_path)

    def plot_skeleton_overlay(self, legend=False, save_path=None):
        """
        Plotting function that overlays the skeletons onto the raw data.
        Saving is optional.
        """

        if self.raw_data_t is None:
            print("Error: Missing raw data.",
                  "Please run .run_thresholding() first.")
            return
        if not self.skeletons:
            print("Error: Missing skeletons.",
                  "Please run .find_skeletons() first.")
            return

        print("Plotting skeleton overlays (see figure)")

        raw_data = self.raw_data_t
        num_z_slices = min(raw_data.shape[2], 6)  # Plot 6 slices max
        nrows = 2
        ncols = 3

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4, nrows * 3.5),
                                 squeeze=False)
        axes = axes.flatten()

        # Set up colors
        cmap = plt.get_cmap('tab10')
        color_list = cmap.colors
        num_colors = len(color_list)

        # Loop through each z-slice
        for i in range(num_z_slices):
            ax = axes[i]
            z = i + 1

            # Plot raw data
            ax.imshow(raw_data[:, :, i], cmap='gray', alpha=1)

            # Find skeletons in this slice
            for key, coords in self.skeletons.items():
                slice_z, region_id = key

                # Check if this skeleton belongs to this subplot
                if slice_z == z and coords.size > 0:
                    color_index = (region_id - 1) % num_colors

                    # Plot X (coords[:, 0]) and Y (coords[:, 1])
                    ax.scatter(coords[:, 0], coords[:, 1], s=1,
                               color=color_list[color_index],
                               label=f"Region {region_id}")

            ax.set_title(f"Z-slice {z}")
            ax.invert_yaxis()
            if legend:
                ax.legend(fontsize='small')

        # Clean up extra axes
        for j in range(num_z_slices, len(axes)):
            axes[j].axis('off')

        big_title = f"Skeleton Overlays "
        big_title += f"(time index = {self.processed_time_index})\n"
        big_title += f"(File: {self.filename})"
        fig.suptitle(big_title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3.0)

        if save_path:
            try:
                plt.savefig(save_path)
                print(f"Saved skeleton overlay plot to {save_path}")
            except Exception as e:
                print(f"Failed to save plot: {e}")

        plt.show()

    def _is_closed_loop(self, snake_coords, threshold=1.5):
        """
        Checks if a given snake (a numpy array of (N, 2) coordinates)
        forms a closed loop using a distance huristic.
        """
        if snake_coords.shape[0] < 3:
            return False  # Need at least 3 points

        start_point = snake_coords[0]
        end_point = snake_coords[-1]
        distance = euclidean(start_point, end_point)

        return distance < threshold

    def fit_active_contours(self,
                            alpha=0.01,
                            beta=10.0,
                            gamma=0.05,
                            w_line=1.0,
                            w_edge=0.0,
                            max_num_iter=500,
                            closed_loop_threshold=1.5,
                            snake_blur_sigma=2.0):
        """
        Fits active contours ('snakes') to the skeletons for the currently
        processed time slice.

        Reads from 'self.skeletons_df' and 'self.raw_data_t'.
        Saves results to 'self.snakes_df'.

        Args:
            gamma (float): Active contour parameter. Higher values make the
                contour more rigid.
            closed_loop_threshold (float): The distance (in pixels) to use
                for the closed-loop heuristic.
            snake_blur_sigma (float): Sigma for Gaussian blur applied to the
                raw image before active fitting.
            beta (float): Rigidity parameter. Higher values make the snake
                less flexible and more rigid.
        """
        if self.raw_data_t is None:
            print("Error: Missing raw data.",
                  "Please run .run_thresholding() first.")
            return

        if self.skeletons_df is None:
            print("Error: Missing skeleton data.",
                  "Please run .find_skeletons() first.")
            return

        print(f"\nFitting active contours (gamma={gamma})...")

        all_snakes_data = []
        all_processed_slices = []
        skeletons = self.skeletons_df

        # Loop through each Z-slice in the skeleton data
        for z_slice in skeletons["Z"].unique():

            # Remember z is 1-based but index is 0-based
            image_slice_raw = self.raw_data_t[:, :, z_slice - 1]

            # Blur the raw image to create a smooth "force field"
            if snake_blur_sigma > 0:
                image_slice_blurred = gaussian_filter(image_slice_raw,
                                                      sigma=snake_blur_sigma,
                                                      mode="reflect")
            else:
                image_slice_blurred = image_slice_raw

            # Normalize the image
            img_min = image_slice_blurred.min()
            img_max = image_slice_blurred.max()
            if img_max > img_min:
                numer = image_slice_blurred - img_min
                denom = img_max - img_min
                image_slice_norm = numer / denom
            else:
                image_slice_norm = image_slice_blurred  # Handle blank images

            # Add this normalized slice to the processed list
            all_processed_slices.append(image_slice_norm)

            # Get all skeletons for this slice
            slice_skeletons = skeletons[skeletons["Z"] == z_slice]

            # Loop through each unique region in this slice
            for region_id in slice_skeletons["Region_id"].unique():

                # Isolate region coordinates
                region_coords = slice_skeletons[
                    slice_skeletons["Region_id"] == region_id
                ][["X", "Y"]].values

                if region_coords.shape[0] < 3:
                    continue  # Not enough points to fit

                # Fix row/col issue
                snake_input_rc = region_coords[:, [1, 0]]
                
                if np.allclose(snake_input_rc[0], snake_input_rc[-1]):
                    initial_snake = snake_input_rc[:-1]
                else:
                    initial_snake = snake_input_rc

                # Perform the fit
                improved_fit = active_contour(
                    image_slice_norm,
                    initial_snake,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    w_line=w_line,
                    w_edge=w_edge,
                    boundary_condition="periodic",
                    max_num_iter=max_num_iter
                )

                # Swap back
                improved_fit = improved_fit[:, [1, 0]]

                # Make a temporary DataFrame and add it to our list
                temp_df = pd.DataFrame(improved_fit, columns=["X", "Y"])
                temp_df["Z"] = z_slice
                temp_df["Region_id"] = region_id
                all_snakes_data.append(temp_df)

        if not all_snakes_data:
            print("Warning: No snakes were fit.")
            self.snakes_df = pd.DataFrame(columns=["X", "Y", "Z", "Region_id"])
        else:
            self.snakes_df = pd.concat(all_snakes_data, ignore_index=True)

        # Save the processed background
        self.processed_image_t = np.stack(all_processed_slices, axis=2)

        print("Active contour fitting complete.\n")
        print(f"No snakes were harmed in the fitting of these data.")

    def plot_snake_overlay(self,
                           background='raw',
                           legend=False,
                           save_path=None):
        """
        Plotting function that overlays the fitted snakes onto the raw data.
        Saving is optional.

        Args:
            background (str): 'raw' (default) or 'processed'.
                Determines the background image.
            legend (bool): Add legend to the plots or not.
            save_path (str): A provided path will save the image at the
                specified path. Leaving this blank will not save the image.
        """
        if self.snakes_df is None:
            print("Error: Missing snake fits.",
                  "Please run .fit_active_contours() first.")
            return

        # Select background data
        if background == 'raw':
            bg_data = self.raw_data_t
            bg_cmap = 'gray'
            title_bg = "Raw Data"
        elif background == 'processed':
            bg_data = self.processed_image_t
            bg_cmap = 'gray'
            title_bg = "Processed (Blurred/Normalized) Data"
        else:
            print("Error: background must be 'raw' or 'processed',",
                  f"not {background}.")
            return

        if bg_data is None:
            print(f"Error: Background data for '{background}'",
                  "is not available.")
            return

        print("Plotting snake overlays (see figure).")

        # Initialize sub-plots
        num_z_slices = min(bg_data.shape[2], 6)
        nrows = 2
        ncols = 3

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4, nrows * 3.5),
                                 squeeze=False)
        axes = axes.flatten()

        # Set up color options
        cmap = plt.get_cmap('tab10')
        color_list = cmap.colors
        num_colors = len(color_list)

        # Loop through each Z-slice
        for i in range(num_z_slices):
            ax = axes[i]
            z = i + 1
            ax.imshow(bg_data[:, :, i], cmap=bg_cmap, alpha=1)

            # Find the snakes within this slice and loop through them
            slice_snakes = self.snakes_df[self.snakes_df["Z"] == z]
            for region_id in slice_snakes["Region_id"].unique():
                snake_coords = slice_snakes[
                    slice_snakes["Region_id"] == region_id
                ][["X", "Y"]].values

                color_index = (region_id - 1) % num_colors

                # Plot snakes
                ax.scatter(snake_coords[:, 0],
                           snake_coords[:, 1],
                           color=color_list[color_index],
                           s=1.0,
                           label=f"Region {region_id}")

            ax.set_title(f"Z-slice {z}")
            ax.invert_yaxis()
            if legend:
                ax.legend(fontsize='small')

        # Clean up unused plots
        for j in range(num_z_slices, len(axes)):
            axes[j].axis('off')

        big_title = f"Active Contour (Snake) Overlays "
        big_title += f"(time_index = {self.processed_time_index})\n"
        big_title += f"(File: {self.filename})"
        fig.suptitle(big_title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3.0)

        if save_path:
            try:
                plt.savefig(save_path)
                print(f"Saved snake overlay plot to {save_path}")
            except Exception as e:
                print(f'Failed to save plot: {e}')

        plt.show()


if __name__ == "__main__":

    # Define cell path
    CELL_PATH = r"C:\Users\jared\Desktop\Research\Nuclear-Fluorescence-"
    CELL_PATH += r"Tracking\src\data\MB1411\Segmented Cells"
    CELL_TYPE = "1411_200R_150G_q25s_25deg"
    CELL_NUMBER = "010_1"

    # Build the full filepath
    full_path = os.path.join(CELL_PATH, f"{CELL_TYPE}_{CELL_NUMBER}.mat")

    # Create an instance of the class and load data
    my_cell = SegmentedCell(full_path)
    my_cell.load_mat_data()

    # Run thresholding for a single time index
    time = 30
    my_cell.run_thresholding(time_index=time, min_size=60)

    # Plot raw and thresholded data (save)
    my_cell.plot_raw_data(save_path="src/figures/raw_data.png")
    my_cell.plot_thresholded_data(save_path="src/figures/thresholded_data.png")

    # Find skeletons from the threshold mask
    #my_cell.find_skeletons()
    my_cell.find_boundaries()

    # Plot skeleton overlays
    my_cell.plot_skeleton_overlay(save_path="src/figures/skeleton_overlay.png")

    # Fit active contours (Snakes)
    my_cell.fit_active_contours(
        alpha=0.01,
        beta=40.0,
        w_line=10.0,
        w_edge=0.0,
        snake_blur_sigma=1.5,
        max_num_iter=5
    )

    # Plot snakes
    my_cell.plot_snake_overlay(background='processed',
                               save_path="src/figures/snake_overlay.png")

    # Print final info
    my_cell.print_info()
