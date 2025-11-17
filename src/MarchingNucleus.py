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
from skimage.morphology import binary_opening, disk
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.morphology import remove_small_objects


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

        # 3D Attributes
        self.mesh_verts = None
        self.mesh_faces = None

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

    def run_thresholding(self, time_index=25, channel_index=1,
                         smooth_sigma=1.0, binary_smooth_size=1):
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

    def generate_3d_surface(self, step_size=1, mesh_smooth_sigma=1.0):
        """
        Generates a 3D surface mesh from the binary mask using Marching Cubes.

        Args:
            step_size (int): Step size in voxels. Larger values (e.g. 2)
                generate coarser meshes but run faster.
            mesh_smooth_sigma (float): Amount of smoothing to apply to the
                mask before generating the mesh.
        """
        if self.post_threshold_data_t is None:
            print("Error: Missing thresholded data.",
                  "Run .run_thresholding() first.")
            return
        
        print("\nGenerating 3D surface mesh (Marching Cubes)...")

        # Get the spacing from the metadata (if available)
        if self.voxel_size is not None:
            v_size = np.array(self.voxel_size).flatten()
            spacing = (v_size[1], v_size[0], v_size[2])  # FIXME: remove magic nums
        else:
            spacing = (1.0, 1.0, 1.0)
            print("Warning: No voxel size found. Assuming isotropic (1.0, 1.0, 1.0)")
        
        # Smooth the binary mask before we mesh it
        # Note that this is a 3D-smoothing. It accounts for the z-slice data
        # above and below each region.
        if mesh_smooth_sigma > 0:
            mask_float = self.post_threshold_data_t.astype(float)
            volume_to_mesh = gaussian_filter(mask_float, sigma=mesh_smooth_sigma)
        else:
            volume_to_mesh = self.post_threshold_data_t
        
        try:
            # Run marching cubes
            verts, faces, normals, values = marching_cubes(
                volume_to_mesh,
                level=0.5,
                spacing=spacing,
                step_size=step_size,
                allow_degenerate=False
            )
            
            self.mesh_verts = verts
            self.mesh_faces = faces
            print(f"3D Surface generated: {len(verts)} vertices, {len(faces)} faces.")
        
        except RuntimeError:
            print("Error: Could not generate surface. Mask might be empty.")

        except Exception as e:
            print(f"An error has occurred during Marching Cubes: {e}")
    
    def clean_mask(self, min_size=50):
        """
        Sweeps through the binary mask to remove small, disconnected floating blobs.

        Args:
            min_size (int): The smallest allowable object size (in voxels).
                Any blob smaller than this is deleted.
        """
        if self.post_threshold_data_t is None:
            print("Error: Missing thresholded data. Run .run_thresholding() first.")
            return
        
        print(f"Cleaning mask (removing blobs < {min_size} voxels)...")

        # Convert to a boolean array
        mask_bool = self.post_threshold_data_t.astype(bool)

        # Clean
        # Note that this is a 3D cleaning. It accounts for the z-slice above
        # and below the region of interest.
        cleaned_mask = remove_small_objects(mask_bool, min_size=min_size)

        self.post_threshold_data_t = cleaned_mask
        print("Mask cleaned.")
    
    def _plot_slices(self, data_3d, title_prefix, save_path=False):
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
        self._plot_slices(self.raw_data_t,
                           "Raw Data (Pre-Thresholding)",
                           save_path=save_path)

    def plot_thresholded_data(self, save_path=False):
        """
        Generates a plot of the final 3D thresholded mask for the selected
        time point (first 6 z-slices).
        """
        print("Plotting thresholded data (see figure)")
        self._plot_slices(self.post_threshold_data_t,
                           "Post-Thresholding Mask",
                           save_path=save_path)
    
    def plot_3d_surface(self, save_path=None):
        """
        Visualizes the generated 3D mesh using Matplotlib
        """
        if self.mesh_verts is None:
            print("Error: No mesh found. Run .generate_3d_surface() first.")
            return
        
        print("Plotting 3D surface...")

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create the mesh collection
        mesh = Poly3DCollection(self.mesh_verts[self.mesh_faces])
        mesh.set_alpha(0.5)
        mesh.set_edgecolor('k')
        mesh.set_linewidth(0.1)
        ax.add_collection3d(mesh)

        # Auto-scale the axes to fit the mesh
        verts = self.mesh_verts
        xlim = (verts[:, 0].min(), verts[:, 0].max())
        ylim = (verts[:, 1].min(), verts[:, 1].max())
        zlim = (verts[:, 2].min(), verts[:, 2].max())

        # Calculate the range (span) of each axis
        x_span = xlim[1] - xlim[0]
        y_span = ylim[1] - ylim[0]
        z_span = zlim[1] - zlim[0]

        # Find max span and the midpoints of the data
        max_span = max(x_span, y_span, z_span)
        x_mid = np.mean(xlim)
        y_mid = np.mean(ylim)
        z_mid = np.mean(zlim)

        # Set axis limits
        ax.set_xlim(x_mid - max_span / 2, x_mid + max_span / 2)
        ax.set_ylim(y_mid - max_span / 2, y_mid + max_span / 2)
        ax.set_zlim(z_mid - max_span / 2, z_mid + max_span / 2)

        # Label axes
        ax.set_xlabel("X (microns)")
        ax.set_ylabel("Y (microns)")
        ax.set_zlabel("Z (microns)")

        title = f"3D Reconstruction (t={self.processed_time_index})\n(File: {self.filename})"
        plt.title(title)

        if save_path:
            plt.savefig(save_path)
            print(f"Saved 3D plot to {save_path}")
        
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
    time = 10
    my_cell.run_thresholding(time_index=time)

    # Clean mask
    my_cell.clean_mask()

    # Plot raw and thresholded data (save)
    my_cell.plot_raw_data(save_path="src/figures/raw_data.png")
    my_cell.plot_thresholded_data(save_path="src/figures/thresholded_data.png")

    # Build 3D model and plot
    my_cell.generate_3d_surface()
    my_cell.plot_3d_surface(save_path="src/figures/3dplot")