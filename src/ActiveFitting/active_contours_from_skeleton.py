import sys
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

sys.path.append('src/')

from SkeletonFinding import skeleton_finder


def skeleton_and_raw(skeleton_file="src/data/skeleton_coords.csv",
                     raw_file="src/data/raw_data.mat",
                     raw_name="movie_t_7z"):
    """
    Returns:
        skeleton_df (Pandas DataFrame): Contains all the skeleton coordinates
        raw (3D numpy array): Contains the raw data for a single time point.
            For most of these images, this will be 150 pixels by 150 pixels by
                7 z-slices.
    """
    
    # Import raw data
    raw = skeleton_finder.import_data(raw_file, raw_name)

    # Import skeleton data
    try:
        skeleton_df = pd.read_csv(skeleton_file)
    except FileNotFoundError:
        print(f"Error: could not find the skeleton file: {skeleton_file}")
        return None, raw
    
    return skeleton_df, raw


def active_contour_fit(initial_fit, image):
    """
    This function takes an image and an initial fit and returns a "snake",
    which is the post-active_contour fit.

    Args:
        initial_fit (NumPy array of (N, 2)): This is the initial guess for
            the best fit to the image.
        image (2D NumPy array): This is the image we will fit to. It must
            already be in gray.
    
    Returns:
        NumPy array of (N, 2) after the fitting.
    """
    snake = active_contour(image, initial_fit, gamma=0.05)

    return snake


def is_closed_loop_heuristic(snake, threshold=1.5):
    """
    Checks if a given snake (a NumPy array of (N, 2) coordinates) forms a closed loop
    """
    NUM_OF_COORDS_INDEX = 0
    if snake.shape[NUM_OF_COORDS_INDEX] < 3:
        return False  # You need at least 3 points to form a loop
    
    start_point = snake[0]
    end_point = snake[-1]

    distance = euclidean(start_point, end_point)

    return distance < threshold


def multi_fitter(skeletons, raw_data):
    """
    This function manages the multiple active-contour calls.
    It will loop through each z-slice and then within each iteration loop through
    each region in that slice.
    
    Args:
        skeletons (Pandas DataFrame): This is the DataFrame containing the
            skeleton coordinates for all the z-slices.
        raw_data (3D NumPy array): This contains all the raw data.
            The 3rd index is the z-slices.
    
    Returns:
        snakes (Pandas DataFrame): contains all the snake coordinates for each
        z-slice and region ID.
    """

    # Initialize a DataFrame for the improved data
    col_names = ["X", "Y", "Z", "Region_id"]
    snakes = pd.DataFrame(columns=col_names)
    
    # Check if there are an unequal number of z-slices
    Z_SLICE_INDEX = 2
    if max(skeletons["Z"].values) != raw_data.shape[Z_SLICE_INDEX]:
        print(f"There seems to be an unequal number of z-slices between the",
              "skeletons and the raw data.")
    
    else:
        # Loop through each z-slice
        z_slices = max(skeletons["Z"].values)
        for i in range(z_slices):
            z = i + 1

            # Build the Boolean mask to extract only the data for this z-slice
            mask = skeletons['Z'] == z
            slice = skeletons[mask]

            # Loop through each region in this z-slice
            for region in range(max(slice["Region_id"])):
                region_id = region + 1

                # Isolate region coordinates
                region_slice = slice[slice["Region_id"] == region_id]
                region_coords = region_slice[["X", "Y"]].values

                # Check if this region is circular or not
                circular = is_closed_loop_heuristic(region_coords)

                # If it is, we need to mathematically close the loop by
                # adding the first point back onto the end
                if circular:
                    start_point = region_coords[0, :].reshape(1, -1)
                    initial_snake = np.vstack([region_coords, start_point])
                else:
                    initial_snake = region_coords
                
                # Perform the fit
                improved_fit = active_contour_fit(initial_snake, raw_data[:, :, i])

                # Make a temporary DataFrame and concatenate it onto the main list
                temp_df = pd.DataFrame(improved_fit, columns=["X", "Y"])
                temp_df["Z"] = z
                temp_df["Region_id"] = region_id
                # FIXME: Figure out this warning
                snakes = pd.concat([snakes, temp_df], ignore_index=True)

    return snakes


if __name__ == "__main__":

    # Import the skeleton data and the raw data
    skeleton_coords, raw_data = skeleton_and_raw("src/data/skeleton_coords.csv", "src/data/raw_data.mat")
    snakes = multi_fitter(skeleton_coords, raw_data)
