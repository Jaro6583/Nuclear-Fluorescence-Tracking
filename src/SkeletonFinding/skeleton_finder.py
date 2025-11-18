import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.morphology import skeletonize
import pandas as pd


def plot_skeleton_overlay(skeletons, save=False, legend=False):

    # Bring in raw data
    try:
        raw = import_data(filename="src/data/raw_data.mat", name="movie_t_7z")
    except Exception as e:
        print(f"Couldn't import raw data: {e}")

    SLICE_INDEX = 2
    num_frames = raw.shape[SLICE_INDEX]

    # Set up plotting
    ncols = int(np.ceil(np.sqrt(num_frames)))
    nrows = min(int(np.ceil(num_frames / ncols)), 2)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 4),
                             squeeze=False)
    axes = axes.flatten()

    # Set up colors
    cmap = plt.get_cmap("tab10")
    color_list = cmap.colors
    num_colors = len(color_list)

    # Loop through each z-slice
    for i in range(ncols * nrows):
        ax = axes[i]
        z = i + 1

        # Plot raw data (background)
        ax.imshow(raw[:, :, i], cmap="gray", alpha=1)

        # Find skeletons in this slice
        for key, coords in skeletons.items():
            slice_z, region_id = key

            # Check if this skeleton belongs to this subplot
            if slice_z == z and coords.size > 0:

                # Grab a unique color for this region
                color_index = (region_id - 1) % num_colors

                # Plot skeleton
                ax.scatter(coords[:, 0], coords[:, 1], s=1,
                           color=color_list[color_index],
                           label=f"Region {region_id}")
                ax.invert_yaxis()

        ax.set_title(f"z = {z}")

        if legend:
            ax.legend(fontsize="small")

    # Clean up extra axes
    for j in range(num_frames, len(axes)):
        axes[j].axis('off')

    # Final edits and plot
    fig.suptitle(f"Skeleton overlays on raw data", fontsize=32)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save:
        plt.savefig("src/figures/skeleton_overlay_plot.png")

    plt.show()

    return None


def export_data(skeletons,
                output_filename="src/data/skeleton_coords.csv",
                save=True):
    """
    Converts a skeleton dictionary of coordinates into a pandas dataframe.
    Optionally export to a CSV.

    Args:
        skeletons (dict): Dictionary of keys (z_slice, region_id) and
            values of numpy arrays.
        output_filename (str): The name of the file that will be produced.
        save (bool): If True, saves the DataFrame to a CSV file.

    Returns:
        pandas.DataFrame: The complete dataframe of skeleton coordinates.
    """
    data_list = []

    for (z_slice, region_id), coords in skeletons.items():

        # Skip empty arrays from images with no regions
        if coords.size == 0:
            continue

        # Create a temporary DataFrame for this specific skeleton
        temp_df = pd.DataFrame(coords, columns=["X", "Y", "Z"])

        # Add metadata
        temp_df['Z'] = z_slice
        temp_df['Region_id'] = region_id

        data_list.append(temp_df)

    # Concatenate all data into a single DataFrame
    if data_list:
        final_df = pd.concat(data_list, ignore_index=True)
    else:
        # Handle the case where there are no skeletons
        print("Warning: An empty skeleton dictionary was submitted.")
        final_df = pd.DataFrame(['X', 'Y', 'Z'])

    # Save data
    if save:
        try:
            final_df.to_csv(output_filename, index=False)
            print(f"Successfully saved skeleton data to {output_filename}")
        except Exception as e:
            print(f"Failed to save to CSV: {e}")

    return final_df


def import_data(filename="src/data/post-threshold_data.mat",
                name="movie_filtered"):

    # Default (in case it can't find the file)
    data = np.zeros((3, 3, 3))

    # Load in data
    try:
        mat_contents = sio.loadmat(filename)
        data = mat_contents[name]
    except FileNotFoundError:
        print(f"File {filename} not found.",
              "Make sure you're running in the correct directory.")
    except Exception as e:
        print(f"An error has occurred while trying to open {filename}: {e}")

    return data


def skeleton_finder(datafile="src/data/post-threshold_data.mat"):
    """
    returns:
        all_skeletons (dict): A dictionary containing all the skeleton data.
            Keys are tuples of (z-slice, Region_id).
            Values are np arrays of size (N, 3) where N is the number of
                points in that region.
    """

    data = import_data(datafile)

    # Prepare dictionary
    all_skeletons = {}

    # Manage the different z-slices
    SLICE_INDEX = 2
    # This finds how many z-slices there are in these data
    num_frames = data.shape[SLICE_INDEX]

    for i in range(num_frames):
        z = i + 1

        # Grab image slice
        image_slice = data[:, :, i]

        # Label the data
        image_slice = measure.label(image_slice, connectivity=2)
        num_objects = image_slice.max()
        # print(f"{num_objects} objects found in frame {z}")

        # Loop through each region in this frame
        for region_label_ID in range(1, num_objects + 1):

            # Target a single region to skeletonize
            region = (image_slice == region_label_ID)
            skeleton = skeletonize(region)

            # Collect coordinates
            rows, cols = np.nonzero(skeleton)

            if rows.size > 0:
                z_coords = np.full(rows.shape, z)
                coords = np.column_stack((cols, rows, z_coords))
                all_skeletons[(z, region_label_ID)] = coords
            else:
                # Just in case no region is found in this slice
                all_skeletons[(z, region_label_ID)] = np.array([])

    return all_skeletons


if __name__ == "__main__":
    skeleton_data = skeleton_finder(
        datafile="src/data/post-threshold_data.mat"
    )
    # print(skeleton_data)
    # plot_skeleton_overlay(skeleton_data, save=True)
    export_data(skeleton_data)
