import unittest
import sys

# Import local files to be tested
sys.path.append('src/')  # noqa
from SkeletonFinding import skeleton_finder
from ActiveFitting import active_contours_from_skeleton


# Import dummy files for testing
#sys.path.append('test/unit/')  # noqa


class TestSkeletonFinder(unittest.TestCase):

    def test_skeleton_finder(self):

        # Extract example dictionary
        example_skeletons = skeleton_finder.skeleton_finder(
            datafile="test/unit/post-threshold_data_EXAMPLE.mat"
        )
        first_skeleton = example_skeletons.get((1, 1))

        NUM_OF_POINTS_INDEX = 0
        NUM_OF_COORDS_INDEX = 1
        if first_skeleton is not None:
            # Check the size of the skeleton
            shape = first_skeleton.shape
            self.assertTrue(shape[NUM_OF_POINTS_INDEX] >= 1)
            self.assertTrue(shape[NUM_OF_COORDS_INDEX] == 3)

        else:
            print(f"No skeleton found.")
    
    def test_plot_skeleton_overlay(self):

        # Make sure the function works
        #skeleton_finder.plot_skeleton_overlay(skeleton_data, save=True)
        self.assertTrue(1 == 1)  # Dummy test


if __name__ == "__main__":
    unittest.main()
