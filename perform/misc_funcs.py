"""Miscellaneous functions that do not have a clear place in other modules."""

import os


def mkdir_shallow(base_dir, new_dir_name):
    """Makes new directory, if it does not already exist.

    Just a helper function for handling path joining.

    Args:
        base_dir: Path to directory in which new directory is to be made.
        new_dir_name: Name of new directory to be created in base_dir.

    Returns:
        Path to new directory.
    """

    new_dir = os.path.join(base_dir, new_dir_name)
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    return new_dir
