"""
This file contains IO utility functions.
"""

import os

@staticmethod
def prepare_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory