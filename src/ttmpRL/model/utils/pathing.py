"""Helper functions for handling models."""

import os


def get_model_path(directory, increment=False):
    """Finds all the .weights.h5 files (neural net weights) and returns the path to
    the most trained version. If there are no models, returns a default model
    name (model-0.weights.h5)

    Parameters:
        directory: str. Directory path in which the .weights.h5 files are contained

    Returns:
        path: str. Path to the file (directory+'/model-newest.weights.h5')
    """

    models = [f for f in os.listdir(directory) if ("model-" in f and f.endswith("h5"))]

    if len(models) > 0:
        max_v = int(max([m.split("-")[1].split(".")[0] for m in models]))
        if increment:
            max_v += 1
    else:
        max_v = 0

    path = os.path.join(directory, f"model-{max_v}.weights.h5")

    return path
