import numpy as np


def transform_to_frame(matrix, frame):
    """Given a transform defined by its matrix in a particular coordinate frame,
    compute its matrix under a given coordinate frame transformation.

    Parameters
    -------------
    matrix : (4, 4) float
      Homogeneous transform matrix

    frame : (4, 4) float
      Homogeneous coordinate frame transformation matrix

    Returns
    --------------
    transformed_matrix : (4, 4) float
      Homogeneous transform matrix for a given frame
    """
    matrix = np.asanyarray(matrix, dtype=np.float64)
    frame = np.asanyarray(frame, dtype=np.float64)

    return frame @ matrix @ np.linalg.inv(frame)
