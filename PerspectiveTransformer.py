import cv2
import numpy as np

class PerspectiveTransformer:
    def __init__(
        self,
        source: np.ndarray[np.float32],
        target: np.ndarray[np.float32]
    ):
        if source.shape != target.shape:
            print(source.shape)
            print(target.shape)
            raise ValueError("Source and target must be of same shape.")
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")
        
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m, _ = cv2.findHomography(source, target)
        if self.m is None:
            raise ValueError("Homography matrix could not be calculated.")
        
    def transform_points(
        self,
        points: np.ndarray[np.float32]
    ) -> np.ndarray[np.float32]:
        
        if points.size == 0:
            return points
        
        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")
        
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2).astype(np.float32)