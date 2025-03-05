from ultralytics import YOLO

import numpy as np
import supervision as sv

def piece_detections(frame: np.ndarray, annotate=False) -> sv.Detections:
    """
    Returns the `sv.Detections` for pieces.
    
    Attributes:
        frame (np.ndarray): The image from which the detections has to be made.
        annotate (bool): Annotates bounding boxes and labels onto the frame in-place.
    """
    
    # detections
    piece_detection_model = YOLO("models/piece_detection_best.pt")
    
    piece_result = piece_detection_model(frame)[0]
    
    piece_detections = sv.Detections.from_ultralytics(piece_result)
    piece_detections = piece_detections.with_nms(threshold=0.5, class_agnostic=True)
    
    if annotate:
        box_annotator = sv.BoxAnnotator(
            color=sv.ColorPalette.from_hex(["#3e9600", "#009957", "#006294", "#c70000", "#3f0096", "#9e0074", "#95ff4a", "#5cffb8", "#40bfff", "#ff5c5c", "#8e3dff", "#ff3dcb"]),
            thickness=1
        )

        frame = box_annotator.annotate(scene=frame, detections=piece_detections)
        
        # labels
        label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(["#3e9600", "#009957", "#006294", "#c70000", "#3f0096", "#9e0074", "#95ff4a", "#5cffb8", "#40bfff", "#ff5c5c", "#8e3dff", "#ff3dcb"]),
            text_padding=1
        )
        
        labels = [
            f"{class_name} {confidence:0.2f}"
            for class_name, confidence
            in zip(piece_detections["class_name"], piece_detections.confidence)
        ]
        
        frame = label_annotator.annotate(scene=frame, detections=piece_detections, labels=labels)
    
    return piece_detections
    

def corner_keypoints(frame: np.ndarray, annotate=False) -> sv.KeyPoints:
    """
    Returns the `sv.KeyPoints` for corners.
    
    Attributes:
        frame (np.ndarray): The image from which the detections has to be made.
        annotate (bool): Annotates keypoints onto the frame in-place.
    """    
    
    corner_detection_model = YOLO("models/corner_detection_best.pt")
    
    corner_result = corner_detection_model(frame)[0]
    keypoints = sv.KeyPoints.from_ultralytics(corner_result)
    
    if keypoints.confidence is None:
        return None
    
    if annotate:
    
        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex("#FF0000"),
            radius=3
        )
        
        filter = keypoints.confidence[0] > 0.5
        frame_reference_points = keypoints.xy[0][filter]
        frame_reference_keypoints = sv.KeyPoints(xy=frame_reference_points[np.newaxis, ...])
        
        frame = vertex_annotator.annotate(scene=frame, key_points=frame_reference_keypoints)
    
    return frame_reference_points