import cv2
import argparse

from ultralytics import YOLO
import supervision as sv

import numpy as np

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="argeparse_desc")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    piece_detection_model = YOLO("models/piece_detection_best.pt")
    corner_detection_model = YOLO("models/corner_detection_best.pt")
    
    box_annotator = sv.BoxAnnotator(
        thickness=1
    )
    label_annotator = sv.LabelAnnotator(
        text_padding=1
    )
    
    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.from_hex("#FF0000"),
        radius=3
    )
    
    while True:
        ret, frame = cap.read()
        # frame = cv2.imread("board2.jpeg")
        
        piece_result = piece_detection_model(frame)[0]
        piece_detections = sv.Detections.from_ultralytics(piece_result)
        piece_detections = piece_detections.with_nms(threshold=0.5, class_agnostic=True)
        
        corner_result = corner_detection_model(frame)[0]
        keypoints = sv.KeyPoints.from_ultralytics(corner_result)
        
        labels = [
            f"{class_name} {confidence:0.2f}"
            for class_name, confidence
            in zip(piece_detections["class_name"], piece_detections.confidence)
        ]
        
        frame = box_annotator.annotate(scene=frame, detections=piece_detections)
        frame = label_annotator.annotate(scene=frame, detections=piece_detections, labels=labels)
        
        if keypoints.confidence is not None:
            filter = keypoints.confidence[0] > 0.5
            frame_reference_points = keypoints.xy[0][filter]
            frame_reference_keypoints = sv.KeyPoints(xy=frame_reference_points[np.newaxis, ...])
            
            frame = vertex_annotator.annotate(scene=frame, key_points=frame_reference_keypoints)
        
        cv2.imshow("frame", frame)
        # break
        
        if cv2.waitKey(30) == 27:
            break


if __name__ == "__main__":
    main()