import cv2
import argparse

from ultralytics import YOLO
import supervision as sv

from detection_helpers import piece_detections, corner_keypoints

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
    
    while True:
        # ret, frame = cap.read()
        frame = cv2.imread("board.jpg")
            
        detections = piece_detections(frame, annotate=True)
        keypoints = corner_keypoints(frame, annotate=True)
        
        cv2.imshow("frame", frame)
        # print(type(frame))
        # break
        
        if cv2.waitKey(30) == 27:
            break


if __name__ == "__main__":
    main()