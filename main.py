import cv2
import argparse

from ultralytics import YOLO
import supervision as sv

from detection_helpers import piece_detections, corner_keypoints
from board_helpers import draw_board, draw_points_on_board, BOARD_POINTS
from PerspectiveTransformer import PerspectiveTransformer

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
    
    while True:
        ret, frame = cap.read()
        # frame = cv2.imread("board.jpg")
        
        detections = piece_detections(
            model=piece_detection_model, 
            frame=frame, 
            annotate=True
        )
        keypoints = corner_keypoints(
            model=corner_detection_model, 
            frame=frame, 
            annotate=True
        )
        
        if keypoints is not None:
            
            transformer = PerspectiveTransformer(
                source=keypoints,  # dim is (1, 8, 2)
                target=BOARD_POINTS
            )
            
            frame_pieces_xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            board_pieces_xy = transformer.transform_points(points=frame_pieces_xy)
                            
            board = draw_board()
            board = draw_points_on_board(
                board=board,
                xy=board_pieces_xy,
                py=-10
            )
            cv2.imshow("board", board)
        
        cv2.imshow("frame", frame)
        # break
        
        if cv2.waitKey(20) == 27:
            break


if __name__ == "__main__":
    main()