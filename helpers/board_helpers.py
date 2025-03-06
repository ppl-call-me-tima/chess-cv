import cv2
import numpy as np
import supervision as sv

from PerspectiveTransformer import PerspectiveTransformer

BOARD_POINTS = np.array([
    (0, 0),
    (0, 800),
    (800, 800),
    (800, 0),

    (100, 100),
    (100, 700),
    (700, 700),
    (700, 100),
])

def draw_board():
    board_size = 8
    square_size = 100
    
    image_size = board_size * square_size
    board = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    
    for row in range(board_size):
        for col in range(board_size):
            
            # NOTE: it is a tiled chessboard, for side-view (top-left is black)
            # TODO: add logic for automatic detection of side of chess-board to auto orient
            
            if (row + col) % 2:
                color = (255, 255, 255)
            else:
                color = (0, 0, 0)
            
            top_left = (col * square_size, row * square_size)
            bottom_right = ((col + 1) * square_size, (row + 1) * square_size)
            
            cv2.rectangle(board, top_left, bottom_right, color, -1)
    
    return board


def draw_points_on_board(
    board: np.ndarray,
    xy: np.ndarray,
    px: int = 0,
    py: int = 0,
    scale: float = 1,
) -> np.ndarray:

    for i in range(len(xy)):

        point = xy[i]

        x = int(point[0] * scale) + px
        y = int(point[1] * scale) + py

        cv2.circle(
            board,
            (x,y),
            radius=2,
            color=(0,0,255),
            thickness=-1
        )

    return board


def generate_board(board: np.ndarray[np.float32], detections: sv.Detections, keypoints: sv.KeyPoints, show=False):
    
    transformer = PerspectiveTransformer(
        source=keypoints,
        target=BOARD_POINTS
    )

    frame_pieces_xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    board_pieces_xy = transformer.transform_points(points=frame_pieces_xy)
                    
    board = draw_points_on_board(
        board=board,
        xy=board_pieces_xy,
        py=-10
    )
    
    if show:
        cv2.imshow("board", board)
    
    return board_pieces_xy