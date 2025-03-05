import cv2
import numpy as np

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