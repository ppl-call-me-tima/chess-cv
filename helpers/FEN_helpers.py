import numpy as np
import supervision as sv

class FEN:
    def __init__(
        self, 
        pitch_pieces_xy: np.ndarray[np.float32], 
        detections: sv.Detections
    ):
        
        # coordinate-piece dict. initialization
        self.class_at_point = {}
        for i in range(len(detections.xyxy)):
            self.class_at_point[tuple(pitch_pieces_xy[i])] = detections.data["class_name"][i]

        # matrix initiailization
        self.matrix = []

        for x in range(0, 800, 100):

            row = []

            for y in range(0, 800, 100):

                lx = x
                ly = y
                rx = x + 100
                ry = y + 100

                piece_found = False

                for xy in pitch_pieces_xy:
                    piece_x = xy[1]
                    piece_y = xy[0]

                    if piece_x >= lx and piece_x <= rx and piece_y >= ly and piece_y <= ry:
                        # print(lx, ly, " : ", rx, ry, " : ", class_at_point[tuple(xy)])
                        row.append(self.class_at_point[tuple(xy)])
                        piece_found = True
                        break

                if not piece_found:
                    row.append("")

            self.matrix.append(row)

    def rotate_anticlockwise(self):
        rotated = []

        for j in range(7, -1, -1):
            col = []

            for i in range(8):
                col.append(self.matrix[i][j])

            rotated.append(col)

        for i in range(8):
            for j in range(8):
                self.matrix[i][j] = rotated[i][j]
    
    def symbol_for(self, class_name):
        if class_name[5:] == "Knight":
            symbol = "N"
        else:
            symbol = class_name[5]

        if class_name[:5] == "Black":
            symbol = symbol.lower()

        return symbol

    def fen(self):
        fen = ""

        for row in self.matrix:
            blank_count = 0

            for class_name in row:
                if class_name == "":
                    blank_count += 1
                else:
                    if blank_count > 0:
                        fen += str(blank_count)
                        blank_count = 0
                    fen += self.symbol_for(class_name)
            else:
                if blank_count > 0:
                    fen += str(blank_count)
                fen += "/"

        fen = fen[:-1]
        return fen