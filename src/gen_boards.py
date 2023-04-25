import cv2 as cv
import numpy as np


# Load the dictionary that was used to generate the markers.
dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_1000)

pix_sz = int(600/25.4 + 0.5)  # dpi/25.4
size = (210-5*2)/(0.04*5+0.01*4)*0.04  # 33.33333 mm

if __name__ == "__main__":
    for idx_board in range(3):
        idx_start = idx_board*35
        board = cv.aruco.GridBoard((5, 7), 0.04, 0.01, dictionary, ids=np.arange(35)+idx_start)

        boardImage = board.generateImage(tuple(np.round(np.array([210, 297])*pix_sz)), 0, 5*pix_sz)
        cv.imwrite(f"out/boardImage_{idx_start}-{idx_start+35}.png", boardImage)
