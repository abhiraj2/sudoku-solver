import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.models import load_model


def read_img():
    img_url = input("Enter the image location here: ")
    img = cv2.imread(img_url, cv2.IMREAD_GRAYSCALE) #Read it directly in grayscale
    return img

def corners(img):
                                    #image, Mode, Approximation method   
    ext_contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finds the contours based on difference in color
    
    ext_contours = ext_contours[0] if len(ext_contours) == 2 else ext_contours[1] 
    ext_contours = sorted(ext_contours, key = cv2.contourArea, reverse = True)
    
    for c in ext_contours:
        peri = cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c,0.015*peri, True)
        if len(corners) == 4:
            return corners
    
    '''
    polygon = ext_contours[0]
    
    z
    # Ramer Doughlas Peucker algorithm:
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in
                                     ext_contours[0]]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in
                                ext_contours[0]]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in
                                    ext_contours[0]]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in
                                  ext_contours[0]]), key=operator.itemgetter(1))
    
    corners = [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]
    return corners
    '''

def ordered_corner_points(corners):
    corners = [(corner[0][0], corner[0][1]) for corner in corners]
    
    top = sorted(corners, key=lambda x: x[1], reverse=False)[:2]
    bottom = sorted(corners, key=lambda x: x[1], reverse=False)[2:]
    
    if top[0][0] > top[1][0]:
        top_r = top[0]
        top_l = top[1]
    else: 
        top_r = top[1]
        top_l = top[0]
    
    if bottom[0][0] > bottom[1][0]:
        bottom_r = bottom[0]
        bottom_l = bottom[1]
    else: 
        bottom_r = bottom[1]
        bottom_l = bottom[0]
    
    
    return top_l, top_r, bottom_r, bottom_l

def crop_and_warp(image, corners):
    
    ordered_corners = ordered_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners
    
    
    width_A = np.sqrt(((bottom_l[0] - bottom_r[0])**2) + ((bottom_l[1] - bottom_r[1])**2))
    width_B = np.sqrt(((top_l[0] - top_r[0])**2) + ((top_l[1] - top_r[1])**2))
    width = max(int(width_A), int(width_B))
    
    height_A = np.sqrt(((bottom_r[0] - top_r[0])**2) + ((bottom_r[1] - top_r[1])**2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0])**2) + ((top_l[1] - bottom_l[1])**2))
    height = max(int(height_A), int(height_B))
    
    dimensions = np.array([[0,0], [width - 1, 0], [width -1, height -1], [0, height -1]], dtype = "float32")
    ordered_corners = np.array(ordered_corners, dtype = "float32")
    grid = cv2.getPerspectiveTransform(ordered_corners, dimensions)
    
    return cv2.warpPerspective(image, grid, (width, height))



def extract_cells(img):
    grid = np.copy(img)
    
    celledge_h = np.shape(grid)[0] // 9
    celledge_w = np.shape(grid)[1] // 9
    
    #grid = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
    #grid = cv2.bitwise_not(grid, grid)
    
    tempgrid = []
    
    for i in range(celledge_h, np.shape(grid)[0], celledge_h):
        for j in range(celledge_w, np.shape(grid)[1], celledge_w):
            rows = grid[i - celledge_h:i]
            tempgrid.append([rows[k][j-celledge_w:j] for k in range(len(rows))])

    finalgrid = []
    for i in range(0, len(tempgrid) -8, 9):
        finalgrid.append(tempgrid[i:i+9])
    
    for i in range(9):
        for j in range(9):
            finalgrid[i][j] = np.array(finalgrid[i][j])
    
    try:
        for i in range(9):
            for j in range(9):
                np.os.remove("Cells/" + str(i) + str(j) + ".jpg", finalgrid[i][j])
    except:
        pass
    
    for i in range(9):
        for j in range(9):
            cv2.imwrite(str("Cells/" + str(i) + str(j) + ".jpg"), finalgrid[i][j])
    
    return finalgrid


def predict(img_grid):
    image = img_grid.copy()
    image = cv2.resize(image, (28, 28))
    image = image.astype('float32')
    image = image.reshape(1, 28, 28, 1)
    image /= 255
    model = load_model('cnn.hdf5')
    pred = model.predict(image.reshape(1, 28, 28, 1), batch_size=1)
    
    return pred.argmax()
    
    
def extract_number_image(img_grid):
    tmp_sudoku = [[0 for i in range(9)] for j in range(9)]
    for i in range(9):
        for j in range(9):

            image = img_grid[i][j]
            image = cv2.resize(image, (28, 28))
            original = image.copy()

            thresh = 128  # define a threshold, 128 is the middle of black and white in grey scale
            # threshold the image
            gray = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]

            # Find contours
            cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)

                if (x < 3 or y < 3 or h < 3 or w < 3):
                    # Note the number is always placed in the center
                    # Since image is 28x28
                    # the number will be in the center thus x >3 and y>3
                    # Additionally any of the external lines of the sudoku will not be thicker than 3
                    continue
                ROI = gray[y:y + h, x:x + w]
                ROI = scale_and_centre(ROI, 120)
                # display_image(ROI)

                # Writing the cleaned cells
                cv2.imwrite("Cells/cell{}{}.png".format(i, j), ROI)
                tmp_sudoku[i][j] = predict(ROI)
    print("\n")
    for i in tmp_sudoku:
        print(i)
        print('\n')
    return tmp_sudoku


def scale_and_centre(img, size, margin=20, background=0):
    """Scales and centres an image onto a new background square."""
    h, w = img.shape[:2]

    def centre_pad(length):
        """Handles centering for a given length that may be odd or even."""
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))








    
sudoku_img = read_img()
proc = cv2.GaussianBlur(sudoku_img.copy(), (9, 9), 0)
process = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#process = cv2.Canny(proc, 10, 100)
process = cv2.bitwise_not(process, process)

kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)

img_test = cv2.erode(process, kernel, iterations=1)

process = cv2.dilate(img_test, kernel, iterations=1)



new_img = crop_and_warp(process, corners(process))
img_test = crop_and_warp(img_test, corners(img_test))

extract_cells(new_img)

extract_number_image(extract_cells(new_img))


plt.imshow(new_img, cmap="gray")
plt.show()
