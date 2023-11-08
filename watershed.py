import cv2
import numpy as np
from skimage import measure
from skimage.segmentation import clear_border

def watershedAlgo(image_path):
    
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2)

    opening = clear_border(opening)

    sure_bg = cv2.dilate(opening, kernel, iterations=10)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image,markers)
    image[markers == -1] = [0,255,255]
    
    regions = measure.regionprops(markers, intensity_image=gray)
    result = []
    
    for i in range(len(regions)):
        result.append(regions[i]['bbox'])
        
    return result, image

def dummyAlgo(image_path):
    image = cv2.imread(image_path)
    return [], image

if __name__ == '__main__':
    
    image_path = 'images/1.jpg'
    boundaries, image = watershedAlgo(image_path)
    
    for i in range(len(boundaries)):
        print(f"{i + 1}: {boundaries[i]} \n")
        
    cv2.imshow("", image)
    cv2.waitKey(0)