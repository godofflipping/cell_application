import cv2

def pHash(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
    image_hash = cv2.img_hash.pHash(gray)
    return int.from_bytes(image_hash.tobytes(), byteorder='big', signed=False)