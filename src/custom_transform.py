import cv2 as cv

def crop_height(img, y1, y2):
    return img[:, y1 : y2, :]

def bgr_to_rgb(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def crop(img, x1, x2, y1, y2):
    return img[:, y1 : y2, x1 : x2]


class BGRToYUV:
    def __call__(self, img):
        return cv.cvtColor(img, cv.COLOR_BGR2YUV)

class CropHeight:
    def __init__(self, y1, y2):
        self.y1 = y1
        self.y2 = y2
        
    def __call__(self, img):
        return crop_height(img, self.y1, self.y2)


class Crop:
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
    
    def __call__(self, img):
        return crop(img, self.x1, self.x2, self.y1, self.y2)

class CropVideo:
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
    
    def __call__(self, img):
        return img[:, :, self.y1 : self.y2, self.x1 : self.x2]