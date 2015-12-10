import cv2

def resize_image(params,im):

    # Get image dimensions
    height, width = im.shape[:2]

    # If the image width is smaller than the proposed small dimension, keep the original size !
    resize_dim = min(params['max_size'],width)

    # We don't want to lose aspect ratio:
    dim = (resize_dim, height * resize_dim/width)

    # Resize and return new image
    return cv2.resize(im,dim)