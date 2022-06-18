import cv2
from skimage.transform import resize

class Image:

    def __init__(self,image_path):
        self.img = image_path
    
    def readImage(self):
        """A method to read the image and preprocess it to 255 pixels

        Args:
            file_path (String): String path to fetch file

        Returns:
            ndarray: A numpy array of image(s)
        """
        img = cv2.imread(self.img)
        return img
    
    def downScale(self):
        """A method to downscale the image to 255 pixels

        Returns:
            ndarry: A numpy array with 3 channels
        """
        image = self.readImage()
        
        img = resize(image, (255, 255), anti_aliasing=True)

        return img