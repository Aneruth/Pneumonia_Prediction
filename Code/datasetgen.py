import os
import numpy as np
from imageRead import Image as im

class DataGeneration:
    
    def __init__(self,data_dir) -> None:      
        self.data_dir = data_dir
        self.data_dir_list = os.listdir(self.data_dir)[1:]
        self.features = [] # An empty list to append all the features
        self.labels = [] # An empty list to append all the labels

    def createDataset(self):
        """A method to create features and labels for our model

        Args:
            data_dir (String): Path to our root directory
        """
        print(f'************* Data creation started *************')
        for folder in self.data_dir_list:
            img_dir_path = os.path.join(self.data_dir,folder)
            img_dir_list = os.listdir(img_dir_path)
            for img_fldr in img_dir_list: # fetch the labels
                if img_fldr != '.DS_Store':
                    path = os.path.join(img_dir_path,img_fldr)
                    for file in os.listdir(path):
                        # A loop to fetch file path
                        file_path = os.path.join(path,file)
                        image = im(file_path)
                        resize_image = im.downScale(image)
                        self.features.append(resize_image)
                        self.labels.append(folder)
            print(f'************* Data extraction for {folder} completed *************')
        print(f'Data extraction completed')
    
    def convertToNumpyArray(self):
        """A method to convert the features and labels to numpy array
        Returns:
            Tuple: a tuple with features and labels as numpy array
        """
        self.createDataset()
        return np.asarray(self.features),np.asarray(self.labels)

if __name__ == '__main__':
    dc = DataGeneration('../data/')
    a,b = dc.convertToNumpyArray()
    print(type(a),type(b))