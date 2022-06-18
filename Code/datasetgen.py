import os
import numpy as np
from imageRead import Image as im

class DataGeneration:

        def createDataset(self,data_dir):
            """A method to create features and labels for our model

            Args:
                data_dir (String): Path to our root directory

            Returns:
                tuple: A tuple which returns features and labels
            """
            features = [] # An empty list to append all the features
            labels = [] # An empty list to append all the labels
            return features,labels