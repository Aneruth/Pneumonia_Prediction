import os
from datasetgen import DataGeneration as dg
from keras.layers import MaxPooling2D,Conv2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.models import Sequential
import tensorflow as tf

tf.compat.v1.set_random_seed(202)

class Model:
    def __init__(self) -> None:
        """A initialiser where we define the variable model as None
        """
        self.model = None
        # Check for a model in folder if weights present then the we skip the 
        # training and load that model else we train and save the model.
        self.status = None
    
    def checkModel(self):
        """A function which checks if the model is trained or not

        Returns:
            boolean: A boolean value where we assign
        """
        model_fldr_path = './savedModels/'
        
        if os.path.isdir(model_fldr_path) == False:
            os.mkdir(model_fldr_path)
        
        self.status = True if len(os.listdir(model_fldr_path)) == 1 else False

        return self.status
    
    def modelInit(self):
        self.model = Sequential([
            Conv2D(16,(3,3),activation = "relu" , input_shape = (255,255,3)) ,
            
            MaxPooling2D(2,2),
            Conv2D(32,(3,3),activation = "relu") ,  
            
            MaxPooling2D(2,2),
            Conv2D(64,(3,3),activation = "relu") ,  
            
            MaxPooling2D(2,2),
            Conv2D(128,(3,3),activation = "relu"),  
            
            MaxPooling2D(2,2),
            Flatten(), 
            
            Dense(550,activation="relu"),      #Adding the Hidden layer
            Dropout(0.1,seed = 2019),
            
            Dense(400,activation ="relu"),
            Dropout(0.3,seed = 2019),
            
            Dense(300,activation="relu"),
            Dropout(0.4,seed = 2019),
            
            Dense(200,activation ="relu"),
            
            Dropout(0.2,seed = 2019),
            Dense(3,activation = "softmax")   #Adding the Output Layer
        ])
        return self.model.summary()

    def createModel(self,status=None):
        status = self.checkModel()

        if status == False:
            adam = Adam(lr=0.001)
            self.model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics = ['acc'])
            # hist = self.model.fit(X_train,y_train , epochs=10, batch_size=64,validation_data=(X_test,y_test))

        return status

if __name__ == '__main__':
    mod = Model()
    print(mod.modelInit())