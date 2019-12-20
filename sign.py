#Importing all the libraries
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils


#initializing datas and classes
datas = []
classes = []
input_img_size = 32

#pass the image path
primary_directory = 'Your image path'


#returning the corresponding label
def assign_label(image,label):
    return label


#appending all the images inside the variables
get_image_folder = [ primary_directory + '/' + i for i in os.listdir(primary_directory)] 
for image_folder in get_image_folder:        #getting all individual folder
    for image in os.listdir(image_folder):   #getting all images using list directory from image folder
        image_label = assign_label(image,image_folder)     #passing label to assign_label function
        path = os.path.join(image_folder,image)
        img = cv2.imread(path,cv2.IMREAD_COLOR)             #reading all the image file
        img = cv2.resize(img,(input_img_size,input_img_size))   #resizing the image to 32*32 size
        
        datas.append(img)
        classes.append(image_label)



#encoding the classes
le = LabelEncoder()
Y = le.fit_transform(classes)
Y = to_categorical(Y,29)
X = np.array(datas)
X = X/255



#splitting the images into train and validation
X_train, X_test , Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)




#building the cnn network
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (32,32,3)))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 

model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(128,activation='relu'))

model.add(Dense(29, activation = "softmax"))


#introducing the required batch size and epochs and assign imagedatagenerator function of keras
batch_size=64
epochs=30
datagen = ImageDataGenerator()

#compile and train the network
model.compile(optimizer=optimizers.Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
History = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),epochs = epochs, validation_data = (X_test,Y_test),verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size)


#plotting accuracy and loss graph
plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.grid(True)
plt.title("Accuracy Graph")
plt.xlabel("Epochs",fontsize=16)
plt.ylabel("Accuracy",fontsize=16)
plt.legend(['train','validation'],loc='lower right')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()



plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.grid(True)
plt.title("Loss Graph")
plt.xlabel("Epochs",fontsize=16)
plt.ylabel("Loss",fontsize=16)
plt.legend(['train','validation'],loc='upper right')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


#save the model
model.save('Your path')



#testing the image
test_image_dir ='Your test image path'
test = cv2.imread(test_image_dir,cv2.IMREAD_COLOR)

test = cv2.resize(test,(32,32))


test = test.reshape(1,32,32,3)
print(test.shape)
result = model.predict_classes(test)
print(result)


