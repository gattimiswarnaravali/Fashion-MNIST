import tensorflow
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

# loading the data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

load_model = tensorflow.keras.models.load_model('fmnist_95.h5')

# test_image = image.load_img('C:\Learnbay\DeploymentLearnbay\FMNIST\Model\image1.png', target_size=(28,28))
# print(type(test_image))
# test_image = image.img_to_array(test_image)
# print(test_image.shape)
# #test_image = test_image[:,:,0]

classes = ['T-shirt/top', 'Trouser', 'Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
# test_image = test_image.reshape(1,28,28,3)
#test_image = test_image/255
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1], X_test.shape[2],1)
y_pred = load_model.predict(X_test)
print(classes[np.argmax(y_pred[20])])

