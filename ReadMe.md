
<!-- #region id="xCIFOwUPxoF5" colab_type="text" -->
# Butterfly Classification using Transfer Learning
A multi-class classification approach to categorize the type of Butterfly using Transfer Learning from CNN 'inception_V3' model  in Tensorflow.
<!-- #endregion -->

<!-- #region id="jhMR2vMg_V9i" colab_type="text" -->
## 1. Aim
To create a classifiacation model predicting the breed of the butterfly belonging to the following classes: 

1.   Black_Swallotail (B)
2.   California_Sister (C)
3.   Milberts_Tortoisedhell(M)
4.   Red_Admiral (R)
5.   Red_Spotted_Purple (S)
6.   The_Blues (T)

Since, I am going to use a   pretrained model using transfer learning, hence would like to take less distinguishable images for some classes to see how good the model performs .

 
<!-- #endregion -->

<!-- #region id="WIN-af8yI7Kv" colab_type="text" -->
### First i'll implement the classification using a simple model from scratch and then transfer learning to see the difference.
<!-- #endregion -->

<!-- #region id="AyTiwOBkI-EU" colab_type="text" -->
## 2. Setup
<!-- #endregion -->

```python colab_type="code" id="lbFmQdsZs5eW" colab={}
import os
import zipfile
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import Model
from os import getcwd
```

```python id="J0tNQAuHM2Lu" colab_type="code" outputId="27756137-208c-43a5-ddb3-7a35bfcc2337" colab={"base_uri": "https://localhost:8080/", "height": 34}
from google.colab import drive
drive.mount('/content/drive')
```

<!-- #region id="8yJdDw0iNeyq" colab_type="text" -->
## 3.1 Loading Dataset
<!-- #endregion -->

```python id="OwfFTs4BNaTk" colab_type="code" outputId="de5050e3-8d5a-42b6-d33e-28fd60d95997" colab={"base_uri": "https://localhost:8080/", "height": 230}
#collecting the path for base directory
base_dir='/content/drive/My Drive/Butterfly-Classifier/datset'
training_dir=os.path.join(base_dir, 'training')
validation_dir=os.path.join(base_dir, 'validation')

train_b_dir=os.path.join(training_dir,'Black_Swallotail')
train_c_dir=os.path.join(training_dir,'California_Sister')
train_m_dir=os.path.join(training_dir,'Milberts_Tortoiseshell')
train_r_dir=os.path.join(training_dir,'Red_Admiral')
train_s_dir=os.path.join(training_dir,'Red_Spotted_Purple')
train_t_dir=os.path.join(training_dir,'The_Blues')



valid_b_dir=os.path.join(validation_dir,'Black_Swallotail')
valid_c_dir=os.path.join(validation_dir,'California_Sister')
valid_m_dir=os.path.join(validation_dir,'Milberts_Tortoiseshell')
valid_r_dir=os.path.join(validation_dir,'Red_Admiral')
valid_s_dir=os.path.join(validation_dir,'Red_Spotted_Purple')
valid_t_dir=os.path.join(validation_dir,'The_Blues')

#Let's find out the total number of horse and human images in the directories:
print('total Black_Swallotail in training: ', len(os.listdir(train_b_dir)))
print('total California_Sister in training: ', len(os.listdir(train_c_dir)))
print('total Milberts_Tortoiseshell in training: ', len(os.listdir(train_m_dir)))
print('total Red_Admiral in training: ', len(os.listdir(train_r_dir)))
print('total Red_Spotted_Purple in training: ', len(os.listdir(train_s_dir)))
print('total The_Blues in training: ', len(os.listdir(train_t_dir)))

print('total Black_Swallotail in validation: ', len(os.listdir(valid_b_dir)))
print('total California_Sister in validation: ', len(os.listdir(valid_c_dir)))
print('total Milberts_Tortoiseshell in validation: ', len(os.listdir(valid_m_dir)))
print('total Red_Admiral in validation: ', len(os.listdir(valid_r_dir)))
print('total Red_Spotted_Purple in validation: ', len(os.listdir(valid_s_dir)))
print('total The_Blues in validation: ', len(os.listdir(valid_t_dir)))


```

<!-- #region id="76tRE431Pn4c" colab_type="text" -->
## 3.2 Looking at the Dataset
<!-- #endregion -->

```python id="vPyGaVcqPylL" colab_type="code" outputId="8bb01273-bcc5-4991-9265-805d2dda9bbe" colab={"base_uri": "https://localhost:8080/", "height": 250}
#let's see the file-names in the directories
train_b_names = os.listdir(train_b_dir)
print(train_b_names[:10])
train_c_names = os.listdir(train_c_dir)
print(train_c_names[:10])
train_m_names = os.listdir(train_m_dir)
print(train_m_names[:10])
train_r_names = os.listdir(train_r_dir)
print(train_r_names[:10])
train_s_names = os.listdir(train_s_dir)
print(train_s_names[:10])
train_t_names = os.listdir(train_t_dir)
print(train_t_names[:10])


validation_b_names = os.listdir(valid_b_dir)
print(validation_b_names[:10])
validation_m_names = os.listdir(valid_m_dir)
print(validation_m_names[:10])
validation_c_names = os.listdir(valid_c_dir)
print(validation_c_names[:10])
validation_r_names = os.listdir(valid_r_dir)
print(validation_r_names[:10])
validation_s_names = os.listdir(valid_s_dir)
print(validation_s_names[:10])
validation_t_names = os.listdir(valid_t_dir)
print(validation_t_names[:10])

```

```python id="UMTzQrsXQdAF" colab_type="code" outputId="57229523-8a0b-4840-d348-e757923c1b36" colab={"base_uri": "https://localhost:8080/", "height": 1000}
#Let's see the images present in the dataset.

import matplotlib.image as mpimg
# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 6
ncols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 7
next_b_pix = [os.path.join(train_b_dir, fname) 
                for fname in train_b_names[pic_index-4:pic_index]]
next_c_pix = [os.path.join(train_c_dir, fname) 
                for fname in train_c_names[pic_index-4:pic_index]]
next_m_pix = [os.path.join(train_m_dir, fname) 
                for fname in train_m_names[pic_index-4:pic_index]]
next_r_pix = [os.path.join(train_r_dir, fname) 
                for fname in train_r_names[pic_index-4:pic_index]]
next_s_pix = [os.path.join(train_s_dir, fname) 
                for fname in train_s_names[pic_index-4:pic_index]]
next_t_pix = [os.path.join(train_t_dir, fname) 
                for fname in train_t_names[pic_index-4:pic_index]]

for i, img_path in enumerate(next_b_pix+next_c_pix+next_m_pix+next_r_pix+next_s_pix+next_t_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') 

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

```

<!-- #region id="GpgfXhGEimUk" colab_type="text" -->
As we can see the few classes are very similar and hence forming a tough job for our classifier. Let's see.

<!-- #endregion -->

<!-- #region id="3avd8c2-Sovu" colab_type="text" -->
## 4. Building Model from Scratch
<!-- #endregion -->

```python id="dDrwt4R-QdbH" colab_type="code" outputId="5e749825-ff09-465e-f5cd-3b18de267a11" colab={"base_uri": "https://localhost:8080/", "height": 727}
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    #after 6 layers we use flatten to create single vector along with activation function

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),

  #since it's a multi-class hence we'll use softmax activation function.

    tf.keras.layers.Dense(6, activation='softmax')
])

model.summary()
```

```python id="7fXVY6FNS1Q5" colab_type="code" colab={}
#compiling the model by setting the type of classifier, optimizer, acc we want in output

#using the RMSprop optimization algorithm is preferable to stochastic 
#gradient descent (SGD), because RMSprop automates learning-rate tuning for us. 
model.compile(optimizer = RMSprop(lr=1e-4),
              loss = 'categorical_crossentropy',metrics=['accuracy'])
```

<!-- #region id="bihsezfITE_0" colab_type="text" -->
## 5.1 Data Preprocessing
<!-- #endregion -->

```python id="NcDI-7vVTH9s" colab_type="code" outputId="7b5d0d95-f5b0-476e-ad43-b1453597a70b" colab={"base_uri": "https://localhost:8080/", "height": 52}
train_datagen = ImageDataGenerator(
      rescale=1./255.,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = train_datagen.flow_from_directory(
        training_dir,  # This is the source directory for training images
        target_size=(200, 200),  # All images will be resized to 200x200
        batch_size=32,
        # Since we use sparse_categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical')
```

<!-- #region id="x4DLGB7RTbJK" colab_type="text" -->
## 6.1 Training on Scratch model
<!-- #endregion -->

```python id="1ScO8SYlTRZJ" colab_type="code" outputId="e5c8fcfb-a1a6-4d7f-ba87-dee24ab4c4cd" colab={"base_uri": "https://localhost:8080/", "height": 1000}
history = model.fit(
      train_generator,
      steps_per_epoch=100,  # 1800 images = batch_size * steps
      epochs=40,
      validation_data=validation_generator,
      validation_steps=25,
     verbose=1
      
)
```

<!-- #region id="JcZ7G98YTiJr" colab_type="text" -->
## 6.2 Visualization of results


<!-- #endregion -->

```python id="I5BRaOY2TiyM" colab_type="code" outputId="33b91475-c242-4099-9c2e-fe4255041e6f" colab={"base_uri": "https://localhost:8080/", "height": 298}
# Plot the chart for accuracy and loss on both training and validation
%matplotlib inline
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()



plt.show()
```

<!-- #region id="gYrDgufyFONI" colab_type="text" -->
We can infer from the above result that the final accuracy attained by the model by the end of 40 epoch is approx: 78% which is quite beautiful. Now let's see the performance of the transfered model
<!-- #endregion -->

<!-- #region id="uSq9vlLlT6Fm" colab_type="text" -->
____________________________________________________________________________
<!-- #endregion -->

<!-- #region id="Q9hi-wXXGTNq" colab_type="text" -->
##  Getting the Transfered Weights
<!-- #endregion -->

```python colab_type="code" id="1xJZ5glPPCRz" outputId="0a8a02b4-7d0e-4d5c-fd3d-1cc5a2d66eb1" colab={"base_uri": "https://localhost:8080/", "height": 232}

#here's a direct api to call the inception model without downloading it 

!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5


# Import the inception model  
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Create an instance of the inception model from the local pre-trained weights
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (200, 200, 3), 
                                include_top = False, 
                                weights = None) 

pre_trained_model.load_weights(local_weights_file)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
    layer.trainable = False 


#pre_trained_model.summary()
#It's pretty large hence not displaying.
```

```python colab_type="code" id="CFsUlwdfs_wg" outputId="52b90b7f-9379-4880-bc0f-bb8af1941d21" colab={"base_uri": "https://localhost:8080/", "height": 34}
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output 
```

```python colab_type="code" id="-bsWZWp5oMq9" colab={}
# Define a Callback class that stops training once accuracy reaches 90.0%
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.9):
            print("\nReached 90.0% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()      
```

```python colab_type="code" id="BMXb913pbvFg" outputId="ee0ebd3d-eb96-47c8-f70a-cd3b54a9f20a" colab={"base_uri": "https://localhost:8080/", "height": 1000}


# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (6, activation='softmax')(x)           

model = Model( pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'categorical_crossentropy', 
              metrics = ['acc'])

model.summary()


```

```python colab_type="code" outputId="75034528-6d19-4161-9a29-0b6c992ebba5" id="vZe4LJ5YgG3S" colab={"base_uri": "https://localhost:8080/", "height": 52}
train_datagen = ImageDataGenerator(
      rescale=1./255.,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = train_datagen.flow_from_directory(
        training_dir,  # This is the source directory for training images
        target_size=(200, 200),  # All images will be resized to 200x200
        batch_size=32,
        # Since we use sparse_categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical')
```

<!-- #region colab_type="text" id="E9SkruA1gG3o" -->
## 6.1 Training on Transfered model
<!-- #endregion -->

```python colab_type="code" id="TblMbgi5gG3s" outputId="3c1a8bd4-32c4-49d8-b2f5-20a7392411d8" colab={"base_uri": "https://localhost:8080/", "height": 534}
history = model.fit(
      train_generator,
      steps_per_epoch=100,  # 1800 images = batch_size * steps
      epochs=40,
      validation_data=validation_generator,
      validation_steps=25,
     verbose=1,
     callbacks=[callbacks]
     
      
)
```

<!-- #region colab_type="text" id="d4oQjIWugG39" -->
## 6.2 Visualization of results


<!-- #endregion -->

```python colab_type="code" id="Xnhb7-0xgG3_" outputId="167dd1c7-a260-4549-db7e-4164cce4685e" colab={"base_uri": "https://localhost:8080/", "height": 298}
# Plot the chart for accuracy and loss on both training and validation
%matplotlib inline
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()



plt.show()
```

<!-- #region id="iY5707JU_g6q" colab_type="text" -->
We can conclude from the above graph that the model has acquired the accuracy of 90% only by the end of 13th ecpoh with really low variance and has definately performed better than the _MODEL FROM SCRATCH_ 
<!-- #endregion -->

<!-- #region id="m7s4dPJ1gnw7" colab_type="text" -->
## Testing the model
<!-- #endregion -->

<!-- #region id="rnIKnI1J9wpw" colab_type="text" -->
Taking an image from the internet and predicting its type. 
<!-- #endregion -->

```python id="9ntXEVYcgpRd" colab_type="code" outputId="c6e26e8e-b676-48ef-f038-87884b3043da" colab={"resources": {"http://localhost:8080/nbextensions/google.colab/files.js": {"data": "Ly8gQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQwovLwovLyBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgIkxpY2Vuc2UiKTsKLy8geW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLgovLyBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXQKLy8KLy8gICAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjAKLy8KLy8gVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZQovLyBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiAiQVMgSVMiIEJBU0lTLAovLyBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC4KLy8gU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZAovLyBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS4KCi8qKgogKiBAZmlsZW92ZXJ2aWV3IEhlbHBlcnMgZm9yIGdvb2dsZS5jb2xhYiBQeXRob24gbW9kdWxlLgogKi8KKGZ1bmN0aW9uKHNjb3BlKSB7CmZ1bmN0aW9uIHNwYW4odGV4dCwgc3R5bGVBdHRyaWJ1dGVzID0ge30pIHsKICBjb25zdCBlbGVtZW50ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3BhbicpOwogIGVsZW1lbnQudGV4dENvbnRlbnQgPSB0ZXh0OwogIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKHN0eWxlQXR0cmlidXRlcykpIHsKICAgIGVsZW1lbnQuc3R5bGVba2V5XSA9IHN0eWxlQXR0cmlidXRlc1trZXldOwogIH0KICByZXR1cm4gZWxlbWVudDsKfQoKLy8gTWF4IG51bWJlciBvZiBieXRlcyB3aGljaCB3aWxsIGJlIHVwbG9hZGVkIGF0IGEgdGltZS4KY29uc3QgTUFYX1BBWUxPQURfU0laRSA9IDEwMCAqIDEwMjQ7Ci8vIE1heCBhbW91bnQgb2YgdGltZSB0byBibG9jayB3YWl0aW5nIGZvciB0aGUgdXNlci4KY29uc3QgRklMRV9DSEFOR0VfVElNRU9VVF9NUyA9IDMwICogMTAwMDsKCmZ1bmN0aW9uIF91cGxvYWRGaWxlcyhpbnB1dElkLCBvdXRwdXRJZCkgewogIGNvbnN0IHN0ZXBzID0gdXBsb2FkRmlsZXNTdGVwKGlucHV0SWQsIG91dHB1dElkKTsKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIC8vIENhY2hlIHN0ZXBzIG9uIHRoZSBvdXRwdXRFbGVtZW50IHRvIG1ha2UgaXQgYXZhaWxhYmxlIGZvciB0aGUgbmV4dCBjYWxsCiAgLy8gdG8gdXBsb2FkRmlsZXNDb250aW51ZSBmcm9tIFB5dGhvbi4KICBvdXRwdXRFbGVtZW50LnN0ZXBzID0gc3RlcHM7CgogIHJldHVybiBfdXBsb2FkRmlsZXNDb250aW51ZShvdXRwdXRJZCk7Cn0KCi8vIFRoaXMgaXMgcm91Z2hseSBhbiBhc3luYyBnZW5lcmF0b3IgKG5vdCBzdXBwb3J0ZWQgaW4gdGhlIGJyb3dzZXIgeWV0KSwKLy8gd2hlcmUgdGhlcmUgYXJlIG11bHRpcGxlIGFzeW5jaHJvbm91cyBzdGVwcyBhbmQgdGhlIFB5dGhvbiBzaWRlIGlzIGdvaW5nCi8vIHRvIHBvbGwgZm9yIGNvbXBsZXRpb24gb2YgZWFjaCBzdGVwLgovLyBUaGlzIHVzZXMgYSBQcm9taXNlIHRvIGJsb2NrIHRoZSBweXRob24gc2lkZSBvbiBjb21wbGV0aW9uIG9mIGVhY2ggc3RlcCwKLy8gdGhlbiBwYXNzZXMgdGhlIHJlc3VsdCBvZiB0aGUgcHJldmlvdXMgc3RlcCBhcyB0aGUgaW5wdXQgdG8gdGhlIG5leHQgc3RlcC4KZnVuY3Rpb24gX3VwbG9hZEZpbGVzQ29udGludWUob3V0cHV0SWQpIHsKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIGNvbnN0IHN0ZXBzID0gb3V0cHV0RWxlbWVudC5zdGVwczsKCiAgY29uc3QgbmV4dCA9IHN0ZXBzLm5leHQob3V0cHV0RWxlbWVudC5sYXN0UHJvbWlzZVZhbHVlKTsKICByZXR1cm4gUHJvbWlzZS5yZXNvbHZlKG5leHQudmFsdWUucHJvbWlzZSkudGhlbigodmFsdWUpID0+IHsKICAgIC8vIENhY2hlIHRoZSBsYXN0IHByb21pc2UgdmFsdWUgdG8gbWFrZSBpdCBhdmFpbGFibGUgdG8gdGhlIG5leHQKICAgIC8vIHN0ZXAgb2YgdGhlIGdlbmVyYXRvci4KICAgIG91dHB1dEVsZW1lbnQubGFzdFByb21pc2VWYWx1ZSA9IHZhbHVlOwogICAgcmV0dXJuIG5leHQudmFsdWUucmVzcG9uc2U7CiAgfSk7Cn0KCi8qKgogKiBHZW5lcmF0b3IgZnVuY3Rpb24gd2hpY2ggaXMgY2FsbGVkIGJldHdlZW4gZWFjaCBhc3luYyBzdGVwIG9mIHRoZSB1cGxvYWQKICogcHJvY2Vzcy4KICogQHBhcmFtIHtzdHJpbmd9IGlucHV0SWQgRWxlbWVudCBJRCBvZiB0aGUgaW5wdXQgZmlsZSBwaWNrZXIgZWxlbWVudC4KICogQHBhcmFtIHtzdHJpbmd9IG91dHB1dElkIEVsZW1lbnQgSUQgb2YgdGhlIG91dHB1dCBkaXNwbGF5LgogKiBAcmV0dXJuIHshSXRlcmFibGU8IU9iamVjdD59IEl0ZXJhYmxlIG9mIG5leHQgc3RlcHMuCiAqLwpmdW5jdGlvbiogdXBsb2FkRmlsZXNTdGVwKGlucHV0SWQsIG91dHB1dElkKSB7CiAgY29uc3QgaW5wdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaW5wdXRJZCk7CiAgaW5wdXRFbGVtZW50LmRpc2FibGVkID0gZmFsc2U7CgogIGNvbnN0IG91dHB1dEVsZW1lbnQgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChvdXRwdXRJZCk7CiAgb3V0cHV0RWxlbWVudC5pbm5lckhUTUwgPSAnJzsKCiAgY29uc3QgcGlja2VkUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICBpbnB1dEVsZW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignY2hhbmdlJywgKGUpID0+IHsKICAgICAgcmVzb2x2ZShlLnRhcmdldC5maWxlcyk7CiAgICB9KTsKICB9KTsKCiAgY29uc3QgY2FuY2VsID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnYnV0dG9uJyk7CiAgaW5wdXRFbGVtZW50LnBhcmVudEVsZW1lbnQuYXBwZW5kQ2hpbGQoY2FuY2VsKTsKICBjYW5jZWwudGV4dENvbnRlbnQgPSAnQ2FuY2VsIHVwbG9hZCc7CiAgY29uc3QgY2FuY2VsUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICBjYW5jZWwub25jbGljayA9ICgpID0+IHsKICAgICAgcmVzb2x2ZShudWxsKTsKICAgIH07CiAgfSk7CgogIC8vIENhbmNlbCB1cGxvYWQgaWYgdXNlciBoYXNuJ3QgcGlja2VkIGFueXRoaW5nIGluIHRpbWVvdXQuCiAgY29uc3QgdGltZW91dFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgc2V0VGltZW91dCgoKSA9PiB7CiAgICAgIHJlc29sdmUobnVsbCk7CiAgICB9LCBGSUxFX0NIQU5HRV9USU1FT1VUX01TKTsKICB9KTsKCiAgLy8gV2FpdCBmb3IgdGhlIHVzZXIgdG8gcGljayB0aGUgZmlsZXMuCiAgY29uc3QgZmlsZXMgPSB5aWVsZCB7CiAgICBwcm9taXNlOiBQcm9taXNlLnJhY2UoW3BpY2tlZFByb21pc2UsIHRpbWVvdXRQcm9taXNlLCBjYW5jZWxQcm9taXNlXSksCiAgICByZXNwb25zZTogewogICAgICBhY3Rpb246ICdzdGFydGluZycsCiAgICB9CiAgfTsKCiAgaWYgKCFmaWxlcykgewogICAgcmV0dXJuIHsKICAgICAgcmVzcG9uc2U6IHsKICAgICAgICBhY3Rpb246ICdjb21wbGV0ZScsCiAgICAgIH0KICAgIH07CiAgfQoKICBjYW5jZWwucmVtb3ZlKCk7CgogIC8vIERpc2FibGUgdGhlIGlucHV0IGVsZW1lbnQgc2luY2UgZnVydGhlciBwaWNrcyBhcmUgbm90IGFsbG93ZWQuCiAgaW5wdXRFbGVtZW50LmRpc2FibGVkID0gdHJ1ZTsKCiAgZm9yIChjb25zdCBmaWxlIG9mIGZpbGVzKSB7CiAgICBjb25zdCBsaSA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2xpJyk7CiAgICBsaS5hcHBlbmQoc3BhbihmaWxlLm5hbWUsIHtmb250V2VpZ2h0OiAnYm9sZCd9KSk7CiAgICBsaS5hcHBlbmQoc3BhbigKICAgICAgICBgKCR7ZmlsZS50eXBlIHx8ICduL2EnfSkgLSAke2ZpbGUuc2l6ZX0gYnl0ZXMsIGAgKwogICAgICAgIGBsYXN0IG1vZGlmaWVkOiAkewogICAgICAgICAgICBmaWxlLmxhc3RNb2RpZmllZERhdGUgPyBmaWxlLmxhc3RNb2RpZmllZERhdGUudG9Mb2NhbGVEYXRlU3RyaW5nKCkgOgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAnbi9hJ30gLSBgKSk7CiAgICBjb25zdCBwZXJjZW50ID0gc3BhbignMCUgZG9uZScpOwogICAgbGkuYXBwZW5kQ2hpbGQocGVyY2VudCk7CgogICAgb3V0cHV0RWxlbWVudC5hcHBlbmRDaGlsZChsaSk7CgogICAgY29uc3QgZmlsZURhdGFQcm9taXNlID0gbmV3IFByb21pc2UoKHJlc29sdmUpID0+IHsKICAgICAgY29uc3QgcmVhZGVyID0gbmV3IEZpbGVSZWFkZXIoKTsKICAgICAgcmVhZGVyLm9ubG9hZCA9IChlKSA9PiB7CiAgICAgICAgcmVzb2x2ZShlLnRhcmdldC5yZXN1bHQpOwogICAgICB9OwogICAgICByZWFkZXIucmVhZEFzQXJyYXlCdWZmZXIoZmlsZSk7CiAgICB9KTsKICAgIC8vIFdhaXQgZm9yIHRoZSBkYXRhIHRvIGJlIHJlYWR5LgogICAgbGV0IGZpbGVEYXRhID0geWllbGQgewogICAgICBwcm9taXNlOiBmaWxlRGF0YVByb21pc2UsCiAgICAgIHJlc3BvbnNlOiB7CiAgICAgICAgYWN0aW9uOiAnY29udGludWUnLAogICAgICB9CiAgICB9OwoKICAgIC8vIFVzZSBhIGNodW5rZWQgc2VuZGluZyB0byBhdm9pZCBtZXNzYWdlIHNpemUgbGltaXRzLiBTZWUgYi82MjExNTY2MC4KICAgIGxldCBwb3NpdGlvbiA9IDA7CiAgICB3aGlsZSAocG9zaXRpb24gPCBmaWxlRGF0YS5ieXRlTGVuZ3RoKSB7CiAgICAgIGNvbnN0IGxlbmd0aCA9IE1hdGgubWluKGZpbGVEYXRhLmJ5dGVMZW5ndGggLSBwb3NpdGlvbiwgTUFYX1BBWUxPQURfU0laRSk7CiAgICAgIGNvbnN0IGNodW5rID0gbmV3IFVpbnQ4QXJyYXkoZmlsZURhdGEsIHBvc2l0aW9uLCBsZW5ndGgpOwogICAgICBwb3NpdGlvbiArPSBsZW5ndGg7CgogICAgICBjb25zdCBiYXNlNj
