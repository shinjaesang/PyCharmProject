import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

path_dir = './images/images/training/'
label_list = os.listdir(path_dir)
label_list.sort()
# print(label_list)

all_files = []
for i in label_list:
    path_dir = './images/images/training/{0}'.format(i)
    file_list=os.listdir(path_dir)
    file_list.sort()
    all_files.append(file_list)

# target = 9
# img = Image.open('./images/images/training/{0}/'.format(target) + all_files[target][0])
# img_arr = np.array(img)
# print(img_arr)
# print(img_arr.shape)

x_train_data=[]
y_train_data=[]
for num in label_list:
    for number in all_files[int(num)]:
        img_path='./images/images/training/{0}/{1}'.format(num,number)
        # print("load:" + img_path)
        img=Image.open(img_path)
        img_arr = np.array(img) / 255.0
        img_arr = np.reshape(img_arr, newshape=(784,1))
        x_train_data.append(img_arr)
        y_tmp = np.zeros(shape=(10))
        y_tmp[int(num)] = 1         #label -> one-hot vector
        y_train_data.append(y_tmp) #one-hot vector save to list



print(len(x_train_data))
print(len(y_train_data))

eval_files = []
for i in label_list:
    path_dir = './images/images/testing/{0}'.format(i)
    file_list=os.listdir(path_dir)
    file_list.sort()
    eval_files.append(file_list)

x_test_data=[]
y_test_data=[]
for num in label_list:
    for number in eval_files[int(num)]:
        img_path='./images/images/testing/{0}/{1}'.format(num,number)
        # print("load:" + img_path)
        img=Image.open(img_path)
        img_arr = np.array(img) / 255.0
        img_arr = np.reshape(img_arr, newshape=(784, 1))
        x_test_data.append(img_arr)
        y_tmp = np.zeros(shape=(10))
        y_tmp[int(num)] = 1  # label -> one-hot vector
        y_test_data.append(y_tmp) # one-hot vector save to list

print(len(x_test_data))
print(len(y_test_data))

x_train_data = np.reshape(x_train_data, newshape=(-1, 784))
y_train_data = np.reshape(y_train_data, newshape=(-1, 10))
x_test_data = np.reshape(x_test_data, newshape=(-1, 784))
y_test_data = np.reshape(y_test_data, newshape=(-1, 10))

input = tf.keras.Input(shape=(784), name="Input")
hidden = tf.keras.layers.Dense(512, activation="relu", name="Hidden1")(input)
output = tf.keras.layers.Dense(10, activation="softmax", name="Output")(hidden)

model = tf.keras.Model(inputs=[input], outputs=[output])
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
                optimizer=opt, metrics=['accuracy'])
model.summary()

history = model.fit(x_train_data, y_train_data, epochs=10, shuffle=True,
                            validation_data=(x_test_data, y_test_data))

plt.plot(history.history['loss'], 'b')
plt.plot(history.history['val_accuracy'], 'r')
plt.show()