import cv2
import os
from numpy import reshape
from numpy import shape
from random import sample
import multiprocessing as mp

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam


class LoadData:
    """
    LoadData class stores and initializes dataset.

     Functions:

     __init__(self, parent_directory, true_class_name, epochs): Loads the true/target class and records the root
     directory for the dataset. Also, loads the invariant true class data into memory. The false class data are randomly
    initialized at each epoch to maintain a better generalization. Also, a single true/target class has less data than
    false classes. The randomization is used to remove decision bias for the network.

     load_images_from_folder(self, path, no_of_files, split_ratio): loads all images from a particular "path" with a
     fixed "no_of_files" and splits it into training-validation data w.r.t the "split_ratio". Returns two groups of files.

     class_loader(self, mode): Required parameter "mode". "true_class" loads target class and "false_class" loads other
     classes. Returns the specific train_files, train_labels, test_files, test_labels.

     dataset_generator(self): Combines true and false class data into one train and one test dataset. Returns the
     complete train_files, train_labels, test_files, test_labels.

    """

    def __init__(self, parent_directory, true_class_name, epochs):
        self.t_class = true_class_name
        self.p_dir = parent_directory
        self.epoch = epochs
        self.max_train_count = len(os.listdir(parent_directory+'/'+true_class_name))
        path = parent_directory + '/' + true_class_name
        self.training_images, self.testing_images = self.load_images_from_folder(path, self.max_train_count, 0.7)
        print(" Positive samples initialized")

    def load_images_from_folder(self, path, no_of_files, split_ratio):
        count = 0
        image_group_1 = []
        image_group_2 = []
        split_limit = int(no_of_files * split_ratio)
        file_names = sample(os.listdir(path), no_of_files)
        for names in file_names:
            img = cv2.imread(os.path.join(path, names), cv2.IMREAD_GRAYSCALE) / 255.0
            if img is not None and count < split_limit:
                image_group_1.append(img)
                count = count + 1
            elif img is not None and count >= split_limit:
                image_group_2.append(img)
        return image_group_1, image_group_2

    def class_loader(self, mode):
        if mode == 'true_class':
            return self.training_images, [1] * len(self.training_images), self.testing_images, [1] * len(self.testing_images)
        elif mode == 'false_class':
            training_images = []
            testing_images = []
            all_folders = os.listdir(self.p_dir)
            all_folders.remove(self.t_class)
            no_of_neg_train_samples_per_class = int(self.max_train_count / len(all_folders))
            for folder_names in all_folders:
                path = self.p_dir + '/' + folder_names
                trn_images, tst_images = self.load_images_from_folder(path, no_of_neg_train_samples_per_class, 0.7)
                training_images += trn_images
                testing_images += tst_images
            return training_images, [0] * len(training_images), testing_images, [0] * len(testing_images)

    def dataset_generator(self):
        print(" Preparing Dataset")
        pos_train_img, pos_train_lab, pos_test_img, pos_test_lab = self.class_loader('true_class')
        print(" Positive samples accessed")
        neg_train_img, neg_train_lab, neg_test_img, neg_test_lab = self.class_loader('false_class')
        print(" Negative samples loaded")
        training_data = pos_train_img + neg_train_img
        training_labels = pos_train_lab + neg_train_lab
        testing_data = pos_test_img + neg_test_img
        testing_labels = pos_test_lab + neg_test_lab
        training_data = reshape(training_data, (shape(training_data)[0], shape(training_data)[1], shape(training_data)[2], 1))
        testing_data = reshape(testing_data, (shape(testing_data)[0], shape(testing_data)[1], shape(testing_data)[2], 1))
        print(" Data Reshaped")
        print(" Training Data shape: ", shape(training_data))
        print(" Testing Data shape: ", shape(testing_data))
        return training_data, training_labels, testing_data, testing_labels

# CNN Model construction
model = Sequential()
model.add(Conv2D(32, (7, 7), input_shape=(128, 128, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

adam = Adam()
model.compile(loss='mse', optimizer=adam)
print(" CNN model compiled")

init = LoadData(parent_directory='C:\Training', true_class_name='A', epochs=2)
print(" LoadData object initialized")

exec_pool = mp.Pool()

# for i in range(1, init.epoch):
train_data, train_labels, test_data, test_labels = init.dataset_generator()
print(" Training !")

exec_pool.map(model.fit(train_data, train_labels, validation_split=0.2, shuffle=True, epochs=1),)

exec_pool.join()
exec_pool.close()
