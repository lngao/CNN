import cv2
import os

from numpy import shape, ones, uint8, float32, reshape, ndarray
from random import sample, choice
from numpy.random import randint, rand
from multiprocessing import Process

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping


class LoadData:
    """
    LoadData class stores and initializes dataset.

     Functions:

     __init__(self, parent_directory, true_class_name, epochs): Loads true and false class data from parent folder
     and trains for said epochs

     load_images_from_folder(self, path, no_of_files, split_ratio): Loads files from path and return two lists w.r.t.
     split ratio. Files loaded limited to no_of_files

     class_loader(self, mode): Required parameter "mode". "true_class" loads target class and "false_class" loads other
     classes. Returns the specific train_files, train_labels, test_files, test_labels.

     dataset_extender(self, sample_data, no_of_samples_to_gen, rotation=True, move=True, scale=True, thickness=True): This
     generates data from a given sample dataset using affine transformation, scaling (different on each axis) and varying
     the thickness of the stroke. Returns the generated dataset.

     artifact_remover(extended_image, original_size): Centers the modified image and resizes to original input size.
     Returns corrected image.

     dataset_generator(self): Combines true and false class data into one train and one test dataset. Returns the
     complete train_files, train_labels, test_files, test_labels.
    """

    def __init__(self, parent_directory, true_class_name, epochs, augment_data_flag, augment_data_count):
        self.t_class = true_class_name
        self.p_dir = parent_directory
        self.epoch = epochs
        self.augment_flag = augment_data_flag
        self.augment_count = augment_data_count
        self.max_train_count = len(os.listdir(parent_directory+'/'+true_class_name))
        path = parent_directory + '/' + true_class_name
        self.p_training_images, self.p_testing_images = self.load_images_from_folder(path, self.max_train_count, 0.7)

    @staticmethod
    def load_images_from_folder(path, no_of_files, split_ratio):
        count = 0
        image_group_1 = []
        image_group_2 = []
        split_limit = int(no_of_files * split_ratio)
        file_names = sample(os.listdir(path), no_of_files)
        for names in file_names:
            img = cv2.imread(os.path.join(path, names), cv2.IMREAD_GRAYSCALE)
            img = (255 - img) / 255.0                                               # invert and normalize image
            if img is not None and count < split_limit:
                image_group_1.append(img)
                count = count + 1
            elif img is not None and count >= split_limit:
                image_group_2.append(img)
        return image_group_1, image_group_2

    def class_loader(self, mode):
        if mode == 'true_class':
            print(' Positive samples loaded')
            return self.p_training_images, [1]*len(self.p_training_images), self.p_testing_images, [1]*len(self.p_testing_images)
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
            print(" Negative samples loaded")
            return training_images, [-1]*len(training_images), testing_images, [-1]*len(testing_images)

    def dataset_extender(self, sample_data, no_of_samples_to_gen, rotation=True, move=True, scale=True, thickness=True):
        index = randint(0, len(sample_data), no_of_samples_to_gen)
        new_img_set = []
        for i in index:
            img = sample_data[i]
            rows, cols = img.shape
            if scale is True:
                scaling_factor = choice([0.4, 0.6, 0.8, 1.2, 1.4])
                height, width = shape(img)
                img = cv2.resize(img, (int(scaling_factor * width), int(scaling_factor * height)),
                                 interpolation=cv2.INTER_CUBIC)
                img = self.artifact_remover(img, (rows, cols))
            if rotation is True:
                rot_angle = float(randint(-35, 35) + rand(1))
                rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rot_angle, 1)
                img = cv2.warpAffine(img, rotation_matrix, (cols, rows))
            if move is True:
                x, y = randint(-15, 15, 2)
                trans_matrix = float32([[1, 0, x], [0, 1, y]])
                img = cv2.warpAffine(img, trans_matrix, (cols, rows))
            if thickness is True:
                op_mode = randint(1, 3)
                kernel = ones((randint(2, 4), randint(2, 4)), uint8)
                if op_mode == 1:
                    img = cv2.erode(img, kernel, iterations=1)
                else:
                    img = cv2.dilate(img, kernel, iterations=1)
            new_img_set.append(img)
        return new_img_set

    @staticmethod
    def artifact_remover(extended_image, original_size):
        fixed_image = ndarray(original_size)
        rows_original, cols_original = original_size
        rows_extended, cols_extended = shape(extended_image)
        row_range_upper = int((rows_extended + rows_original) / 2)
        col_range_upper = int((cols_extended + cols_original) / 2)
        col_range_lower = abs(int((cols_extended - cols_original) / 2))
        row_range_lower = abs(int((rows_extended - rows_original) / 2))
        if rows_extended > rows_original or cols_extended > cols_original:
            fixed_image = extended_image[row_range_lower:row_range_upper, col_range_lower:col_range_upper]
        else:
            fixed_image[row_range_lower:row_range_upper, col_range_lower:col_range_upper] = extended_image
        return fixed_image

    def dataset_generator(self):
        print(" Preparing Dataset")
        p_train_img, p_train_lab, p_test_img, p_test_lab = self.class_loader('true_class')
        n_train_img, n_train_lab, n_test_img, n_test_lab = self.class_loader('false_class')
        if self.augment_flag is True:
            aug_p_train_image = self.dataset_extender(p_train_img, self.augment_count[0])
            print(" Generated " + str(self.augment_count[0]) + " augmented positive data samples")
            aug_n_train_image = self.dataset_extender(n_train_img, self.augment_count[1])
            print(" Generated " + str(self.augment_count[1]) + " augmented negative data samples")
            training_data = p_train_img + aug_p_train_image + n_train_img + aug_n_train_image
            training_labels = p_train_lab + [1]*self.augment_count[0] + n_train_lab + [0]*self.augment_count[1]
        else:
            training_data = p_train_img + n_train_img
            training_labels = p_train_lab + n_train_lab
        testing_data = p_test_img + n_test_img
        testing_labels = p_test_lab + n_test_lab
        training_data = reshape(training_data, (shape(training_data)[0], shape(training_data)[1], shape(training_data)[2], 1))
        testing_data = reshape(testing_data, (shape(testing_data)[0], shape(testing_data)[1], shape(testing_data)[2], 1))
        print(" Data Reshaped")
        print(" Training Data shape: ", shape(training_data))
        print(" Testing Data shape: ", shape(testing_data))
        return training_data, training_labels, testing_data, testing_labels


class CNN(LoadData):
    """
    __init__(self, parent_directory, true_class_name, output_folder, epoch_count)

    load_neural_network(): Loads a predefined network

    callback_set(output_folder): Sets call back. Using early termination on validation accuracy and saving model at each
    iteration in set output folder

    train_network(epoch_count, callbacks_list): Trains network for set epoch_count with all callbacks in callback_list

    evaluate_network(test_data, test_labels): Evaluates and prints accuracy of testing data to file
    """

    def __init__(self, parent_directory, true_class_name, output_folder, epoch_count, augment_data_flag, augment_data_count):
        LoadData.__init__(self, parent_directory, true_class_name, epoch_count, augment_data_flag, augment_data_count)
        self.model = Sequential()
        self.load_neural_network()
        callback_list = self.callback_set(output_folder)
        self.train_network(epoch_count, callback_list)

    def load_neural_network(self):
        # CNN Model construction
        self.model.add(Conv2D(64, (7, 7), input_shape=(128, 128, 1)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (5, 5)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(128, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(70))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.55))
        self.model.add(Dense(25))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.55))
        self.model.add(Dense(1))
        self.model.add(Activation('tanh'))

        adam = Adam()
        self.model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
        print(" CNN model compiled")

    @staticmethod
    def callback_set(output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        file_path = output_folder + "/weights - {val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=False,
                                     save_weights_only=True, mode='auto', period=1)
        early_stop = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=2, verbose=1, mode='auto')
        return [checkpoint, early_stop]

    def train_network(self, epoch_count, callbacks_list):
        for _ in range(0, epoch_count):
            train_data, train_labels, test_data, test_labels = super().dataset_generator()
            self.model.fit(train_data, train_labels, validation_split=0.2, shuffle=True, epochs=1,
                           callbacks=callbacks_list)
            self.evaluate_network(test_data, test_labels)

    def evaluate_network(self, test_data, test_labels):
        score = self.model.evaluate(test_data, test_labels)
        print('Accuracy :', score)
        try:
            with open("Accuracy_log.txt", mode="a") as file:
                file.write("\n"+super().t_class+str(score[1]))
        except PermissionError:
            print(" Accuracy log write failed ! Use a different path or change permissions")
        finally:
            print(" Accuracy log updated !")

# main()
if __name__ == "__main__":
    root = 'C:\Training'
    all_classes = os.listdir(root)
    for classes in all_classes:
        p = Process(target=CNN, args=(root , classes, "./output/" + classes, 100, True, (5000, 20000)))
        p.start()
