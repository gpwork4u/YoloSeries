from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from pathlib import Path
from math import ceil
from data_process import DataGenerator

import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import os


class Yolo_V1:

    def __init__(self, width=448, height=448, channels=3):
        self.build_model(width, height, channels)
        self.data_gen = DataGenerator()

    def build_model(self, width, height, channels):
        inputs = Input(shape=(height, width, channels))
        x = Conv2D(filters=64, kernel_size=7,
                   strides=2, activation='relu', padding='same')(inputs)
        x = MaxPooling2D(2, strides=2)(x)
        x = Conv2D(filters=192, kernel_size=3, strides=1,
                   activation='relu', padding='same')(x)
        x = MaxPooling2D(2, strides=2)(x)
        x = Conv2D(filters=128, kernel_size=1, strides=1,
                   activation='relu', padding='same')(x)
        x = Conv2D(filters=256, kernel_size=3, strides=1,
                   activation='relu', padding='same')(x)
        x = Conv2D(filters=256, kernel_size=1, strides=1,
                   activation='relu', padding='same')(x)
        x = Conv2D(filters=512, kernel_size=3, strides=1,
                   activation='relu', padding='same')(x)
        x = MaxPooling2D(2, strides=2)(x)
        for i in range(4):
            x = Conv2D(filters=256, kernel_size=1,
                       strides=1, activation='relu', padding='same')(x)
            x = Conv2D(filters=512, kernel_size=3,
                       strides=1, activation='relu', padding='same')(x)
        x = Conv2D(filters=512, kernel_size=1, strides=1,
                   activation='relu', padding='same')(x)
        x = Conv2D(filters=1024, kernel_size=3,
                   strides=1, activation='relu', padding='same')(x)
        x = MaxPooling2D(2, strides=2)(x)
        for i in range(2):
            x = Conv2D(filters=512, kernel_size=1,
                       strides=1, activation='relu', padding='same')(x)
            x = Conv2D(filters=1024, kernel_size=3,
                       strides=1, activation='relu', padding='same')(x)
        x = Conv2D(filters=1024, kernel_size=3,
                   strides=1, activation='relu', padding='same')(x)
        x = Conv2D(filters=1024, kernel_size=3,
                   strides=2, activation='relu', padding='same')(x)
        x = Conv2D(filters=1024, kernel_size=3,
                   strides=1, activation='relu', padding='same')(x)
        x = Conv2D(filters=1024, kernel_size=3,
                   strides=1, activation='relu', padding='same')(x)
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dense(7*7*30, activation='relu')(x)
        outputs = Reshape((7, 7, 30))(x)
        self.model = Model(inputs, outputs)

    def prepare_data(self, train_path: str, valid_path: str = None, image_type='jpg'):
        labels_path = list(Path(train_path).glob('labels/*'))
        self.train_labels_path = [str(f) for f in labels_path]
        self.train_images_path = [os.path.join(
            train_path, 'images', f.name.replace('xml', image_type)) for f in labels_path]

    def caculate_iou(self, bbox, ground_truth):
        bbox_x_min = bbox[..., 0] - bbox[..., 2]/2
        bbox_x_max = bbox[..., 0] + bbox[..., 2]/2
        bbox_y_min = bbox[..., 1] - bbox[..., 3]/2
        bbox_y_max = bbox[..., 1] + bbox[..., 3]/2
        gbox_x_min = ground_truth[..., 0] - ground_truth[..., 2]/2
        gbox_x_max = ground_truth[..., 0] + ground_truth[..., 2]/2
        gbox_y_min = ground_truth[..., 1] - ground_truth[..., 3]/2
        gbox_y_max = ground_truth[..., 1] + ground_truth[..., 3]/2

        inter_x_min = K.max([bbox_x_min, gbox_x_min], axis=0)
        inter_x_max = K.min([bbox_x_max, gbox_x_max], axis=0)
        inter_y_min = K.max([bbox_y_min, gbox_y_min], axis=0)
        inter_y_max = K.min([bbox_y_max, gbox_y_max], axis=0)
        inter = (inter_x_max - inter_x_min)*(inter_y_max - inter_y_min)
        union = (bbox_x_max - bbox_x_min)*(bbox_y_max - bbox_y_min) + \
            (gbox_x_max - gbox_x_min)*(gbox_y_max - gbox_y_min) - inter
        zeros_array = K.zeros_like(union)
        iou = inter / union
        return K.max([inter / union, zeros_array], axis=0)

    def loss(self, y_true, y_pred):
        loss = 0
        LANDA_coord = 5
        LANDA_noobj = 0.5
        bboxs = [y_pred[:, :, :, 0:5], y_pred[:, :, :, 5:10]]
        ground_truth = [y_true[:, :, :, 0:5], y_true[:, :, :, 5:10]]
        response_array = y_true[:, :, :, 4]
        noobj_array = K.ones_like(response_array)
        category = y_true[:, :, :, 10:]
        category_p = y_pred[:, :, :, 10:]

        for g_box in ground_truth:
            bbox_index = K.argmax([self.caculate_iou(bboxs[0], g_box),
                                   self.caculate_iou(bboxs[1], g_box)], axis=0)
            for i, bbox in enumerate(bboxs):
                x = g_box[:, :, :, 0]
                y = g_box[:, :, :, 1]
                w = g_box[:, :, :, 2]
                h = g_box[:, :, :, 3]
                c = g_box[:, :, :, 4]
                x_p = bbox[:, :, :, 0]
                y_p = bbox[:, :, :, 1]
                w_p = bbox[:, :, :, 2]
                h_p = bbox[:, :, :, 3]
                c_p = bbox[:, :, :, 4]
                bbox_response_array = K.cast(K.equal(bbox_index, i), 'float32')
                loss += LANDA_coord * K.sum(response_array *
                                            bbox_response_array *
                                            (K.square(x - x_p) + K.square(y - y_p)
                                             + K.square(K.sqrt(w) -
                                                        K.sqrt(w_p))
                                             + K.square(K.sqrt(h) - K.sqrt(h_p))))
                loss += K.sum(response_array *
                              bbox_response_array * (K.square(c_p - c)))
                loss += K.sum(K.expand_dims(response_array, axis=-1) *
                              K.square(K.softmax(category) - K.softmax(category_p)))
        loss += LANDA_noobj * noobj_array * K.square(c_p - c)
        return loss

    def fit(self, lr=0.0001, epochs=10, batch_size=8, ** kwargs):
        train_generator = self.data_gen.generator(
            self.train_images_path, self.train_labels_path, batch_size=batch_size)
        steps_per_epoch = ceil(len(self.train_images_path)/batch_size)
        self.model.compile(loss=self.loss, optimizer=Adam(lr=0.0001))
        self.model.fit(train_generator, epochs=epochs,
                       steps_per_epoch=steps_per_epoch, **kwargs)


if __name__ == '__main__':
    yolov1 = Yolo_V1()
    yolov1.prepare_data(r'D:\project\yolo\data_voc\VOCdevkit\VOC2012')
    yolov1.fit(batch_size=4)
