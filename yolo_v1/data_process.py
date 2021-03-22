from pathlib import Path
import cv2
import numpy as np
import xml.etree.cElementTree as ET
import random


class DataGenerator:
    def __init__(self):
        self.labels = []

    def parse_xml(self, xml_path, S=7):
        tree = ET.ElementTree(file=xml_path)
        root = tree.getroot()
        y_true = np.array([[[0] * 30] * S] * S, dtype='float32')
        objects = root.findall('object')
        img_h = int(root.find('size').find('height').text)
        img_w = int(root.find('size').find('width').text)
        for obj in objects:
            c = obj.find('name').text
            if c not in self.labels:
                self.labels.append(c)
            c_index = self.labels.index(c)

            coor = obj.find('bndbox')
            x_min = int(coor.find('xmin').text) / img_w
            x_max = int(coor.find('xmax').text) / img_w
            y_min = int(coor.find('ymin').text) / img_h
            y_max = int(coor.find('ymax').text) / img_h
            x_c = (x_min + x_max) / 2
            y_c = (x_min + x_max) / 2
            w = x_max - x_min
            h = y_max - y_min
            s_row = int(y_c * S)
            s_column = int(x_c * S)
            y_true[s_row][s_column][0] = x_c
            y_true[s_row][s_column][1] = y_c
            y_true[s_row][s_column][2] = w
            y_true[s_row][s_column][3] = h
            y_true[s_row][s_column][4] = 1
            y_true[s_row][s_column][5:10] = y_true[s_row][s_column][0:5]
            y_true[s_row][s_column][10+c_index] = 1
        return y_true

    def get_batch_data(self, images_path: list, labels_path: list, img_size=(448, 448)):
        images = []
        labels = []
        for image_path, label_path in zip(images_path, labels_path):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size).astype('float32')/255
            label = self.parse_xml(label_path)
            images.append(img)
            labels.append(label)
        return np.array(images), np.array(labels)

    def generator(self, images_path: list, labels_path: list, batch_size=8):
        images_path = np.array(images_path)
        labels_path = np.array(labels_path)
        while True:
            index = list(range(len(images_path)))
            random.shuffle(index)
            for i in range(0, len(index), batch_size):
                batch_image_path = images_path[index[i:i+batch_size]]
                batch_label_path = labels_path[index[i:i+batch_size]]
                batch_image, batch_label = self.get_batch_data(
                    batch_image_path, batch_label_path)
                yield batch_image, batch_label


def decode_label(label):
    pass
