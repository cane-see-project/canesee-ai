# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 18:09:23 2019

@author: Mohamed
"""

# load yolov3 model and perform object detection
import numpy as np
import cv2
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import time

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1


def get_label(self):
    if self.label == -1:
        self.label = np.argmax(self.classes)

    return self.label


def get_score(self):
    if self.score == -1:
        self.score = self.classes[self.get_label()]

    return self.score


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


# decode predicted yhat arrays to dimintional boxes
def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    print("decode   ", netout.shape)
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if (objectness.all() <= obj_thresh): continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
            boxes.append(box)
    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    union = w1 * h1 + w2 * h2 - intersect
    return float(intersect) / union


# non-max suppression
def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


# load and prepare an image
def load_image_pixels(filename, shape):
    # load saved image
    if type(filename) == str:
        # load the image to get its shape
        image = load_img(filename)
        width, height = image.size
        # load the image with the required size
        image = load_img(filename, target_size=shape)
    # receive image from ip-webcam
    else:
        image = filename
        width, height, _ = image.shape
        image = cv2.resize(image, shape)

    # convert to numpy array
    image = img_to_array(image)
    print(image.shape)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height


# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i] * 100)
                # don't break, many labels may trigger for one box

    return v_boxes, v_labels, v_scores


# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores):
    # load the image
    if type(filename) == str:
        data = pyplot.imread(filename)
    else:
        data = filename

    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s(%.1f)" % (v_labels[i], v_scores[i])
        pyplot.text(x1, y1, label, color='red')
    # show the plot
    pyplot.show()


# rounding numbers
def rounding(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


def object_position(width, height, cx, cy):
    i, j = 0, 0
    x_unit = cx / width
    y_unit = cy / height
    if 0 < x_unit < 1/3:
        i = 0
    elif 1/3 < x_unit < 2/3:
        i = 1
    elif 2/3 < x_unit < 1:
        i = 2

    if  0 < y_unit < 1/3:
        j = 0
    elif 1/3 < y_unit < 2/3:
        j = 1
    elif 2/3 < y_unit < 1:
        j = 2

    return i, j


def detect(received_image):
    # load yolov3 model
    model = load_model('model.h5')
    # define the expected input shape for the model
    start = time.clock()
    input_w, input_h = 416, 416
    # define the probability threshold for detected objects
    class_threshold = 0.6
    # define the anchors
    anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
    # define the labels
    labels = ['شخص', 'دراجة', 'سيارة', 'دراجات نارية', 'طائرة', 'حافلة', 'قطار', 'شاحنة نقل', 'قارب',
              'إشارة ضوئية', 'صنبور الاطفاء', 'لافتة توقف', 'عداد موقف السيارات', 'مقعد', 'طائر', 'قط', 'الكلب',
              'حصان', 'خروف', 'بقرة', 'فيل', 'يتحمل', 'الحمار الوحشي', 'زرافة', 'حقيبة ظهر', 'مظلة', 'حقيبة يد',
              'ربطة عنق', 'حقيبة سفر', 'الطبق الطائر', 'زحلوقة', 'لوح التزلج', 'الكرة الرياضية', 'طائرة ورقية',
              'مضرب البيسبول', 'قفاز البيسبول', 'لوح تزلج', 'مزلجة', 'مضرب التنس', 'زجاجة', 'كأس نبيذ', 'كوب',
              'فرع', 'سكين', 'ملعقة', 'عاء', 'موز', 'تفاحة', 'ساندويتش', 'البرتقالي', 'بروكلي', 'جزرة', 'نقانق',
              'بيتزا', 'الدونات', 'كيك', 'كرسي', 'كنبة', 'النبات المحفوظ بوعاء', 'السرير', 'طاولة الطعام', 'الحمام',
              'رصد التلفزيون', 'حاسوب محمول', 'الفأر', 'التحكم عن بعد', 'لوحة المفاتيح',
              'الهاتف الخلوي', 'الميكروويف', 'فرن', 'محمصة خبز كهربائية', 'مكتب المدير', 'ثلاجة', 'كتاب', 'ساعة حائط',
              'مزهرية', 'مقص', 'دمية دب', 'مجفف شعر', 'فرشاة الأسنان']

    position = [["أعلى اليسار", "أعلى", "أعلى اليمين"],
                ["اليسار", "الوسط", "اليمين"],
                ["أسفل اليسار", "أسفل", "أسفل اليمين"]]

    # define our new photo
    # load and prepare image , return image and actual size
    image, image_w, image_h = load_image_pixels(received_image, (input_w, input_h))
    print("actual image size  (%3d, %3d)" % (image_w, image_h))
    """ make prediction ,
    yhat is 3 arrays predict both the bounding boxes and class labels but are encoded
    based on anchor boxes where each box return as ashape
     {{{ [centerX, centerY, width, height, objectExist ,class1,class1,....,Cn] }}} """
    yhat = model.predict(image)
    # summarize the shape of the list of arrays
    print([a.shape for a in yhat])

    boxes = list()
    for i in range(len(yhat)):
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
        print("shape ", i," >>>",yhat[i][0].shape)
    # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    # suppress non-maximal boxes
    do_nms(boxes, 0.5)
    # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
    place = []

    for i, box in enumerate(v_boxes):
        centroidX = (box.xmax + box.xmin) / 2
        centroidY = (box.ymax + box.ymin) / 2
        x, y = object_position(image_w, image_h, centroidX, centroidY)
        print("%-15s %-3.1f  [%3d, %3d]  %12s" % (v_labels[i], rounding(v_scores[i], 1), centroidX, centroidY, position[y][x]))
        place.append(position[y][x])

    end = time.clock()
    print("time is : ", end-start)

detect("photos/street5.jpg")
