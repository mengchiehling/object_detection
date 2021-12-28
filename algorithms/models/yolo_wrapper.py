import os
from typing import Tuple, Union, Dict

import cv2
import numpy as np
from cv2.dnn import readNetFromDarknet, DNN_BACKEND_OPENCV, DNN_TARGET_CPU, blobFromImage, NMSBoxes

from algorithms.utils.utils_basic import load_model_key
from algorithms.io.path_definition import get_project_dir

yolo_key = load_model_key('YOLO')


class YoloInferenceWrapper:

    def __init__(self):

        """
        initialize YOLOv3 network from opencv 4.2.0. Works with AlexeyAB/Yolov3
        https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html

        For Yolov4-csp, opencv 4.5.0 works.

        """

        _cfg, _weights = self.load_yolo_meta(DoC=yolo_key)
        self.net, self.classes, self.output_layers, self.colors, self.net_height, \
        self.net_width = self.load_cv2_net(_cfg=_cfg, _weights=_weights)

    def predict(self, img_input: np.ndarray) -> Tuple[str, float]:

        """

        Args:
            img_input: a uint8 nd.ndarray with dimension H, W, 3
        Returns:
            a tuple of brand name and its associated probability
        """

        assert img_input.shape[2] == 3, 'the image does not have the right color channels'

        outs, height, width = self.detection(np.uint8(img_input), self.net, self.output_layers,
                                             self.net_height, self.net_width)
        boxes, class_scores, class_ids = self.object_filter(outs, height, width)
        detections = self.nms_filter(boxes, class_scores, class_ids)

        labels = []

        for class_id, outputs in detections.items():
            boxes = outputs['boxes']
            class_scores = outputs['class_scores']

            for box, class_score in zip(boxes, class_scores):
                labels.append((str(self.classes[class_id]), class_score))

        if len(labels) == 0:
            label = "Not detected"
            label_proba = 0
        else:
            labels.sort(key=lambda x: x[1], reverse=True)
            label = labels[0][0]
            label_proba = np.round(labels[0][1] * 100, 1)

        return label, label_proba

    def load_yolo_meta(self, DoC: str) -> Tuple:

        '''

        Each model is uniquely specified by two labels:
        the date at which it is created (DoC) and the structure (structure)
        For more details, please check either AlexeyAB/darknet or pjreddie/darknet in github
        pjreddie/darknet no longer updates so AlexeyAB/darknet might be a better place to go

        https://github.com/AlexeyAB/darknet

        Args:
            DoC: Date of model training
        Returns:
            A tuple of YOLO network configuration and trained weights

        '''

        dir_model = os.path.join(get_project_dir(), 'data', 'app', 'models')

        model_dest = os.path.join(dir_model, DoC)
        assert os.path.isdir(model_dest), f"directory {model_dest} does not exist"

        _cfg = f"{model_dest}/fashion.cfg"
        assert os.path.isfile(_cfg), f".cfg file {_cfg} does not exist"

        _weights = f"{model_dest}/fashion.weights"
        assert os.path.isfile(_weights), f".weights file {_weights} does not exist"

        return _cfg, _weights

    def load_cv2_net(self, _cfg: str, _weights: str) -> Tuple:

        '''
        This function takes the .data, .cfg, and .weights files as inputs
        and return the network, classes of output, network layers, and dimensionality of the image input in YOLO

        Args:
            _cfg: yolo neural network architecture
            _weights: yolo neural network weights
        Returns:
            cv2.dnn_Net, object classes, names of output layers, random colors for bounding box, image height, image width
        '''

        # net = cv2.dnn.readNet(_weights, _cfg)
        net = readNetFromDarknet(_cfg, _weights)
        net.setPreferableBackend(DNN_BACKEND_OPENCV)
        net.setPreferableTarget(DNN_TARGET_CPU)

        layer_names = net.getLayerNames()
        # a list of string specifying the names of output layers
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        classes_names = os.path.join(os.path.dirname(_cfg), "classes.names")

        with open(classes_names, 'r') as f:
            classes = f.read().splitlines()
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        with open(_cfg, 'r') as f:
            x = f.readlines()
        height = int([a for a in x if ('height' in a)][0].rstrip().split("=")[-1])
        width = int([a for a in x if ('width' in a)][0].rstrip().split("=")[-1])

        return net, classes, output_layers, colors, height, width

    def detection(self, img_file: Union[str, 'np.ndarray[np.uint8]'], net: cv2.dnn_Net, output_layers,
                  net_height: int, net_width: int):

        '''
        Args:
            img_file: img for object detection
            net: cv2 network
            output_layers: YOLO output layers
            net_height: image height
            net_width: image width
        Returns:
            raw detected objects for each layer, image height, image width
        '''

        if isinstance(img_file, str):
            img = cv2.imread(img_file)
        else:
            img = img_file

        height, width, _ = img.shape
        blob = blobFromImage(img, 1 / 255.0, (net_height, net_width), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(blob)
        # object detection (n_layers, n_objects, predictions)
        outs = net.forward(output_layers)

        return outs, height, width

    def get_box(self, height, width, detection):

        center_x = int(detection[0] * width)
        center_y = int(detection[1] * height)
        w = int(detection[2] * width)
        h = int(detection[3] * height)
        # Rectangle coordinates
        x = int(center_x - w / 2)
        y = int(center_y - h / 2)

        box = [x, y, w, h]

        return box

    def object_filter(self, outs, height, width, obj_conf_thresh: float = 0.25) -> Tuple:

        '''
        Filters out detected object with object confidence lower than threshold obj_conf_thresh

        Args:
            outs:
            height: image height
            width: image width
            obj_conf_thresh: minimum object confidence to be considered as a real object
        Returns:
            object detection bounding boxes, class confidence scores, and class ids
        '''

        class_ids = []
        class_scores = []
        boxes = []

        for out in outs:
            for detection in out:
                obj_confidence = detection[4]
                scores = obj_confidence * detection[5:]
                class_id = np.argmax(scores)
                if obj_confidence > obj_conf_thresh:
                    # Object detected
                    box = self.get_box(height, width, detection)
                    boxes.append(box)
                    class_scores.append(float(np.max(scores)))
                    class_ids.append(class_id)

        return np.array(boxes), np.array(class_scores), class_ids

    def nms_filter(self, boxes, class_scores, class_ids, score_threshold=0.5, nms_threshold=0.4) -> Dict:

        '''
        Performs non maximum suppression given boxes and corresponding scores.
        Parameters descriptions follow https://www.kite.com/python/docs/cv2.dnn.NMSBoxes
        Args:
            boxes:
            class_scores: a set of corresponding confidences
            class_ids:
            score_threshold: a threshold used to filter boxes by score.
            nms_threshold: or called as iou threshold. a threshold used in non maximum suppression.
            Overlapped bounding boxes with IOU overlap larger than nms_threshold will be considered as the same object,
            then only the box with highest confidence will be kept
        Returns:

        '''

        get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

        # class by class
        unique_class_ids = np.unique(class_ids)

        detections = {}

        for class_id in unique_class_ids:

            ilocs = get_indexes(class_id, class_ids)
            cls_boxes = boxes[ilocs]
            cls_class_scores = class_scores[ilocs]

            try:
                indexes = NMSBoxes(cls_boxes.tolist(), cls_class_scores.tolist(), score_threshold=score_threshold,
                                   nms_threshold=nms_threshold).flatten()
            except:
                continue

            detections[class_id] = {}
            detections[class_id]['boxes'] = cls_boxes[indexes]
            detections[class_id]['class_scores'] = cls_class_scores[indexes]

        return detections
