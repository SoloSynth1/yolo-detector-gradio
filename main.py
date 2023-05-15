import cv2
from environs import Env
import gradio as gr
import numpy as np

env = Env()
env.read_env()  # read .env, if exists

CLASS_LABEL_PATH = env("CLASS_LABEL_PATH")
MODEL_PATH = env("MODEL_PATH")
CONFIG_PATH = env("CONFIG_PATH")
DETECTION_THRESHOLD = env.float("DETECTION_THRESHOLD")


# setting up model-related assets
net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, MODEL_PATH)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = [line.strip() for line in open(CLASS_LABEL_PATH, 'r').readlines()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))


def image_classifier(inp):
    """Detect objects in an image.

    It will first process the image to 414x414. Then YOLOv3-based model is used to get the detection results.

    Args:
        inp: A numpy array that represents the image.

    Returns:
        A list of dictionaries containing objects detected. Each dict will contain the object's class label, class_id,
        score, and the bounding box positions.
    """
    # get image dimensions for scaling/de-scaling
    height, width, channels = inp.shape

    # feed image into model and get output
    blob = cv2.dnn.blobFromImage(inp, 1. / 255., (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # parsing the results, filter all results with score < threshold
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]
            if confidence >= DETECTION_THRESHOLD:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # non-maximum suppression
    nms_indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)
    nms_boxes, nms_classes = (np.take(arr, nms_indices, axis=0) for arr in [boxes, class_ids])

    # font = cv2.FONT_HERSHEY_PLAIN

    bbox_thick = int(0.6 * (height + width) / 600)
    for i, box in enumerate(nms_boxes):
        x, y, w, h = (int(value) for value in box)

        # c1, c2 are the bounding box dimensions
        c1, c2 = (x, y), (x + w, y + h)

        # label = str()
        bbox_mess = '%s: %.2f' % (classes[nms_classes[i]], confidences[i])

        # calculating font size
        t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=bbox_thick // 2)[0]

        # c3 is the label box dimensions
        c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)

        color = colors[nms_classes[i]]
        cv2.rectangle(inp, c1, c2, color, bbox_thick)   # bounding box

        cv2.rectangle(inp, c1, (c3[0], c3[1]), color, -1)  # filled label box

        cv2.putText(inp, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)  # label with score
    return inp


if __name__ == "__main__":
    demo = gr.Interface(fn=image_classifier, inputs="image", outputs="image", live=True,
                        title="garbage-detector-gradio")
    demo.launch(server_name="0.0.0.0")
