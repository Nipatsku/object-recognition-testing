# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# print(f'Hi, {name}')

from imageai.Detection import ObjectDetection
from imageai.Prediction import ImagePrediction
import os

def object_recognition(picture_path, result_count=10):
    cwd = os.getcwd()

    prediction = ImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath(os.path.join(cwd, "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
    prediction.loadModel()

    predictions, probabilities = prediction.predictImage(picture_path, result_count=result_count)
    #for eachPrediction, eachProbability in zip(predictions, probabilities):
        #print(eachPrediction, " : ", eachProbability, "%")

    return predictions, probabilities

# ImageAI object detection (https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/README.md)
def object_detection():
    cwd = os.getcwd()

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(cwd, "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(cwd, "resources/lucas.jpg"),
                                                 output_image_path=os.path.join(cwd, "output.jpg"),
                                                 minimum_percentage_probability=30)

if __name__ == '__main__':
    object_recognition('resources/lucas.jpg')
