import cv2
import numpy as np
import os

#take images of people, use cv library to detect and then resize and crop faces
#save new face image files

class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)

    def detect(self, image, biggest_only=True):
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30, 30)
        biggest_only = True
        faces_coord = self.classifier.detectMultiScale(image,
                                                       scaleFactor=scale_factor,
                                                       minNeighbors=min_neighbors,
                                                       minSize=min_size,
                                                       flags=cv2.CASCADE_SCALE_IMAGE)
        return faces_coord

def collect_dataset():
    images = []
    labels = []
    labels_dic = {}
    people = [person for person in os.listdir("people/")]
    #i is order of files, person is name of person
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("people/" + person):
            #first arg is file name, returns a grayscale image
            img = cv2.imread("people/" + person + '/' + image, 0)
            if img is not None:
                #list of matrices
                images.append(img)
                #list of corresponding labels
                labels.append(person)

    return (images, np.array(labels), labels_dic)

def cut_faces(image, faces_coord):
    faces = []

    for (x, y, w, h) in faces_coord:
        w_rm = int(0.3 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])

    return faces

def resize(images, size=(224, 224)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size,
                                    interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size,
                                    interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm

def normalize_faces(image, faces_coord):

    faces = cut_faces(image, faces_coord)
    faces = resize(faces)

    return faces

#start here
images, labels, labels_dic = collect_dataset()
count = 0
for image in images:
    detector = FaceDetector("haarcascade_frontalface_default.xml")
    faces_coord = detector.detect(image, True)
    faces = normalize_faces(image ,faces_coord)
    #can be multiple faces in one image
    for i, face in enumerate(faces):
        cv2.imwrite('%s.jpeg' % (count), faces[i])
        count += 1
