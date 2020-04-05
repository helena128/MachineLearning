from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imutils import paths
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

''' TODO: CHANGE THIS SECTION FOR REUSE '''
COEF1 = 256
COEF2 = 370
COEF3 = 124
FILENAME1 = 'cat.1046.jpg'
FILENAME2 = 'dog.1025.jpg'
FILENAME3 = 'cat.1042.jpg'
FILENAME4 = 'cat.1003.jpg'

def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

imagePaths = sorted(list(paths.list_images('train')))
data = []
labels = []

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath, 1)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    hist = extract_histogram(image)
    data.append(hist)
    labels.append(label)

le = LabelEncoder()
labels = le.fit_transform(labels)

'''
See what was marked as 0
print(labels[0])
img=mpimg.imread(imagePaths[0])
imgplot = plt.imshow(img)
plt.show()
'''
# Split into train and test
(train_data, test_data, train_labels, test_labels) = train_test_split(np.array(data), \
	labels, test_size=0.25, random_state=9)

# Create model
model = LinearSVC(C = 0.51, random_state = 9)
model.fit(train_data, train_labels)

# Predict
predictions = model.predict(test_data)
print('Coef ', COEF1, ': ', round(model.coef_[0][COEF1], 3))
print('Coef ', COEF2, ': ', round(model.coef_[0][COEF2], 3))
print('Coef ', COEF3, ': ', round(model.coef_[0][COEF3], 3))
# print(classification_report(test_labels, predictions, target_names=le.classes_))
print('F1 score: ', f1_score(test_labels, predictions, average='macro'))

# Predict for unknown images
def predict_unknown_image(path):
	singleImage = cv2.imread('test/' + path)
	histt = extract_histogram(singleImage)
	histt2 = histt.reshape(1, -1)
	prediction = model.predict(histt2)
	print('For file ', path, ' prediction is ', prediction)

predict_unknown_image(FILENAME1)
predict_unknown_image(FILENAME2)
predict_unknown_image(FILENAME3)
predict_unknown_image(FILENAME4)