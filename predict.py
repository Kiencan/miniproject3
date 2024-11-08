import numpy as np
import joblib
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

label_dictionary = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "A", 11: "B",12: "C", 13: "D",  14: "E", 15: "F", 16: "G", 17: "H", 18: "I", 19: "J", 20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T", 30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z"}

def predict_by_cnn(img):
    new_model = tf.keras.models.load_model('my_model.h5')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert image to grayscale
    gray = 255 - gray
    # resize image to 28x28 pixels
    gray_resized = cv2.resize(gray, (28, 28))

    # reshape image to ( , 28, 28, 1)
    gray_reshaped = gray_resized.reshape(1, 28, 28, 1)
    y_pred_test = new_model.predict(gray_reshaped)
    
    return label_dictionary[y_pred_test.argmax()]

def process_cnn(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = 255 - gray_img
    gray_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(gray_blur, kernel, iterations=3)
    edges = cv2.Canny(dilate, 100, 200)
    ret, thresh = cv2.threshold(dilate, 150, 255, cv2.THRESH_BINARY)

    contours2, hierarchy2 = cv2.findContours(thresh, cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE)
    
    for i in range(len(contours2)):
        x2, y2, w2, h2 = cv2.boundingRect(contours2[i])
        if h2 > 50:    
            character_img = img[y2:y2+h2, x2:x2+w2]

            character_predict = predict_by_cnn(character_img)

            cv2.rectangle(img, (x2, y2 ), (x2 + w2, y2 + h2), (0, 0, 0), 2)
            cv2.putText(img, character_predict, (x2+w2//2, y2+h2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    return img

def predict_by_logistic(img):
    log_model = joblib.load('logistic_model.sav')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert image to grayscale
    gray = 255 - gray
    # resize image to 28x28 pixels
    gray_resized = cv2.resize(gray, (28, 28))

    # reshape image to ( , 28, 28, 1)
    gray_reshaped = gray_resized.reshape(-1, 784)
    y_pred_test = log_model.predict(gray_reshaped)
    
    return label_dictionary[y_pred_test[0]]

def process_logistic(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = 255 - gray_img
    gray_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(gray_blur, kernel, iterations=3)
    edges = cv2.Canny(dilate, 100, 200)
    ret, thresh = cv2.threshold(dilate, 150, 255, cv2.THRESH_BINARY)

    contours2, hierarchy2 = cv2.findContours(thresh, cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE)
    
    for i in range(len(contours2)):
        x2, y2, w2, h2 = cv2.boundingRect(contours2[i])
        if h2 > 50:    
            character_img = img[y2:y2+h2, x2:x2+w2]

            character_predict = predict_by_logistic(character_img)

            cv2.rectangle(img, (x2, y2 ), (x2 + w2, y2 + h2), (0, 0, 0), 2)
            cv2.putText(img, character_predict, (x2+w2//2, y2+h2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    return img

if __name__ == "__main__":
    number_img = cv2.imread('../img/123.png')
    print(number_img)
    img = preprocess(number_img)
    plt.imshow(number_img)
    plt.show()