from tkinter import messagebox
from tkinter import simpledialog
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfilename

import numpy as np
import matplotlib.pyplot as plt
import os

from keras.utils.np_utils import to_categorical
from keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, Convolution2D
from keras.models import Sequential, model_from_json
import pickle

import cv2
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

# Initialize the main Tkinter window
main = tk.Tk()
main.title("Iris Recognition using Machine Learning Technique")
main.geometry("1300x1200")

global filename
global model

# Function to extract iris features from an image
def getirisfeatures(image):
    img = cv2.imread(image, 0)
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, param1=63, param2=70, minRadius=0, maxRadius=0)
    
    if circles is not None:
        height, width = img.shape
        r = 0
        mask = np.zeros((height, width), np.uint8)
        
        for i in circles[0, :]:
            cv2.circle(cimg, (i[0], i[1]), int(i[2]), (0, 0, 0))
            cv2.circle(mask, (i[0], i[1]), int(i[2]), (255, 255, 255), thickness=0)
        
        masked_data = cv2.bitwise_and(cimg, cimg, mask=mask)
        _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        crop = img[y:y+h, x:x+w]
        cv2.imwrite("test.png", crop)
    else:
        count += 1
        miss.append(image)
    
    return cv2.imread("test.png")

# Function to upload dataset
def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', tk.END)
    text.insert(tk.END, filename + " loaded\n\n")

# Function to load model
def loadModel():
    global model
    text.delete('1.0', tk.END)
    X_train = np.load('model/X.txt.npy')
    Y_train = np.load('model/Y.txt.npy')
    text.insert(tk.END, "Dataset contains total " + str(X_train.shape[0]) + " iris images\n\n")
    
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights("model/model_weights.h5")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        text.insert(tk.END, "CNN Model Loaded Successfully\n")
        
        with open('model/history.pckl', 'rb') as f:
            data = pickle.load(f)
        
        accuracy = data['accuracy']
        text.insert(tk.END, "CNN Model Prediction Accuracy = " + str(accuracy[-1]) + "\n\n")
        text.insert(tk.END, "See Console to view CNN layers\n")
        print(model.summary())
    else:
        model = Sequential()
        model.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(108, activation='softmax'))
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = model.fit(X_train, Y_train, batch_size=16, epochs=60, shuffle=True, verbose=2)
        
        model.save_weights('model/model_weights.h5')
        model_json = model.to_json()
        
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        
        with open('model/history.pckl', 'wb') as f:
            pickle.dump(hist.history, f)
        
        with open('model/history.pckl', 'rb') as f:
            data = pickle.load(f)
        
        accuracy = data['accuracy']
        text.insert(tk.END, "CNN Model Prediction Accuracy = " + str(accuracy[-1]) + "\n\n")
        text.insert(tk.END, "See Console to view CNN layers\n")
        print(model.summary())

# Function to predict iris
def predictIris():
    filename = filedialog.askopenfilename(initialdir="testSamples")
    image = getirisfeatures(filename)
    img = cv2.resize(image, (64, 64))
    img = np.array(img)
    img = img.reshape(1, 64, 64, 3)
    img = img.astype('float32')
    img = img / 255
    preds = model.predict(img)
    predict = np.argmax(preds) + 1
    print(predict)
    
    img = cv2.imread(filename)
    img = cv2.resize(img, (600, 430))
    cv2.putText(img, "Predicted Iris ID: " + str(predict), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow("Predicted Iris", img)
    cv2.waitKey(0)

# Function to plot graph
def graph():
    with open('model/history.pckl', 'rb') as f:
        data = pickle.load(f)
    
    accuracy = data['accuracy']
    loss = data['loss']
    
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro', color='yellow')
    plt.plot(accuracy, 'ro', color='blue')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('CNN Accuracy & Loss Graph')
    plt.show()

# Function to close the application
def close():
    main.destroy()

# Setting up the GUI components
font1 = ('times', 16, 'bold')
title = tk.Label(main, text='Iris Recognition using Machine Learning Technique')
title.config(bg='goldenrod', fg='white')
title.config(font=font1)
title.config(height=3, width=120)
title.place(x=0, y=5)

font2 = ('times', 12, 'bold')
text = tk.Text(main, height=20, width=150)
scroll = tk.Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font2)

font3 = ('times', 13, 'bold')
uploadButton = tk.Button(main, text="Upload Iris Dataset", command=uploadDataset, bg="#ffb3fe")
uploadButton.place(x=50, y=550)
uploadButton.config(font=font3)

modelButton = tk.Button(main, text="Generate & Load CNN Model", command=loadModel, bg="#ffb3fe")
modelButton.place(x=240, y=550)
modelButton.config(font=font3)
graphButton = tk.Button(main, text="Accuracy & Loss Graph", command=graph, bg="#ffb3fe")
graphButton.place(x=505, y=550)
graphButton.config(font=font3)

predictButton = tk.Button(main, text="Upload Iris Test Image & Recognize", command=predictIris, bg="#ffb3fe")
predictButton.place(x=730, y=550)
predictButton.config(font=font3)

exitButton = tk.Button(main, text="Exit", command=close, bg="#ffb3fe")
exitButton.place(x=1050, y=550)
exitButton.config(font=font3)

main.config(bg='grey')
main.mainloop()

