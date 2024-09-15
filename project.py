import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Step 1: Load and preprocess the data
def load_data():
    train_dir = r'C:\Users\Rohan Uplenchwar\OneDrive\Desktop\PYTHON\Seismic-Detection\archive (2)\Seismic_data\train'  # Replace with your actual path
    test_dir = r'C:\Users\Rohan Uplenchwar\OneDrive\Desktop\PYTHON\Seismic-Detection\archive (2)\Seismic_data\test'    # Replace with your actual path

    # Data augmentation and image preprocessing
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_data = datagen.flow_from_directory(
        train_dir, 
        target_size=(150, 150), 
        batch_size=32, 
        class_mode='binary'  # Change to 'categorical' if multi-class classification
    )
    
    test_data = datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    return train_data, test_data

# Step 2: Build a CNN model
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Change to 'softmax' if multi-class

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 3: Train the model
def train_model(model, train_data, test_data, epochs=20):
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                      patience=3, 
                                      verbose=1, 
                                      factor=0.5, 
                                      min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=5, 
                                   verbose=1, 
                                   restore_best_weights=True)

    history = model.fit(
        train_data,
        steps_per_epoch=train_data.samples // train_data.batch_size,
        epochs=epochs,
        validation_data=test_data,
        validation_steps=test_data.samples // test_data.batch_size,
        callbacks=[lr_reduction, early_stopping]
    )
    return history

# Step 4: Plot training results
def plot_results(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Step 5: Load and preprocess the image for prediction
def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize((150, 150))  # Resize to match model input
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Step 6: Create a GUI for testing images
def create_gui(model, history):
    def load_image():
        file_path = filedialog.askopenfilename()
        if file_path:
            img = preprocess_image(file_path)
            prediction = model.predict(img)
            label = 'Seismic Event Detected' if prediction[0][0] > 0.5 else 'No Event Detected'
            val_accuracy = history.history['val_accuracy'][-1] * 100  # Retrieve validation accuracy
            result_label.config(text=f'Prediction: {label}\nModel Validation Accuracy: {val_accuracy:.2f}%')

            img = Image.open(file_path)
            img = img.resize((150, 150))
            img = ImageTk.PhotoImage(img)
            image_label.config(image=img)
            image_label.image = img

    # Create the GUI window
    window = tk.Tk()
    window.title("Seismic Event Detection")

    upload_button = tk.Button(window, text="Upload Image", command=load_image)
    upload_button.pack()

    result_label = tk.Label(window, text="Result will be shown here")
    result_label.pack()

    image_label = tk.Label(window)
    image_label.pack()

    window.mainloop()

# Main function to run training and testing
if __name__ == '__main__':
    # Load data
    train_data, test_data = load_data()

    # Build model
    model = build_model()

    # Train model
    history = train_model(model, train_data, test_data)

    # Plot results
    plot_results(history)

    # Create UI for testing
    create_gui(model, history)
