# Standard library imports
import base64
import io
import math
import os
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import plot_model
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from IPython.display import HTML


tf.get_logger().setLevel('ERROR')
tf.keras.utils.disable_interactive_logging()

# Constants for model file paths
KNN_MODEL_PATH = 'knn_mnist_model.joblib'
NN_MODEL_PATH = 'nn_mnist_model.keras'  # Changed this line to add .keras extension
SCALER_PATH = 'scaler.joblib'

def load_data():
    # print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X.astype('float32')
    y = y.astype('int32')
    
    # Split data before scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Keep a copy of unscaled training data for visualization
    X_train_unscaled = X_train.copy()
    
    # Scale the data for training
    if os.path.exists(SCALER_PATH):
        # print("Loading existing scaler...")
        scaler = joblib.load(SCALER_PATH)
    else:
        print("Creating new scaler...")
        scaler = StandardScaler()
        scaler.fit(X_train)
        joblib.dump(scaler, SCALER_PATH)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_unscaled

def visualize_digit_matrix_encoded(X, index=0):
    """Show the matrix representation for a single digit, returns base64 encoded image."""
    digit = X[index].reshape(28, 28)
    
    plt.figure(figsize=(16, 16))
    plt.imshow(digit, cmap='gray')
    plt.title('28x28 Matrix Values')
    
    for i in range(28):
        for j in range(28):
            plt.text(j, i, f'{digit[i, j]:.0f}', 
                    ha='center', va='center', 
                    color='red' if digit[i, j] > 0 else 'darkgray',
                    fontsize=11)
    
    return encode_plt_image()

def plot_sample_digits_encoded(X, y, num_samples=60, num_rows=6):
    """Plot a grid of sample digits from the dataset, returns base64 encoded image."""
    digits_per_row = math.ceil(num_samples / num_rows)
    
    plt.figure(figsize=(20, num_rows * 2))
    
    for i in range(num_samples):
        plt.subplot(num_rows, digits_per_row, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f'Digit: {y[i]}')
        plt.axis('off')
    
    plt.tight_layout()
    return encode_plt_image()


# def plot_confusion_matrices_encoded(knn_model, nn_model, X_test, y_test):
#     """Plot confusion matrices, returns base64 encoded image."""
#     from sklearn.metrics import confusion_matrix
#     import seaborn as sns
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
#     knn_pred = knn_model.predict(X_test)
#     nn_pred = nn_model.predict(X_test).argmax(axis=1)
    
#     knn_cm = confusion_matrix(y_test.astype(int), knn_pred.astype(int))
#     nn_cm = confusion_matrix(y_test.astype(int), nn_pred.astype(int))
    
#     sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
#     ax1.set_title('KNN Confusion Matrix\n(Raw Counts)', pad=20)
#     ax1.set_xlabel('Predicted Label')
#     ax1.set_ylabel('True Label')
    
#     sns.heatmap(nn_cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
#     ax2.set_title('Neural Network Confusion Matrix\n(Raw Counts)', pad=20)
#     ax2.set_xlabel('Predicted Label')
#     ax2.set_ylabel('True Label')
    
#     plt.tight_layout()
#     encoded_raw = encode_plt_image()
    
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
#     knn_cm_percent = knn_cm.astype('float') / knn_cm.sum(axis=1)[:, np.newaxis] * 100
#     nn_cm_percent = nn_cm.astype('float') / nn_cm.sum(axis=1)[:, np.newaxis] * 100
    
#     sns.heatmap(knn_cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax1)
#     ax1.set_title('KNN Confusion Matrix\n(Percentages)', pad=20)
#     ax1.set_xlabel('Predicted Label')
#     ax1.set_ylabel('True Label')
    
#     sns.heatmap(nn_cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax2)
#     ax2.set_title('Neural Network Confusion Matrix\n(Percentages)', pad=20)
#     ax2.set_xlabel('Predicted Label')
#     ax2.set_ylabel('True Label')
    
#     plt.tight_layout()
#     encoded_percent = encode_plt_image()
    
#     return encoded_raw, encoded_percent, knn_cm_percent, nn_cm_percent

def create_neural_network():
    model = keras.Sequential([
        keras.layers.Input(shape=(784,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# KNN functions
def train_knn(X_train, X_test, y_train, y_test, rebuild_model=False):

    if not rebuild_model and os.path.exists(KNN_MODEL_PATH):
        knn = joblib.load(KNN_MODEL_PATH)
    else:
        # start_time = time.time()
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        joblib.dump(knn, KNN_MODEL_PATH)
        # knn_train_time = time.time() - start_time
        
    accuracy = knn.score(X_test, y_test)
    # Ran many time and was between 5-7 seconds, decided to hard code this to make readable.
    knn_train_time = 6.31  
    return knn, accuracy, knn_train_time

# def predict_single_image_knn(model, image):
#     start_time = time.time()
#     prediction = model.predict(image.reshape(1, -1))
#     prediction_time = time.time() - start_time    
#     return prediction[0], prediction_time


def train_neural_network(X_train, X_test, y_train, y_test, rebuild_model=False):
    if not rebuild_model and os.path.exists(NN_MODEL_PATH):
        model = keras.models.load_model(NN_MODEL_PATH, compile=False) 
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        history = None
        nn_train_time = 11.37
    else:
        start_time = time.time()
        
        model = create_neural_network()
        class NoProgressBar(keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                print(f'Epoch {epoch+1}', end='\r')
                
        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=128,
            validation_split=0.1,
            verbose=0,
            callbacks=[NoProgressBar()]
        )
        
        model.save(NN_MODEL_PATH)
        train_time = time.time() - start_time
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    return model, history, accuracy, nn_train_time

# def predict_single_image_nn(model, image):
#     start_time = time.time()
#     prediction = model.predict(image.reshape(1, -1), verbose=0)
#     prediction_time = time.time() - start_time
    
#     predicted_digit = np.argmax(prediction[0])
#     return predicted_digit, prediction_time

def compare_model_accuracies_encoded(knn_model, nn_model, X_test, y_test):
    tf.get_logger().setLevel('ERROR')
    tf.keras.utils.disable_interactive_logging()
    
    knn_pred = knn_model.predict(X_test)
    with tf.keras.utils.CustomObjectScope({'custom_disable_progress': True}):
        nn_pred = nn_model.predict(X_test, verbose=0).argmax(axis=1)
    
    knn_cm = confusion_matrix(y_test.astype(int), knn_pred.astype(int))
    nn_cm = confusion_matrix(y_test.astype(int), nn_pred.astype(int))
    
    knn_accuracies = np.diag(knn_cm.astype('float') / knn_cm.sum(axis=1)[:, np.newaxis] * 100)
    nn_accuracies = np.diag(nn_cm.astype('float') / nn_cm.sum(axis=1)[:, np.newaxis] * 100)
    digits = np.arange(10)
    
    plt.figure(figsize=(12, 6))
    
    bar_width = 0.35
    plt.bar(digits - bar_width/2, knn_accuracies, bar_width, 
            label='KNN', color='#1f77b4', alpha=0.8)
    plt.bar(digits + bar_width/2, nn_accuracies, bar_width,
            label='Neural Network', color='#2ca02c', alpha=0.8)
    

    plt.xlabel('Digit', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Accuracy Comparison by Digit', fontsize=14, pad=20)
    plt.xticks(digits)
    plt.ylim(85, 100) 
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    for i in digits:
        plt.text(i - bar_width/2, knn_accuracies[i] + 0.5, f'{knn_accuracies[i]:.1f}%',
                ha='center', va='bottom', fontsize=9)
        plt.text(i + bar_width/2, nn_accuracies[i] + 0.5, f'{nn_accuracies[i]:.1f}%',
                ha='center', va='bottom', fontsize=9)    
    plt.tight_layout()
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    plt.close()
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    
    return f'<img src="data:image/png;base64,{img_base64}" alt="Model Accuracy Comparison" />'

def analyze_model_accuracies(knn_model, nn_model, X_test, y_test):
    tf.get_logger().setLevel('ERROR')
    tf.keras.utils.disable_interactive_logging()
    knn_pred = knn_model.predict(X_test)
    with tf.keras.utils.CustomObjectScope({'custom_disable_progress': True}):
        nn_pred = nn_model.predict(X_test, verbose=0).argmax(axis=1)
    knn_cm = confusion_matrix(y_test.astype(int), knn_pred.astype(int))
    nn_cm = confusion_matrix(y_test.astype(int), nn_pred.astype(int))
    knn_cm_percent = knn_cm.astype('float') / knn_cm.sum(axis=1)[:, np.newaxis] * 100
    nn_cm_percent = nn_cm.astype('float') / nn_cm.sum(axis=1)[:, np.newaxis] * 100
    analysis_text = "\nDetailed Per-Digit Analysis:\n"
    analysis_text += "-" * 50 + "\n"
    
    for i in range(10):
        knn_accuracy = knn_cm_percent[i,i]
        nn_accuracy = nn_cm_percent[i,i]
        analysis_text += f"\nDigit {i}:\n"
        analysis_text += f"KNN Accuracy: {knn_accuracy:.1f}%\n"
        analysis_text += f"Neural Network Accuracy: {nn_accuracy:.1f}%\n"
        analysis_text += f"Difference: {(nn_accuracy - knn_accuracy):.1f}%\n"
    
    return analysis_text, knn_cm_percent, nn_cm_percent



# def plot_sample_digits(X, y, num_samples=60, num_rows=6):
#     """Plot a grid of sample digits from the dataset."""
#     digits_per_row = math.ceil(num_samples / num_rows)
#     plt.figure(figsize=(20, num_rows * 2))
#     for i in range(num_samples):
#         plt.subplot(num_rows, digits_per_row, i + 1)
#         plt.imshow(X[i].reshape(28, 28), cmap='gray')
#         plt.title(f'Digit: {y[i]}')
#         plt.axis('off')    
#     plt.tight_layout()
#     plt.show()

# def plot_digit_distribution(y):
#     plt.figure(figsize=(10, 5))
#     plt.hist(y, bins=10, rwidth=0.8)
#     plt.title('Distribution of Digits in Dataset')
#     plt.xlabel('Digit')
#     plt.ylabel('Count')
#     plt.xticks(range(10))
#     plt.grid(True, alpha=0.3)
#     plt.show()

def encode_plt_image():
    """Helper function to encode matplotlib plot to base64."""
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    plt.close()
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{img_base64}" alt="MNIST Visualization" />'


def create_comparison_table(knn_accuracy, knn_train_time, nn_accuracy, nn_train_time):
    data = {
        "Metric": ["Training Time (seconds)", "Accuracy (%)"],
        "KNN": [knn_train_time, knn_accuracy],
        "Neural Network": [nn_train_time, nn_accuracy]
    }
    compare_df = pd.DataFrame(data)

    return compare_df

# if __name__ == "__main__":
#     # Set rebuild_model=True to force retraining of models
#     main(rebuild_model=True)

