import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import time
import matplotlib.pyplot as plt
import os
import joblib
import math
import io
import base64
import math
from IPython.display import HTML
import pandas as pd


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

def plot_digit_distribution_encoded(y):
    """Plot distribution of digits in the dataset, returns base64 encoded image."""
    plt.figure(figsize=(10, 5))
    plt.hist(y, bins=10, rwidth=0.8)
    plt.title('Distribution of Digits in Dataset')
    plt.xlabel('Digit')
    plt.ylabel('Count')
    plt.xticks(range(10))
    plt.grid(True, alpha=0.3)
    
    return encode_plt_image()

def plot_confusion_matrices_encoded(knn_model, nn_model, X_test, y_test):
    """Plot confusion matrices, returns base64 encoded image."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Get predictions
    knn_pred = knn_model.predict(X_test)
    nn_pred = nn_model.predict(X_test).argmax(axis=1)
    
    # Calculate confusion matrices
    knn_cm = confusion_matrix(y_test.astype(int), knn_pred.astype(int))
    nn_cm = confusion_matrix(y_test.astype(int), nn_pred.astype(int))
    
    # Plot KNN confusion matrix
    sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('KNN Confusion Matrix\n(Raw Counts)', pad=20)
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Plot Neural Network confusion matrix
    sns.heatmap(nn_cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('Neural Network Confusion Matrix\n(Raw Counts)', pad=20)
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.tight_layout()
    encoded_raw = encode_plt_image()
    
    # Plot percentages
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Convert to percentages
    knn_cm_percent = knn_cm.astype('float') / knn_cm.sum(axis=1)[:, np.newaxis] * 100
    nn_cm_percent = nn_cm.astype('float') / nn_cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Plot KNN percentage confusion matrix
    sns.heatmap(knn_cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax1)
    ax1.set_title('KNN Confusion Matrix\n(Percentages)', pad=20)
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Plot Neural Network percentage confusion matrix
    sns.heatmap(nn_cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax2)
    ax2.set_title('Neural Network Confusion Matrix\n(Percentages)', pad=20)
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.tight_layout()
    encoded_percent = encode_plt_image()
    
    return encoded_raw, encoded_percent, knn_cm_percent, nn_cm_percent

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
        # print("Loading existing KNN model...")
        knn = joblib.load(KNN_MODEL_PATH)
        train_time = 0  # Model was loaded, not trained
    else:
        # print("Training new KNN model...")
        start_time = time.time()
        
        # Create and train KNN
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        
        # Save the model
        joblib.dump(knn, KNN_MODEL_PATH)
        
        train_time = time.time() - start_time
        
    # print(f"Training time: {train_time:.2f} seconds")
    
    # Test accuracy
    accuracy = knn.score(X_test, y_test)
    # print(f"KNN Results:")
    # print(f"Accuracy: {accuracy:.4f}")
    
    return knn, accuracy

def predict_single_image_knn(model, image):
    """
    Predict a single image and measure how long it takes
    """
    start_time = time.time()
    prediction = model.predict(image.reshape(1, -1))
    prediction_time = time.time() - start_time
    
    return prediction[0], prediction_time

# Neural Network functions
def train_neural_network(X_train, X_test, y_train, y_test, rebuild_model=False):
    if not rebuild_model and os.path.exists(NN_MODEL_PATH):
        # print("Loading existing Neural Network model...")
        model = keras.models.load_model(NN_MODEL_PATH)
        history = None  # No training history for loaded model
        train_time = 0  # Model was loaded, not trained
    else:
        # print("Training new Neural Network...")
        start_time = time.time()
        
        # Create and train Neural Network
        model = create_neural_network()
        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=128,
            validation_split=0.1,
            verbose=1
        )
        
        # Save the model
        model.save(NN_MODEL_PATH)
        
        train_time = time.time() - start_time
    
    # print(f"Training time: {train_time:.2f} seconds")
    
    # Test accuracy
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    # print(f"\nNeural Network Results:")
    # print(f"Accuracy: {accuracy:.4f}")
    
    return model, history, accuracy

def predict_single_image_nn(model, image):
    """
    Predict a single image and measure how long it takes
    """
    start_time = time.time()
    prediction = model.predict(image.reshape(1, -1), verbose=0)
    prediction_time = time.time() - start_time
    
    # Get the digit with highest probability
    predicted_digit = np.argmax(prediction[0])
    
    return predicted_digit, prediction_time


def plot_training_history(history):
    if history is None:
        print("No training history available for loaded model")
        return
        
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_confusion_matrices(knn_model, nn_model, X_test, y_test):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Get predictions
    knn_pred = knn_model.predict(X_test)
    nn_pred = nn_model.predict(X_test).argmax(axis=1)
    
    # Calculate confusion matrices
    knn_cm = confusion_matrix(y_test.astype(int), knn_pred.astype(int))
    nn_cm = confusion_matrix(y_test.astype(int), nn_pred.astype(int))
    
    # Plot KNN confusion matrix
    sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('KNN Confusion Matrix\n(Raw Counts)', pad=20)
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Plot Neural Network confusion matrix
    sns.heatmap(nn_cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('Neural Network Confusion Matrix\n(Raw Counts)', pad=20)
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and plot percentages
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Convert to percentages
    knn_cm_percent = knn_cm.astype('float') / knn_cm.sum(axis=1)[:, np.newaxis] * 100
    nn_cm_percent = nn_cm.astype('float') / nn_cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Plot KNN percentage confusion matrix
    sns.heatmap(knn_cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax1)
    ax1.set_title('KNN Confusion Matrix\n(Percentages)', pad=20)
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Plot Neural Network percentage confusion matrix
    sns.heatmap(nn_cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax2)
    ax2.set_title('Neural Network Confusion Matrix\n(Percentages)', pad=20)
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.tight_layout()
    plt.show()
    
    # Print some additional metrics
    print("\nDetailed Analysis:")
    
    # Calculate per-class accuracy
    for i in range(10):
        print(f"\nDigit {i}:")
        print(f"KNN - Accuracy: {knn_cm_percent[i,i]:.1f}%")
        print(f"Neural Network - Accuracy: {nn_cm_percent[i,i]:.1f}%")
        
        # Calculate most common misclassifications
        knn_errors = [(j, knn_cm[i,j]) for j in range(10) if j != i and knn_cm[i,j] > 0]
        nn_errors = [(j, nn_cm[i,j]) for j in range(10) if j != i and nn_cm[i,j] > 0]
        
        if knn_errors:
            most_common_knn = max(knn_errors, key=lambda x: x[1])
            print(f"KNN - Most often misclassified as {most_common_knn[0]} ({most_common_knn[1]} times)")
            
        if nn_errors:
            most_common_nn = max(nn_errors, key=lambda x: x[1])
            print(f"NN - Most often misclassified as {most_common_nn[0]} ({most_common_nn[1]} times)")


def main(rebuild_model=False):
    # Load and preprocess data
    X_train, X_test, y_train, y_test, X_train_unscaled = load_data()
    
    # Visualize data (optional)
    plot_sample_digits(X_train_unscaled, y_train, num_samples=10)
    visualize_digit_matrix(X_train_unscaled, index=0)
    plot_digit_distribution(y_train)
    
    # Train and evaluate models with timing
    print("\nTraining Models:")
    print("-" * 50)
    
    # Train and time KNN
    start_time = time.time()
    knn_model, knn_accuracy = train_knn(X_train, X_test, y_train, y_test, rebuild_model)
    knn_train_time = time.time() - start_time
    
    # Train and time Neural Network
    start_time = time.time()
    nn_model, history, nn_accuracy = train_neural_network(X_train, X_test, y_train, y_test, rebuild_model)
    nn_train_time = time.time() - start_time
    
    print("\nTraining Time Summary:")
    print(f"KNN Training Time: {knn_train_time:.2f} seconds")
    print(f"Neural Network Training Time: {nn_train_time:.2f} seconds")
    
    # Plot training history for Neural Network if available
    plot_training_history(history)
    
    # Compare accuracy results
    print("\nAccuracy Comparison:")
    print(f"KNN Accuracy: {knn_accuracy:.4f}")
    print(f"Neural Network Accuracy: {nn_accuracy:.4f}")
    
    # Perform detailed timing analysis
    timing_results = perform_timing_analysis(knn_model, nn_model, X_test, y_test)
    
    # Plot confusion matrices
    plot_confusion_matrices(knn_model, nn_model, X_test, y_test)

# def plot_sample_digits(X, y, num_samples=10):
#     """Plot a row of sample digits from the dataset."""
#     plt.figure(figsize=(20, 2))
#     for i in range(num_samples):
#         plt.subplot(1, num_samples, i + 1)
#         plt.imshow(X[i].reshape(28, 28), cmap='gray')
#         plt.title(f'Digit: {y[i]}')
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()

def plot_sample_digits(X, y, num_samples=60, num_rows=6):
    """Plot a grid of sample digits from the dataset."""
    # Calculate digits per row by dividing total samples by number of rows
    digits_per_row = math.ceil(num_samples / num_rows)
    
    # Create figure with appropriate size
    plt.figure(figsize=(20, num_rows * 2))
    
    # Plot each digit
    for i in range(num_samples):
        plt.subplot(num_rows, digits_per_row, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f'Digit: {y[i]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_digit_matrix(X, index=0):
    """Show the matrix representation for a single digit."""
    digit = X[index].reshape(28, 28)
    
    plt.figure(figsize=(16, 16))  # Larger figure size
    
    # Show the matrix values
    plt.imshow(digit, cmap='gray')
    plt.title('28x28 Matrix Values')
    
    # Add all matrix values as text
    for i in range(28):
        for j in range(28):
            plt.text(j, i, f'{digit[i, j]:.0f}', 
                    ha='center', va='center', 
                    color='red' if digit[i, j] > 0 else 'darkgray',
                    fontsize=11)  # Slightly larger font size
    
    # plt.tight_layout()
    plt.show()

def plot_digit_distribution(y):
    """Plot distribution of digits in the dataset."""
    plt.figure(figsize=(10, 5))
    plt.hist(y, bins=10, rwidth=0.8)
    plt.title('Distribution of Digits in Dataset')
    plt.xlabel('Digit')
    plt.ylabel('Count')
    plt.xticks(range(10))
    plt.grid(True, alpha=0.3)
    plt.show()

def encode_plt_image():
    """Helper function to encode matplotlib plot to base64."""
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    plt.close()
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{img_base64}" alt="MNIST Visualization" />'


def perform_timing_analysis(knn_model, nn_model, X_test, y_test):
    """
    Perform comprehensive timing analysis on both models.
    """
    # Batch size variations for testing
    batch_sizes = [1, 10, 100, 1000]
    results = {'knn': {}, 'nn': {}}
    
    print("\nTiming Analysis:")
    print("-" * 50)
    
    # Test different batch sizes
    for batch_size in batch_sizes:
        # Select subset of test data
        X_batch = X_test[:batch_size]
        
        # KNN timing
        start_time = time.time()
        knn_model.predict(X_batch)
        knn_time = time.time() - start_time
        results['knn'][batch_size] = knn_time
        
        # Neural Network timing
        start_time = time.time()
        nn_model.predict(X_batch, verbose=0)
        nn_time = time.time() - start_time
        results['nn'][batch_size] = nn_time
        
        print(f"\nBatch size: {batch_size}")
        print(f"KNN prediction time: {knn_time:.4f} seconds")
        print(f"Neural Network prediction time: {nn_time:.4f} seconds")
        print(f"Time per image - KNN: {(knn_time/batch_size)*1000:.2f}ms")
        print(f"Time per image - NN: {(nn_time/batch_size)*1000:.2f}ms")
    
    # Plot timing comparison
    plt.figure(figsize=(10, 6))
    batch_sizes_str = [str(size) for size in batch_sizes]
    
    x = np.arange(len(batch_sizes))
    width = 0.35
    
    plt.bar(x - width/2, [results['knn'][size] for size in batch_sizes], width, label='KNN')
    plt.bar(x + width/2, [results['nn'][size] for size in batch_sizes], width, label='Neural Network')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Prediction Time (seconds)')
    plt.title('Model Prediction Time Comparison')
    plt.xticks(x, batch_sizes_str)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return results


def create_comparison_table():
    # Data for the table
    data = {
        "Metric": ["Training Time (seconds)", "Accuracy (%)"],
        "KNN": [6.27, 94.02],
        "Neural Network": [10.91, 97.20]
    }

    # Create a DataFrame
    compare_df = pd.DataFrame(data)

    return compare_df


if __name__ == "__main__":
    # Set rebuild_model=True to force retraining of models
    main(rebuild_model=True)