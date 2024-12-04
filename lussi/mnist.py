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

# Constants for model file paths
KNN_MODEL_PATH = 'knn_mnist_model.joblib'
NN_MODEL_PATH = 'nn_mnist_model.keras'  # Changed this line to add .keras extension
SCALER_PATH = 'scaler.joblib'

def load_data():
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X.astype('float32')
    y = y.astype('int32')  # Add this line to convert labels to integers
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the data - save scaler for future use
    if os.path.exists(SCALER_PATH):
        print("Loading existing scaler...")
        scaler = joblib.load(SCALER_PATH)
    else:
        print("Creating new scaler...")
        scaler = StandardScaler()
        scaler.fit(X_train)
        joblib.dump(scaler, SCALER_PATH)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def train_knn(X_train, X_test, y_train, y_test, rebuild_model=False):
    if not rebuild_model and os.path.exists(KNN_MODEL_PATH):
        print("Loading existing KNN model...")
        knn = joblib.load(KNN_MODEL_PATH)
    else:
        print("Training new KNN model...")
        start_time = time.time()
        
        # Create and train KNN
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        
        # Save the model
        joblib.dump(knn, KNN_MODEL_PATH)
        
        train_time = time.time() - start_time
        print(f"Training time: {train_time:.2f} seconds")
    
    # Evaluate
    start_time = time.time()
    accuracy = knn.score(X_test, y_test)
    inference_time = time.time() - start_time
    
    print(f"KNN Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Inference time: {inference_time:.2f} seconds")
    
    return knn, accuracy

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

def train_neural_network(X_train, X_test, y_train, y_test, rebuild_model=False):
    if not rebuild_model and os.path.exists(NN_MODEL_PATH):
        print("Loading existing Neural Network model...")
        model = keras.models.load_model(NN_MODEL_PATH)
        history = None  # No training history for loaded model
    else:
        print("Training new Neural Network...")
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
        print(f"Training time: {train_time:.2f} seconds")
    
    # Evaluate
    start_time = time.time()
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    inference_time = time.time() - start_time
    
    print(f"\nNeural Network Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Inference time: {inference_time:.2f} seconds")
    
    return model, history, accuracy

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
    X_train, X_test, y_train, y_test = load_data()
    
    # Train and evaluate KNN
    knn_model, knn_accuracy = train_knn(X_train, X_test, y_train, y_test, rebuild_model)
    
    # Train and evaluate Neural Network
    nn_model, history, nn_accuracy = train_neural_network(X_train, X_test, y_train, y_test, rebuild_model)
    
    # Plot training history for Neural Network if available
    plot_training_history(history)
    
    # Compare results
    print("\nComparison:")
    print(f"KNN Accuracy: {knn_accuracy:.4f}")
    print(f"Neural Network Accuracy: {nn_accuracy:.4f}")
    plot_confusion_matrices(knn_model, nn_model, X_test, y_test)


if __name__ == "__main__":
    # Set rebuild_model=True to force retraining of models
    main(rebuild_model=False)