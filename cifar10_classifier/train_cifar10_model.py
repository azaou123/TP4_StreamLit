import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import joblib

# Charger les données CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normaliser les valeurs des pixels
X_train, X_test = X_train / 255.0, X_test / 255.0

# Construire le modèle CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Sauvegarder le modèle
model.save("cifar10_model.h5")
print("Modèle CIFAR-10 entraîné et sauvegardé sous cifar10_model.h5")
