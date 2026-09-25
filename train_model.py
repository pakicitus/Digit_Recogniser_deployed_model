"""
train_model.py
---------------
Optional helper script — NOT required by the Streamlit app itself, which
only loads an already-trained model. Mirrors the architecture from the
original training notebook:

    Dense(250, relu, L2=1e-5) -> Dense(100, relu, L2=1e-5) -> Dense(10, linear)

The output layer is LINEAR (raw logits) — softmax is applied separately at
inference time (see predict.py), matching how the model was trained with
SparseCategoricalCrossentropy(from_logits=True).

Usage:
    python train_model.py
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist

SEED = 6969
L2_REG = 0.00001


def build_model() -> tf.keras.Model:
    """Fully-connected Deep Neural Network for MNIST digit classification (logits output)."""
    tf.random.set_seed(SEED)
    model = Sequential(
        [
            Dense(units=250, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG)),
            Dense(units=100, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG)),
            Dense(units=10, activation="linear"),  # raw logits -- softmax applied at inference time
        ],
        name="RegularisedComplex",
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=["accuracy"],
    )
    return model


def load_and_prepare_data():
    """
    Load MNIST and flatten/normalize to match the app's preprocessing.py:
    28x28 -> 784, pixel values scaled to [0, 1]. Labels stay as plain
    integers (sparse), matching SparseCategoricalCrossentropy.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32").reshape(-1, 784) / 255.0
    x_test = x_test.astype("float32").reshape(-1, 784) / 255.0

    return (x_train, y_train), (x_test, y_test)


def main():
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_and_prepare_data()

    print("Building model...")
    model = build_model()
    model.build(input_shape=(None, 784))
    model.summary()

    print("Training...")
    model.fit(
        x_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        verbose=2,
    )

    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc*100:.2f}%")

    model.save("saved_model.keras")
    print("Model saved to saved_model.keras")


if __name__ == "__main__":
    main()
