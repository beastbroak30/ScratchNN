# Scratch Neural Network for MNIST Digits

This project implements a simple neural network from scratch (no deep learning libraries) to classify handwritten digits from the MNIST dataset.



## Architecture
- **Input layer:** 784 units (one for each pixel in a 28x28 image)
- **Hidden layer:** 10 units, ReLU activation
- **Output layer:** 10 units, softmax activation (one for each digit 0-9)

---

## How to Use `main.py`

1. **Training and Saving the Model:**
   - Uncomment the following lines in `main.py`:
     ```python
     W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.15, 800, m, n)
     save_model(W1, b1, W2, b2, 'model')
     ```
   - Comment out the line:
     ```python
     W1, b1, W2, b2 = load_model('model')
     ```
   - Run `main.py` to train and save the model weights.

2. **Loading and Evaluating the Model:**
   - Uncomment the line:
     ```python
     W1, b1, W2, b2 = load_model('model')
     ```
   - Comment out the training and saving lines above.
   - To evaluate accuracy, use:
     ```python
     evaluate_accuracy(X_dev, Y_dev, W1, b1, W2, b2)
     ```
   - You can also use the digit testing feature by setting `digit_to_test`.

> **Tip:** Only run training when you want to retrain the model. For testing and evaluation, just load the saved weights.





## Variable Shapes
| Variable | Shape         | Description                |
|----------|--------------|----------------------------|
| $X$      | $784 \times m$ | Input images               |
| $Y$      | $10 \times m$  | One-hot labels             |
| $W^{[1]}$| $10 \times 784$| Hidden layer weights       |
| $b^{[1]}$| $10 \times 1$  | Hidden layer biases        |
| $A^{[1]}$| $10 \times m$  | Hidden layer activations   |
| $W^{[2]}$| $10 \times 10$ | Output layer weights       |
| $b^{[2]}$| $10 \times 1$  | Output layer biases        |
| $A^{[2]}$| $10 \times m$  | Output layer activations   |

---

## Mathematical Functions
- **ReLU:** $\text{ReLU}(z) = \max(0, z)$
- **Softmax:** $\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$

---

## Usage
1. Place your MNIST CSV in `dataset/train.csv`.
2. Run `main.py` to train, save, and test the model.
3. Use the provided functions to save/load weights and test predictions.

---

## License
MIT

---
