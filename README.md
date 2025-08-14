# Scratch Neural Network for MNIST Digits

This project implements a simple neural network from scratch (no deep learning libraries) to classify handwritten digits from the MNIST dataset.

<blockquote>
<strong>@beastbroak30</strong> This project was built from scratch as a personal learning journey. It took me more than 14 days to figure out and learn the math and code, and it was a real challenge to see how much can be achieved with just mathematics and NumPy. I plan to try porting this neural network to embedded systems may be not the best but my own anyways!
</blockquote>

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

## Forward Propagation
Given input $X \in \mathbb{R}^{784 \times m}$ (each column is an image):

1. **Hidden layer pre-activation:**
	$$ Z^{[1]} = W^{[1]} X + b^{[1]} $$
	- $W^{[1]} \in \mathbb{R}^{10 \times 784}$
	- $b^{[1]} \in \mathbb{R}^{10 \times 1}$
2. **Hidden layer activation:**
	$$ A^{[1]} = \text{ReLU}(Z^{[1]}) $$
3. **Output layer pre-activation:**
	$$ Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]} $$
	- $W^{[2]} \in \mathbb{R}^{10 \times 10}$
	- $b^{[2]} \in \mathbb{R}^{10 \times 1}$
4. **Output layer activation (softmax):**
	$$ A^{[2]} = \text{softmax}(Z^{[2]}) $$

---

## Backward Propagation
Let $Y \in \mathbb{R}^{10 \times m}$ be the one-hot encoded labels.

1. **Output error:**
	$$ dZ^{[2]} = A^{[2]} - Y $$
2. **Gradients for output layer:**
	$$ dW^{[2]} = \frac{1}{m} dZ^{[2]} (A^{[1]})^T $$
	$$ db^{[2]} = \frac{1}{m} \sum dZ^{[2]} $$
3. **Hidden layer error:**
	$$ dZ^{[1]} = (W^{[2]})^T dZ^{[2]} \odot \text{ReLU}'(Z^{[1]}) $$
4. **Gradients for hidden layer:**
	$$ dW^{[1]} = \frac{1}{m} dZ^{[1]} X^T $$
	$$ db^{[1]} = \frac{1}{m} \sum dZ^{[1]} $$

---

## Parameter Updates
For learning rate $\alpha$:
- $W^{[l]} := W^{[l]} - \alpha dW^{[l]}$
- $b^{[l]} := b^{[l]} - \alpha db^{[l]}$

---

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
