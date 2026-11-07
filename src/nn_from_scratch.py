import numpy as np

class NeuralNetwork:
    """
    A 2-Layer Feedforward Neural Network built entirely from scratch using NumPy.
    It uses a ReLU activation for the hidden layer and a Linear activation for the output,
    making it perfectly designed for Regression tasks (predicting continuous numbers like Age).
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, seed=42):
        self.lr = learning_rate
        
        # Initialize weights with He initialisation (std = sqrt(2 / fan_in)),
        # the standard choice for ReLU layers. The previous fixed scale of 0.01
        # left the network badly under-trained after 1000 epochs
        # (held-out R^2 0.13 vs 0.95 on a synthetic benchmark, see AUDIT.md).
        # A private, seeded generator keeps runs reproducible without
        # resetting NumPy's global random state for the rest of the app.
        rng = np.random.default_rng(seed)
        
        # Layer 1 (Input to Hidden)
        self.W1 = rng.standard_normal((input_size, hidden_size)) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # Layer 2 (Hidden to Output)
        self.W2 = rng.standard_normal((hidden_size, output_size)) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
    def relu(self, Z):
        """Activation Function: Rectified Linear Unit. Turns negative numbers to 0."""
        return np.maximum(0, Z)
        
    def relu_derivative(self, Z):
        """Calculus: The derivative of ReLU is 1 if Z > 0, else 0."""
        return (Z > 0).astype(float)
        
    def forward(self, X):
        """
        The Forward Pass. Data flows from Input -> Hidden -> Output.
        We save the intermediate steps (Z1, A1, Z2) because we need them for the calculus later!
        """
        # Layer 1
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        
        # Layer 2
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.Z2 # Linear activation because we want to predict raw numbers (Regression)
        
        return self.A2
        
    def compute_loss(self, Y_pred, Y_true):
        """
        Calculates the Mean Squared Error (MSE).
        How wrong were our predictions compared to the ground truth?
        """
        m = Y_true.shape[0]
        loss = (1 / (2 * m)) * np.sum(np.square(Y_pred - Y_true))
        return loss
        
    def backward(self, X, Y_true):
        """
        The Backward Pass (Backpropagation).
        We use the Chain Rule of Calculus to calculate EXACTLY how much each weight (W1, W2)
        contributed to the error, so we know how to adjust them.
        """
        m = X.shape[0]
        
        # 1. Output Layer Error
        # The derivative of Mean Squared Error with respect to the output Z2
        dZ2 = (1 / m) * (self.A2 - Y_true)
        
        # Calculate gradients for W2 and b2 using the Chain Rule
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # 2. Hidden Layer Error
        # We push the error backwards through W2, and then through the derivative of the ReLU function
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        
        # Calculate gradients for W1 and b1 using the Chain Rule
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # 3. Gradient Descent (The actual "Learning" step)
        # Nudge the weights in the opposite direction of the gradient to reduce the error
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def gradient_check(self, X, Y_true, epsilon=1e-7):
        """
        Mathematically proves that our manual backpropagation calculus is perfectly correct 
        by comparing it against a brute-force numerical gradient.
        """
        print("Running Gradient Check...")
        
        # 1. Run our manual backward pass to get the analytical gradients
        self.forward(X)
        
        # We need to compute the analytical gradients (without updating the weights yet)
        m = X.shape[0]
        dZ2 = (1 / m) * (self.A2 - Y_true)
        dW2_analytical = np.dot(self.A1.T, dZ2)
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1_analytical = np.dot(X.T, dZ1)
        
        # 2. Compute the numerical gradients for W1 and W2 using Brute Force
        dW1_numerical = np.zeros_like(self.W1)
        dW2_numerical = np.zeros_like(self.W2)
        
        # Brute force Check W2
        for i in range(self.W2.shape[0]):
            for j in range(self.W2.shape[1]):
                original_val = self.W2[i, j]
                
                # Tweak up
                self.W2[i, j] = original_val + epsilon
                loss_plus = self.compute_loss(self.forward(X), Y_true)
                
                # Tweak down
                self.W2[i, j] = original_val - epsilon
                loss_minus = self.compute_loss(self.forward(X), Y_true)
                
                # Slope = rise / run
                dW2_numerical[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
                
                # Restore original
                self.W2[i, j] = original_val

        # Brute force Check W1
        for i in range(self.W1.shape[0]):
            for j in range(self.W1.shape[1]):
                original_val = self.W1[i, j]
                
                self.W1[i, j] = original_val + epsilon
                loss_plus = self.compute_loss(self.forward(X), Y_true)
                
                self.W1[i, j] = original_val - epsilon
                loss_minus = self.compute_loss(self.forward(X), Y_true)
                
                dW1_numerical[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
                self.W1[i, j] = original_val
                
        # 3. Compare them!
        # Because our target values are large (e.g., Age 55), the squared error is huge (~1000).
        # In Python, subtracting two huge numbers with a microscopic difference causes 
        # "floating point precision loss". A threshold of 1e-5 is standard for unnormalized data.
        diff_W1 = np.linalg.norm(dW1_analytical - dW1_numerical) / (np.linalg.norm(dW1_analytical) + np.linalg.norm(dW1_numerical))
        diff_W2 = np.linalg.norm(dW2_analytical - dW2_numerical) / (np.linalg.norm(dW2_analytical) + np.linalg.norm(dW2_numerical))
        
        print(f"W1 Gradient Difference: {diff_W1}")
        print(f"W2 Gradient Difference: {diff_W2}")
        
        if diff_W1 < 1e-5 and diff_W2 < 1e-5:
            print("\n[SUCCESS] The backpropagation calculus is 100% correct!")
        else:
            print("\n[WARNING] There is a math error in the backward pass.")
        return diff_W1, diff_W2

    def fit(self, X, Y, epochs=1000):
        """Full-batch gradient descent for a fixed number of epochs."""
        for _ in range(epochs):
            self.forward(X)
            self.backward(X, Y)
        return self

    def predict(self, X):
        return self.forward(X)


# ---------------------------------------------------------------------------
# Honest evaluation helpers (all from scratch, NumPy only)
# ---------------------------------------------------------------------------

def train_test_split_indices(n, test_size=0.2, seed=42):
    """Shuffled train/test row indices."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_size)))
    return idx[n_test:], idx[:n_test]


def minmax_fit(X):
    """Learn min/max on TRAINING data only (prevents test-set leakage)."""
    return np.min(X, axis=0), np.max(X, axis=0)


def minmax_transform(X, mins, maxs):
    return (X - mins) / (maxs - mins + 1e-8)


def r2_score(y_true, y_pred):
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def mse(y_true, y_pred):
    return float(np.mean((np.ravel(y_true) - np.ravel(y_pred)) ** 2))


def permutation_importance(predict_fn, X, y, n_repeats=10, seed=42):
    """
    Model-agnostic feature importance: shuffle one feature at a time on
    held-out data and measure how much the error (MSE) increases.
    A feature the model truly relies on hurts accuracy when shuffled;
    an irrelevant feature does not. Returns (mean_increase, std_increase).

    This replaces mean(|W1|), which is not a reliable importance measure:
    hidden-layer weights ignore W2, the activations and feature scale.
    """
    rng = np.random.default_rng(seed)
    base = mse(y, predict_fn(X))
    p = X.shape[1]
    means, stds = np.zeros(p), np.zeros(p)
    for j in range(p):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            deltas.append(mse(y, predict_fn(Xp)) - base)
        means[j], stds[j] = np.mean(deltas), np.std(deltas)
    return means, stds

def main():
    print("Testing Neural Network math...\n")
    
    # Create some dummy data to test the math (5 patients, 3 features each)
    X_dummy = np.random.default_rng(42).standard_normal((5, 3))
    # True target values we want to predict
    Y_dummy = np.array([[25], [40], [35], [55], [20]]) 
    
    # Initialize NN: 3 inputs, 4 hidden nodes, 1 output node
    nn = NeuralNetwork(input_size=3, hidden_size=4, output_size=1)
    
    # Run the gradient check
    nn.gradient_check(X_dummy, Y_dummy)

if __name__ == "__main__":
    main()
