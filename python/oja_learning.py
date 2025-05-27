import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")  # Use a non-interactive backend for matplotlib

class OjaNeuron:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = np.random.rand(input_size)

    def update_weights(self, input: np.ndarray, learning_rate: float = 0.01):
        y = np.dot(self.weights, input)
        self.weights += learning_rate * (input * y - y * y * self.weights)


def test_neuron():
    np.random.seed(42)  # For reproducibility
    # Generate a random dataset
    data = np.random.rand(1000, 3)  # Generate a dataset with 100 samples and 3 features
    data = StandardScaler().fit_transform(data)  # Standardize the dataset

    pca = PCA(n_components=1)
    pca.fit_transform(data)

    initial_learning_rate = 0.01
    learning_rate = initial_learning_rate
    min_learning_rate = 0.001
    num_epochs = 10000

    interval = 100

    plt.figure(figsize=(10, 6))
    plt.title("Oja's Learning Rule")
    plt.xlabel("Epoch")
    plt.ylabel("Difference")

    differences = []

    neuron = OjaNeuron(input_size=3)
    for epoch in tqdm(range(num_epochs), desc="Training Neuron"):
        for input_vector in data:
            neuron.update_weights(input_vector, learning_rate=learning_rate)
            learning_rate = initial_learning_rate * (min_learning_rate / initial_learning_rate) ** (epoch / num_epochs)
        
        if epoch % interval == 0:
            difference = neuron.weights - pca.components_[0]
            differences.append(np.linalg.norm(difference))


    print("PCA components:", pca.components_[0])
    print("Updated weights:", neuron.weights / np.linalg.norm(neuron.weights))
    print("Difference:", neuron.weights - pca.components_[0])

    for i in tqdm(range(len(differences)), desc="Plotting"):
        plt.scatter(i * interval, differences[i], color="blue", alpha=0.5)

    plt.savefig("oja_learning.png")
    plt.close()



if __name__ == "__main__":
    test_neuron()