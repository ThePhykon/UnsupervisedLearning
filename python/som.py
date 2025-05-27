import numpy as np
import tqdm
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os

from typing import Annotated, Callable
from functools import partial
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

matplotlib.use("Agg")  # Use a non-interactive backend for matplotlib

class SOM:
    num_neurons: int # Number of neurons in the grid (derived from grid_shape)
    input_dim: int # Dimension of the input data

    name: str # Name of the SOM instance

    weights: Annotated[np.ndarray, "grid[0], grid[1], input_dim"] # Weights of the neurons in the grid
    grid_shape: tuple[int, int] # Shape of the grid (rows, columns)

    learning_rate: float # Learning rate for weight updates
    radius: float # Radius for neighborhood function

    def __init__(self, input_dim: int, grid_shape: tuple[int, int], name: str = "SOM"):
        # Initialize the SOM with the given parameters
        self.input_dim = input_dim

        self.name = name

        if len(grid_shape) != 2:
            raise ValueError("Grid must have a 2D shape!")
        
        self.grid_shape = grid_shape
        self.num_neurons = grid_shape[0] * grid_shape[1]

        # Initialize weights randomly
        np.random.seed(42)  # For reproducibility
        self.weights = np.random.rand(grid_shape[0], grid_shape[1], input_dim)

        if not os.path.exists("som_pictures"):
            os.mkdir("som_pictures")

    def _find_bmu(self, x: np.ndarray) -> tuple[int, int]:
        # Calculate distance for each weight with the input x
        distances = np.linalg.norm(x - self.weights, axis=2)

        # Retrieve index of weight with smallest distance -> coordinates of the BMU
        return np.unravel_index(np.argmin(distances), distances.shape)
    
    def neighbor_func(self, bmu: tuple[int, int], i: int, j: int) -> float:
        # Calculate the distance from the BMU to the current neuron
        distance = np.linalg.norm(np.array(bmu) - np.array((i, j)))

        # Gaussian neighborhood function
        return np.exp(-distance / (2 * self.radius ** 2))
    
    def _update_weights(self, x: np.ndarray, bmu: tuple[int, int]):
        # Iterate over the grid and update weights based on the BMU
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                # Calculate the influence of the neighbor
                influence = self.neighbor_func(bmu, i, j)
                # Update the weights
                self.weights[i,j] += influence * self.learning_rate * (x - self.weights[i,j])
    
    def train(self, data: np.ndarray, num_epochs: int, learning_rate: float, radius: float, normalize: bool = True, callback: Callable | None = None, callback_args: tuple = ()):   
        # Normalize the data if required
        if normalize:
            scaler = MinMaxScaler()
            data = scaler.fit_transform(data)
        
        self.radius = radius
        self.learning_rate = learning_rate
        initial_learning_rate = learning_rate
        initial_radius = radius
        min_learning_rate = 0.001
        min_radius = 1

        if callback is not None:
            callback(self, 0, data, *callback_args)  # Initial callback

        # ===== Training Loop =====
        for epoch in tqdm.tqdm(range(num_epochs), desc="Training SOM"):
            # Update weights for each input sample
            for x in data:
                bmu = self._find_bmu(x)
                self._update_weights(x, bmu)

            # Adjust learning rate and radius (exponential decay)
            self.learning_rate = initial_learning_rate * (min_learning_rate / initial_learning_rate) ** (epoch / num_epochs)
            self.radius = initial_radius * (min_radius / initial_radius) ** (epoch / num_epochs)

            # Call the callback function if provided
            if callback is not None:
                callback(self, (epoch + 1), data, *callback_args)


    def predict(self, x: np.ndarray) -> tuple[int, int]:
        bmu = self._find_bmu(x)
        return bmu
    
    def plot_umatrix(self, data: np.ndarray | None = None, path: str = "som_pictures/umatrix.png", label: str = "U-Matrix", samples: tuple[np.ndarray, str] | None = None):
        umatrix = np.zeros((self.grid_shape[0], self.grid_shape[1]))

        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                # Calculate the average distance to the neighbors
                neighbors = []
                if i > 0:
                    neighbors.append(self.weights[i-1, j])
                if i < self.grid_shape[0] - 1:
                    neighbors.append(self.weights[i+1, j])
                if j > 0:
                    neighbors.append(self.weights[i, j-1])
                if j < self.grid_shape[1] - 1:
                    neighbors.append(self.weights[i, j+1])
                
                umatrix[i, j] = np.mean(np.linalg.norm(self.weights[i,j] - np.array(neighbors), axis=1))
        
        plt.figure(figsize=(8, 8))
        plt.imshow(umatrix, cmap='inferno')
        plt.colorbar()
        plt.title(label)
        plt.xlabel("Neuron X")
        plt.ylabel("Neuron Y")

        if samples is not None:
            for point, name in samples:
                bmu = self._find_bmu(point)
                plt.scatter(bmu[1], bmu[0], c='lime', marker='o', alpha=0.5, label=name)

        plt.savefig(path)
        plt.close()

    def plot_input_space(self, data = None, path: str = "som_pictures/input_space.png", label: str = "SOM Input Space"):
        if self.input_dim != 2:
            raise ValueError("Input dimension must be 2 for this plot.")
        
        plt.figure(figsize=(8, 8))

        if data is not None:
            for i in range(data.shape[0]):
                plt.scatter(data[i, 0], data[i, 1], c='red', marker='x', alpha=0.5)

        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                plt.scatter(self.weights[i, j, 0], self.weights[i, j, 1], c='blue', marker='o')

                if i > 0:
                    plt.plot([self.weights[i,j,0], self.weights[i-1,j,0]], 
                             [self.weights[i,j,1], self.weights[i-1,j,1]], 
                             'k-', alpha=0.3)
                if j > 0:
                    plt.plot([self.weights[i,j,0], self.weights[i,j-1,0]], 
                             [self.weights[i,j,1], self.weights[i,j-1,1]], 
                             'k-', alpha=0.3)
                
        plt.title(label)
        plt.xlabel("Input Dimension 1")
        plt.ylabel("Input Dimension 2")

        plt.savefig(path)
        plt.close()

    def save_weights(self, path: str | None = None):
        if path is None:
            if not os.path.exists("som/weights"):
                os.mkdir("som/weights")

            path = f"weights/{self.name}.npy"
        # Save the weights to a file
        np.save(path, self.weights)

    def load_weights(self, path: str | None = None):
        if path is None:
            path = f"weights/{self.name}.npy"
        
        # Load the weights from a file
        if os.path.exists(path):
            self.weights = np.load(path)
        else:
            raise FileNotFoundError(f"Weights file {path} not found.")

# Simple random generated dataset for testing
# This function creates a dataset with 3 clusters in 2D space
def test_dataset():
    # Create a dataset with 3 clusters in 2D
    data, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
    return data

def plot_callback(som, epoch, data, plot_umatrix: bool = True, plot_input_space: bool = True, interval: int = 10):
    # Plot U-Matrix and Input Space every 10 epochs
    if epoch % interval == 0:
        if not hasattr(plot_callback, "num"):
            plot_callback.num = 0
    
        else:
            plot_callback.num += 1

        num = plot_callback.num

        if plot_umatrix:
            som.plot_umatrix(data=data, path=f"som_pictures/{som.name}_umatrix/umatrix_{num}.png", label=f"SOM U-Matrix at Epoch {epoch}, LR={som.learning_rate:.4f}, Radius={som.radius:.4f}")
        if plot_input_space:
            som.plot_input_space(data=data, path=f"som_pictures/{som.name}_input_space/input_space_{num}.png", label=f"SOM Input Space at Epoch {epoch}")

def run_test():
    # Create a dataset
    data = test_dataset()

    # Initialize SOM
    som = SOM(input_dim=2, grid_shape=(20, 20), name="Test_Example_SOM")

    # Create directories for storing plots
    if os.path.exists(f"som_pictures/{som.name}_umatrix/"):
        for file in os.listdir(f"som_pictures/{som.name}_umatrix/"):
            os.remove(f"som_pictures/{som.name}_umatrix/{file}")

    else:
        os.mkdir(f"som_pictures/{som.name}_umatrix/")

    if os.path.exists(f"som_pictures/{som.name}_input_space/"):
        for file in os.listdir(f"som_pictures/{som.name}_input_space/"):
            os.remove(f"som_pictures/{som.name}_input_space/{file}")
    else:
        os.mkdir(f"som_pictures/{som.name}_input_space/")

    # Train SOM
    som.train(data, num_epochs=1200, learning_rate=0.01, radius=4, callback=plot_callback, callback_args=(True, True, 10))

def run_gapminder():
    url = "https://raw.githubusercontent.com/resbaz/r-novice-gapminder/master/data/gapminder-FiveYearData.csv"
    data = pd.read_csv(url)
    arr = data.groupby("country")[["year", "pop", "lifeExp", "gdpPercap"]].mean().to_numpy()

    som = SOM(input_dim=4, grid_shape=(20, 20), name="Gapminder_SOM")

    #Create directories for storing plots
    if os.path.exists(f"som_pictures/{som.name}_umatrix/"):
        for file in os.listdir(f"som_pictures/{som.name}_umatrix/"):
            os.remove(f"som_pictures/{som.name}_umatrix/{file}")

    else:
        os.mkdir(f"som_pictures/{som.name}_umatrix/")

    #Define a callback function to plot U-Matrix and Input Space
    som.train(arr, num_epochs=30000, learning_rate=0.2, radius=4.0, callback=plot_callback, callback_args=(True, False, 500))
    som.save_weights()

    # som.load_weights(path="som/weights/weights.npy")

    samples = [(item, "Gapminder") for item in arr]

    som.plot_umatrix(data=arr, path=f"som_pictures/{som.name}_umatrix/umatrix.png", label="Gapminder U-Matrix", samples=samples)

if __name__ == "__main__":
    run_test()
    # run_gapminder()