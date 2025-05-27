import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm  # Import tqdm for progress bar

class SOM:
    def __init__(self, grid_size=(10, 10), input_dim=2, learning_rate=0.1, radius=None, epochs=100):
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.radius = max(grid_size) / 2 if radius is None else radius
        self.weights = np.random.rand(grid_size[0], grid_size[1], input_dim)
        self.grid_pos = np.array([[ [i, j] for j in range(grid_size[1])] 
                                 for i in range(grid_size[0])])
    
    def find_bmu(self, x):
        distances = np.linalg.norm(self.weights - x, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx
    
    def update_weights(self, x, bmu_idx, current_epoch):
        lr = self.learning_rate * (1 - current_epoch / self.epochs)
        r = self.radius * (1 - current_epoch / self.epochs)
        
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                grid_dist = np.linalg.norm(self.grid_pos[i,j] - self.grid_pos[bmu_idx])
                if grid_dist <= r:
                    influence = np.exp(-grid_dist**2 / (2 * r**2))
                    self.weights[i,j] += lr * influence * (x - self.weights[i,j])
    
    def train(self, data):
        for epoch in tqdm(range(self.epochs), desc="Training SOM"):  # Add tqdm progress bar
            np.random.shuffle(data)
            for x in data:
                bmu_idx = self.find_bmu(x)
                self.update_weights(x, bmu_idx, epoch)
    
    def plot(self, data, labels):
        plt.figure(figsize=(10, 8))
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                plt.scatter(self.weights[i,j,0], self.weights[i,j,1], 
                            color='red', marker='o', s=50, alpha=0.5)
                if i > 0:
                    plt.plot([self.weights[i,j,0], self.weights[i-1,j,0]], 
                             [self.weights[i,j,1], self.weights[i-1,j,1]], 
                             'k-', alpha=0.3)
                if j > 0:
                    plt.plot([self.weights[i,j,0], self.weights[i,j-1,0]], 
                             [self.weights[i,j,1], self.weights[i,j-1,1]], 
                             'k-', alpha=0.3)
        
        # Plot data points with labels
        for point, label in zip(data, labels):
            plt.scatter(point[0], point[1], label=label, alpha=0.6)
        
        plt.title('Self-Organizing Map with Gapminder Data')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        plt.grid()
        plt.savefig('som_gapminder_result.png')

    def plot_map(self, data, labels, sample_size=100):
        """
        Plots a color-coded map of the SOM grid and maps a few samples of the data to their BMUs.
        """
        plt.figure(figsize=(10, 8))
        color_map = np.zeros((self.grid_size[0], self.grid_size[1]))

        # Assign random colors to each grid cell
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                color_map[i, j] = np.random.rand()

        plt.imshow(color_map, cmap='viridis', alpha=0.6, origin='upper')
        plt.colorbar(label='Grid Cell Color')

        # Plot a few samples of the data to their BMUs
        sample_indices = np.random.choice(len(data), size=min(sample_size, len(data)), replace=False)
        for idx in sample_indices:
            x = data[idx]
            bmu_idx = self.find_bmu(x)
            plt.scatter(bmu_idx[1], bmu_idx[0], label=labels[idx], alpha=0.7)

        plt.title('SOM Grid with Data Samples')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.legend(loc='upper right', fontsize='small', ncol=2)
        plt.grid(False)
        plt.savefig('som_grid_map.png')

# Load Gapminder dataset
gapminder = pd.read_csv('https://raw.githubusercontent.com/resbaz/r-novice-gapminder-files/master/data/gapminder-FiveYearData.csv')

# Select features (e.g., life expectancy and GDP per capita) and normalize
features = gapminder[['lifeExp', 'gdpPercap']].values
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)

# Extract country labels for visualization
labels = gapminder['country'].values

# SOM initialization and training
som = SOM(grid_size=(20, 20), input_dim=2, learning_rate=0.5, epochs=100)
som.train(normalized_features)

# Visualize results
som.plot(normalized_features, labels)
som.plot_map(normalized_features, labels)
