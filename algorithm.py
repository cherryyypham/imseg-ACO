import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from scipy import ndimage
import random
from IPython.display import clear_output
from sklearn.cluster import KMeans

class AntColonyOptimization:
    """
    Ant Colony Optimization algorithm for image segmentation.
    
    This class implements an ACO approach to segment grayscale images into multiple regions
    by simulating artificial ants that deposit pheromones based on image characteristics.
    
    Attributes:
        image (ndarray): Input grayscale image to be segmented
        height (int): Height of the input image
        width (int): Width of the input image
        n_segments (int): Number of segments to identify
        ants (int): Number of ants in the colony
        iterations (int): Maximum number of iterations for the algorithm
        alpha (float): Pheromone importance factor
        beta (float): Heuristic information importance factor
        p (float): Pheromone evaporation rate
        q (float): Pheromone deposit factor
        tau_0 (float): Initial pheromone value
        tau (ndarray): Pheromone matrix for each segment
        edge_info (ndarray): Edge information extracted from the image
        n (ndarray): Heuristic information based on edge information
        probability_maps (ndarray): Probability maps for each segment
        segment_labels (ndarray): Initial segment labels from k-means
        best_segmentation (ndarray): Best segmentation found so far
        best_fitness (float): Fitness value of the best segmentation
    """
    def __init__(self, image, n_segments=5, ants=100, iterations=50, alpha=1.0, beta=2.0, p=0.5,
                q=100.0, tau_0=0.1):
        """
        Initialize the Ant Colony Optimization algorithm for image segmentation.
        """
        self.image = image               # ndarray: input grayscale image to be segmented
        self.height, self.width = image.shape
        self.n_segments = n_segments     # int: number of segments to identify
        self.ants = ants                 # int: num ants in colony
        self.iterations = iterations     # int: num iterations for algo
        self.alpha = alpha               # float: pheromone importance factor
        self.beta = beta                 # float: heuristic importance factor
        self.p = p                       # float: pheromone evaporation rate
        self.q = q                       # float: pheromone deposit factor
        self.tau_0 = tau_0               # float: initial pheromone value

        # Initialize pheromone matrix - one for each potential segment
        self.tau = np.ones((self.n_segments, self.height, self.width)) * tau_0

        # Edge information
        self.edge_info = filters.sobel(image)
        self.edge_info = self.edge_info / np.max(self.edge_info)

        # Heuristic information based on edge information
        self.n = 1.0 / (self.edge_info + 0.1)

        # Initialize probability maps for each segment
        self.probability_maps = np.zeros((self.n_segments, self.height, self.width))
        
        # Create initial segmentation seeds using k-means
        self.initialize_kmeans_seeds()
        
        # Best solutions tracking
        self.best_segmentation = None
        self.best_fitness = float('-inf')

    def run(self, visualize_progress=True):
        """
        Run the Ant Colony Optimization algorithm for image segmentation.
        
        Args:
            visualize_progress (bool, optional): Whether to visualize the progress
                during optimization. Defaults to True.
                
        Returns:
            ndarray: The best segmentation found by the algorithm.
        """
        for iteration in range(self.iterations):
            print(f"Running Iteration {iteration}")
            # Arrays to store ant paths for each segment
            all_segment_paths = [[] for _ in range(self.n_segments)]
            
            # For each segment, run ants
            for segment in range(self.n_segments):
                # Initialize ant positions based on segment probability
                ant_positions = self._initialize_ant_positions(segment)
                ant_paths = [[] for _ in range(self.ants)]
                max_steps = (self.height * self.width) // 20
                
                # Each ant explores based on pheromone for this segment
                for ant_i in range(self.ants):
                    current_pos = ant_positions[ant_i]
                    ant_paths[ant_i].append(current_pos)
                    
                    for _ in range(max_steps):
                        next_positions = self._get_neighbors(current_pos)
                        
                        if not next_positions:
                            break
                        
                        next_pos = self._choose_next_position(current_pos, next_positions, segment)
                        current_pos = next_pos
                        ant_paths[ant_i].append(current_pos)
                
                all_segment_paths[segment] = ant_paths
            
            # Create segmentations from ant paths
            segmentations = self._create_segmentations(all_segment_paths)
            
            # Evaluate fitness of the overall segmentation
            fitness = self._evaluate_fitness(segmentations)
            
            # Update best segmentation if better
            if fitness > self.best_fitness:
                self.best_segmentation = segmentations
                self.best_fitness = fitness
            
            # Update pheromone trails for each segment
            self._update_pheromone(all_segment_paths, segmentations, fitness)
            
            # Update probability maps for each segment
            self._update_probability_maps()
            
            # Visualize progress
            if visualize_progress and (iteration % 5 == 0 or iteration == self.iterations - 1):
                clear_output(wait=True)
                self._visualize_progress(iteration, segmentations)
        
        return self.best_segmentation

    def initialize_kmeans_seeds(self):
        """
        Initialize segment seeds using k-means clustering
        """
        # Reshape image for k-means
        pixels = self.image.reshape(-1, 1)
        
        # Apply k-means clustering
        kmeans = KMeans(n_clusters=self.n_segments, random_state=42)
        labels = kmeans.fit_predict(pixels)
        
        # Reshape labels back to image dimensions
        self.segment_labels = labels.reshape(self.height, self.width)
        
        # Create initial probability maps for each segment
        for segment in range(self.n_segments):
            # Initialize higher pheromone for pixels in this segment
            self.tau[segment][self.segment_labels == segment] = self.tau_0 * 3
            
            # Initialize probability map for this segment
            self.probability_maps[segment][self.segment_labels == segment] = 1.0


    def _initialize_ant_positions(self, segment):
        """
        Initialize ant positions with bias towards the provided segment
        """
        positions = []
        for _ in range(self.ants):
            # Place ant on a pixel likely to belong to this segment
            if random.random() < 0.7:
                # Get positions where probability > 0.5
                high_prob_positions = np.where(self.probability_maps[segment] > 0.5)
                if len(high_prob_positions[0]) > 0:
                    idx = random.randint(0, len(high_prob_positions[0]) - 1)
                    positions.append((high_prob_positions[0][idx], high_prob_positions[1][idx]))
                else:
                    # Random position if no high probability positions
                    positions.append((random.randint(0, self.height-1), random.randint(0, self.width-1)))
            else:
                # Random position
                positions.append((random.randint(0, self.height-1), random.randint(0, self.width-1)))
        return positions

    def _get_neighbors(self, pos):
        """
        Get the 8-connected neighbors of a given position.
        """
        i, j = pos
        neighbors = []

        # 8-neighborhood: https://www.imageprocessingplace.com/downloads_V3/root_downloads/tutorials/contour_tracing_Abeer_George_Ghuneim/connect.html
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue

                ni, nj = i + di, j + dj

                if 0 <= ni < self.height and 0 <= nj < self.width:
                    neighbors.append((ni, nj))

        return neighbors

    def _choose_next_position(self, current_pos, next_positions, segment):
        """
        Choose next position based on pheromone and heuristic for the specific segment
        """
        probabilities = []

        for pos in next_positions:
            i, j = pos
            # Calculate ACO probability for this segment
            pheromone_val = self.tau[segment, i, j] ** self.alpha
            heuristic_val = self.n[i, j] ** self.beta
            probability = pheromone_val * heuristic_val
            probabilities.append(probability)

        # Normalize probabilities
        total = sum(probabilities)
        if total == 0:
            return random.choice(next_positions)

        probabilities = [p / total for p in probabilities]

        # Choose next position based on randomized probabilities
        r = random.random()
        cumsum = 0
        for i, probability in enumerate(probabilities):
            cumsum += probability
            if r <= cumsum:
                return next_positions[i]

        return next_positions[-1]

    def _create_segmentations(self, all_segment_paths):
        """
        Create segmentation mask for each segment based on ant paths
        """
        # Initialize segmentation masks
        segmentations = np.zeros((self.n_segments, self.height, self.width), dtype=bool)
        
        # For each segment
        for segment, paths in enumerate(all_segment_paths):
            segment_mask = np.zeros((self.height, self.width), dtype=bool)
            
            # Mark all pixels visited by ants for this segment
            for path in paths:
                for i, j in path:
                    segment_mask[i, j] = True
            
            # Apply morphological operations to clean up
            segment_mask = ndimage.binary_dilation(segment_mask, iterations=2)
            segment_mask = ndimage.binary_erosion(segment_mask, iterations=1)
            segment_mask = ndimage.binary_fill_holes(segment_mask)
            
            # Store in segmentations array
            segmentations[segment] = segment_mask
        
        # Resolve overlaps
        final_segmentation = np.zeros((self.height, self.width), dtype=int)
        
        for i in range(self.height):
            for j in range(self.width):
                # Get segment with maximum probability for this pixel
                segment_probs = [self.probability_maps[s, i, j] for s in range(self.n_segments)]
                final_segmentation[i, j] = np.argmax(segment_probs)
        
        return final_segmentation

    def _evaluate_fitness(self, segmentation):
        """
        Evaluate the fitness of a multi-region segmentation
        """
        # Calculate inter-region edge strength
        edge_fitness = 0
        region_homogeneity = 0
        
        # For each segment
        for segment in range(self.n_segments):
            # Get the mask for this segment
            mask = (segmentation == segment)
            
            # Skip if empty segment
            if np.sum(mask) == 0:
                continue
            
            # Calculate boundary strength
            boundary = mask ^ ndimage.binary_erosion(mask)
            if np.sum(boundary) > 0:
                edge_fitness += np.mean(self.edge_info[boundary])
            
            # Calculate region homogeneity
            region_pixels = self.image[mask]
            if len(region_pixels) > 1:
                region_std = np.std(region_pixels)
                region_homogeneity += 1.0 / (region_std + 0.1)
        
        # Penalize if any segment is empty
        n_empty_segments = sum(1 for segment in range(self.n_segments) 
                              if np.sum(segmentation == segment) == 0)
        empty_penalty = 0.5 * n_empty_segments
        
        # Calculate overall fitness
        fitness = edge_fitness + 0.8 * region_homogeneity - empty_penalty
        
        return fitness

    def _update_pheromone(self, all_segment_paths, segmentation, fitness):
        """
        Update pheromone levels for each segment
        """
        # Evaporation for all segments
        self.tau *= (1 - self.p)
        
        # For each segment
        for segment in range(self.n_segments):
            # Get the mask for this segment
            mask = (segmentation == segment)
            
            # Skip if empty segment
            if np.sum(mask) == 0:
                continue
            
            # Calculate deposit amount
            q = self.q * fitness / self.n_segments
            
            # Update pheromone for this segment's paths
            for path in all_segment_paths[segment]:
                for i, j in path:
                    self.tau[segment, i, j] += q
            
            # Boost pheromone in the segment region
            self.tau[segment][mask] += q * 1.5

    def _update_probability_maps(self):
        """
        Update probability maps for each segment based on pheromone levels
        """
        # Calculate total pheromone at each pixel across all segments
        total_pheromone = np.sum(self.tau, axis=0)
        
        # Calculate probability for each segment
        for segment in range(self.n_segments):
            self.probability_maps[segment] = self.tau[segment] / (total_pheromone + 1e-10)

    def _visualize_progress(self, iteration, segmentation):
        """
        Visualize the current state of segmentation
        """
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(self.image, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')
        
        # Current segmentation
        plt.subplot(1, 2, 2)
        plt.imshow(segmentation, cmap='nipy_spectral')
        plt.title(f"Segmentation (Iteration {iteration+1})")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()