from algorithm import AntColonyOptimization
from application import run_aco_segmentation, visualize_results

def main():
    """Main function to run the ACO image segmentation."""
    
    # Define the path to the image (can be a URL or local path)
    image_path = 'https://helloartsy.com/wp-content/uploads/kids/places/how-to-draw-a-house/how-to-draw-a-house-step-6.jpg'
    
    # Set up parameters for the ACO algorithm
    params = {
        'resize': (200, 200),  # Image size
        'ants': 20,            # Number of ants
        'iterations': 30,      # Number of iterations
        'alpha': 1.0,          # Pheromone importance
        'beta': 2.0,           # Heuristic importance
        'p': 0.3,              # Evaporation rate
        'q': 100.0             # Pheromone deposit factor
    }
    
    print("Starting ACO image segmentation...")
    print(f"Image: {image_path}")
    print(f"Parameters: {params}")
    
    # Run the segmentation
    img, segmentation = run_aco_segmentation(
        algorithm=AntColonyOptimization,
        image_path=image_path,
        **params
    )
    
    # Visualize the results
    visualize_results(img, segmentation)
    
    print("Segmentation complete!")

if __name__ == "__main__":
    main()