from algorithm import AntColonyOptimization
from application import run_aco_segmentation, visualize_results

def main():
    """Main function to run the ACO image segmentation."""
    
    # Define the path to the image (can be a URL or local path)
    image_path = 'https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-023-41576-6/MediaObjects/41598_2023_41576_Fig1_HTML.jpg'
    
    # Set up parameters for the ACO algorithm
    params = {
        'resize': (200, 200), 
        'n_segments': 5,
        'ants': 100,
        'iterations': 1,
        'alpha': 1.0,
        'beta': 2.5,
        'p': 0.1,
        'q': 150.0
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