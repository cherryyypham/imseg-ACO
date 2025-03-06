import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from algorithm import AntColonyOptimization

def load_and_preprocess_image(image_path, resize=None):
    """Load and preprocess image for segmentation"""
    # Load image
    img = io.imread(image_path)

    # Resize if needed
    if resize is not None:
        from skimage.transform import resize as sk_resize
        img = sk_resize(img, resize)

    # Convert to grayscale
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = color.rgb2gray(img)

    return img

def run_aco_segmentation(algorithm, image_path, resize=None, ants=20, iterations=30,
                        alpha=1.0, beta=2.0, p=0.3, q=100.0):
    # Load and preprocess image
    img = load_and_preprocess_image(image_path, resize)

    # Run ACO segmentation
    aco = AntColonyOptimization(
        image=img,
        ants=ants,
        iterations=iterations,
        alpha=alpha,
        beta=beta,
        p=p,
        q=q
    )

    segmentation = aco.run()

    # Visualize final results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(segmentation, cmap='gray')
    plt.title("ACO Segmentation")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    # Create colored overlay
    overlay = np.zeros((*img.shape, 3))
    overlay[..., 0] = img  # Red channel
    overlay[..., 1] = img  # Green channel
    overlay[..., 2] = img  # Blue channel

    # Highlight segmentation in red
    overlay[segmentation, 0] = 1.0
    overlay[segmentation, 1] = 0.0
    overlay[segmentation, 2] = 0.0

    plt.imshow(overlay)
    plt.title("Segmentation Overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return segmentation

# TO DO: Rewrite this function
def visualize_results(img, segmentation):
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # Segmentation result
    plt.subplot(1, 3, 2)
    plt.imshow(segmentation, cmap='gray')
    plt.title("ACO Segmentation")
    plt.axis('off')

    # Colored overlay
    plt.subplot(1, 3, 3)
    # Create colored overlay
    overlay = np.zeros((*img.shape, 3))
    overlay[..., 0] = img  # Red channel
    overlay[..., 1] = img  # Green channel
    overlay[..., 2] = img  # Blue channel

    # Highlight segmentation in red
    overlay[segmentation, 0] = 1.0
    overlay[segmentation, 1] = 0.0
    overlay[segmentation, 2] = 0.0

    plt.imshow(overlay)
    plt.title("Segmentation Overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.show()