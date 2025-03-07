import matplotlib.pyplot as plt
from skimage import io, color
from algorithm import AntColonyOptimization

def load_and_preprocess_image(image_path, resize=None):
    """
    Load and preprocess image for segmentation
    """
    img = io.imread(image_path)

    # Resize if needed
    if resize is not None:
        from skimage.transform import resize as sk_resize
        img = sk_resize(img, resize)

    # Convert to grayscale
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = color.rgb2gray(img)

    return img

def run_aco_segmentation(algorithm, image_path, resize=None, n_segments=5, ants=50, iterations=10,
                         alpha=1.0, beta=2.0, p=0.1, q=100.0):
    """
    Run ACO segmentation on an image
    """
    img = load_and_preprocess_image(image_path, resize)
    aco = AntColonyOptimization(
        image=img,
        n_segments=n_segments,
        ants=ants,
        iterations=iterations,
        alpha=alpha,
        beta=beta,
        p=p,
        q=q
    ) 
    segmentation = aco.run()
    return img, segmentation

def visualize_results(img, segmentation):
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # Segmentation result
    plt.subplot(1, 2, 2)
    plt.imshow(segmentation, cmap='gray')
    plt.title("ACO Segmentation")
    plt.axis('off')

    plt.tight_layout()
    plt.show()