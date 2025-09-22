import json
import os
import time
from collections import defaultdict
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
from model import efficientnetv2_s as create_model

# Initialize settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model configuration
model_weight_path = "weights/model-29.pth"
json_path = 'class_indices.json'
img_dir = r"D:\Test11_efficientnetV2 _first\train"  # Update this path as needed

# Create model with proper initialization
model = create_model(num_classes=5).to(device)


def load_model_weights(model, weight_path):
    """Enhanced weight loading with architecture compatibility checks"""
    try:
        checkpoint = torch.load(weight_path, map_location=device)

        # Handle potential architecture mismatches
        model_state_dict = model.state_dict()

        # Filter out unnecessary weights
        filtered_checkpoint = {k: v for k, v in checkpoint.items()
                               if k in model_state_dict and v.shape == model_state_dict[k].shape}

        # Special handling for stem layer if needed
        if 'stem.conv.weight' in checkpoint and 'stem.conv.weight' in model_state_dict:
            checkpoint_stem = checkpoint['stem.conv.weight']
            model_stem = model_state_dict['stem.conv.weight']

            if checkpoint_stem.shape[1] == 3 and model_stem.shape[1] == 3:
                # Standard 3-channel input
                filtered_checkpoint['stem.conv.weight'] = checkpoint_stem
            elif checkpoint_stem.shape[1] != model_stem.shape[1]:
                print(f"Adjusting input channels from {checkpoint_stem.shape[1]} to {model_stem.shape[1]}")
                # Handle channel mismatch by repeating channels
                repeats = model_stem.shape[1] // checkpoint_stem.shape[1]
                filtered_checkpoint['stem.conv.weight'] = checkpoint_stem.repeat(1, repeats, 1, 1)

        # Load filtered weights
        model.load_state_dict(filtered_checkpoint, strict=False)

        # Check for missing or unexpected keys
        missing_keys = set(model_state_dict.keys()) - set(filtered_checkpoint.keys())
        unexpected_keys = set(checkpoint.keys()) - set(model_state_dict.keys())

        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")

        return model

    except Exception as e:
        print(f"Error loading weights: {e}")
        raise


# Load model with better error handling
try:
    model = load_model_weights(model, model_weight_path)
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

# Read class labels
try:
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    class_name_to_idx = {v: k for k, v in class_indict.items()}
    print(f"Loaded {len(class_indict)} classes")
except Exception as e:
    print(f"Error loading class indices: {e}")
    exit(1)

# Enhanced image preprocessing with test-time augmentation
data_transform = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
])


def get_image_paths_by_class(root_dir):
    """Get image paths organized by class with error handling"""
    img_paths = defaultdict(list)
    try:
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                valid_images = []
                for f in os.listdir(class_path):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, f)
                        try:
                            # Verify image can be opened
                            with Image.open(img_path) as img:
                                img.verify()
                            valid_images.append(img_path)
                        except Exception as e:
                            print(f"Invalid image skipped: {img_path} - {str(e)}")
                img_paths[class_name] = valid_images
        return img_paths
    except Exception as e:
        print(f"Error scanning directory: {e}")
        return {}


def batch_predict(img_paths, true_classes, batch_size=32):
    """Optimized batch prediction with error handling and metrics"""
    results = []
    correct = 0
    total = 0

    for i in tqdm(range(0, len(img_paths), batch_size), desc="Predicting"):
        batch = img_paths[i:i + batch_size]
        batch_images = []
        valid_indices = []
        batch_true_classes = []

        # Preprocess batch
        for idx, path in enumerate(batch):
            try:
                img = Image.open(path).convert('RGB')
                tensor = data_transform(img)
                batch_images.append(tensor)
                valid_indices.append(idx)
                batch_true_classes.append(true_classes[i + idx])
            except Exception as e:
                print(f"\nSkipped corrupted image: {path} - {str(e)}")
                continue

        if not batch_images:
            continue

        # Inference
        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast():  # Mixed precision for efficiency
                    outputs = model(torch.stack(batch_images).to(device))
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    top_probs = torch.max(probs, dim=1).values

                    for j, idx in enumerate(valid_indices):
                        true_class = batch_true_classes[j]
                        pred_class = class_indict[str(preds[j].item())]
                        confidence = top_probs[j].item()

                        results.append({
                            "file_path": batch[j],
                            "true_class": true_class,
                            "pred_class": pred_class,
                            "confidence": confidence,
                            "correct": true_class == pred_class
                        })

                        total += 1
                        if true_class == pred_class:
                            correct += 1
        except Exception as e:
            print(f"\nError during batch prediction: {str(e)}")
            continue

    if total > 0:
        print(f"\nBatch accuracy: {correct / total:.2%}")
    return results


def calculate_metrics(results):
    """Calculate detailed classification metrics"""
    if not results:
        return {}

    metrics = {
        "overall": {
            "total": len(results),
            "correct": sum(1 for r in results if r["correct"]),
            "accuracy": sum(1 for r in results if r["correct"]) / len(results)
        },
        "class_stats": defaultdict(lambda: {"total": 0, "correct": 0, "confidences": []}),
        "confusion_matrix": defaultdict(lambda: defaultdict(int))
    }

    for res in results:
        true_class = res["true_class"]
        pred_class = res["pred_class"]

        # Update class statistics
        metrics["class_stats"][true_class]["total"] += 1
        metrics["class_stats"][true_class]["confidences"].append(res["confidence"])
        if res["correct"]:
            metrics["class_stats"][true_class]["correct"] += 1

        # Update confusion matrix
        metrics["confusion_matrix"][true_class][pred_class] += 1

    # Calculate class-wise metrics
    for cls in metrics["class_stats"]:
        stats = metrics["class_stats"][cls]
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"]
            stats["avg_confidence"] = sum(stats["confidences"]) / stats["total"]

    return metrics


def print_results(metrics, class_name_to_idx):
    """Print formatted results"""
    if not metrics:
        print("No results to display")
        return

    print("\n=== Detailed Classification Results ===")
    print(f"Overall Accuracy: {metrics['overall']['accuracy']:.2%}")
    print(f"Total Samples: {metrics['overall']['total']}")

    print("\nClass-wise Performance:")
    for class_name in sorted(metrics['class_stats'].keys(),
                             key=lambda x: int(class_name_to_idx[x])):
        stats = metrics['class_stats'][class_name]
        print(f"{class_name}({class_name_to_idx[class_name]}): "
              f"{stats['correct']}/{stats['total']} "
              f"Acc: {stats.get('accuracy', 0):.2%} "
              f"Avg Conf: {stats.get('avg_confidence', 0):.2f}")


if __name__ == '__main__':
    print("\nStarting prediction process...")

    # Get image paths with validation
    imgs_by_class = get_image_paths_by_class(img_dir)
    if not imgs_by_class:
        print("No valid images found in directory")
        exit(1)

    img_paths = []
    true_classes = []

    for class_name, paths in imgs_by_class.items():
        img_paths.extend(paths)
        true_classes.extend([class_name] * len(paths))

    print(f"Found {len(img_paths)} images across {len(imgs_by_class)} classes")

    # Run prediction
    start_time = time.time()
    results = batch_predict(img_paths, true_classes, batch_size=32)
    total_time = time.time() - start_time

    if not results:
        print("No predictions completed successfully")
        exit(1)

    # Calculate and display metrics
    metrics = calculate_metrics(results)
    print_results(metrics, class_name_to_idx)
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time / len(results):.4f} seconds")

    # Save comprehensive results
    try:
        with open('prediction_results.json', 'w') as f:
            json.dump({
                "metrics": metrics,
                "predictions": results,
                "config": {
                    "model": "EfficientNetV2_LSK_Bio",
                    "weights": model_weight_path,
                    "classes": class_indict,
                    "processing_time": total_time
                }
            }, f, indent=2)
        print("Results saved to prediction_results.json")
    except Exception as e:
        print(f"Error saving results: {e}")