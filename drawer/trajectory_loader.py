import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path


def load_trajectory_file(filename: str) -> Dict[str, Any]:
    """
    Load trajectory data from a JSON file saved by the trajectory drawer.
    
    Args:
        filename: Path to the JSON trajectory file
    
    Returns:
        Dict containing trajectory data with metadata
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
        KeyError: If the file doesn't contain expected trajectory format
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Validate the data structure
    if "trajectories" not in data:
        raise KeyError("File does not contain 'trajectories' key")
    
    return data


def trajectory_to_model_input(trajectory_data: Dict[str, Any]) -> torch.Tensor:
    """
    Convert trajectory data to model input format.
    
    Args:
        trajectory_data: Dictionary containing trajectory data from load_trajectory_file
    
    Returns:
        torch.Tensor: Model input of shape (1, 128, 8) ready for the encoder
    
    Example:
        >>> data = load_trajectory_file("my_trajectory.json")
        >>> model_input = trajectory_to_model_input(data)
        >>> print(model_input.shape)  # torch.Size([1, 128, 8])
    """
    trajectories = trajectory_data["trajectories"]
    
    # Extract position sequences for each object
    position_sequences = []
    for obj_id in range(4):  # Support up to 4 objects
        obj_key = f"object_{obj_id}"
        if obj_key in trajectories:
            positions = trajectories[obj_key]["positions"]
            position_sequences.append(positions)
        else:
            # Add empty sequence for missing objects
            position_sequences.append([])
    
    # Filter out empty sequences and convert to numpy arrays
    valid_sequences = []
    for seq in position_sequences:
        if seq:  # Only include non-empty sequences
            valid_sequences.append(seq)
    
    if not valid_sequences:
        raise ValueError("No valid trajectory sequences found")
    
    # Convert to the format expected by positions_to_model_input
    from handle_input import positions_to_model_input
    return positions_to_model_input(valid_sequences)


def load_trajectory_as_centroids(filename: str) -> np.ndarray:
    """
    Load trajectory data as raw centroids for rotation averaging.
    
    This function extracts the raw position sequences from a trajectory file
    and returns them as a NumPy array suitable for rotation averaging operations.
    
    Args:
        filename: Path to the JSON trajectory file
    
    Returns:
        np.ndarray: Centroid data of shape (n_objects, n_frames, 2)
                    Ready for use with centroids2model_input and rotate_clip_centroids
    
    Example:
        >>> centroids = load_trajectory_as_centroids("example_data/trajectory_curve_left_sharp.json")
        >>> print(centroids.shape)  # (n_objects, n_frames, 2)
        >>> 
        >>> # Use for rotation averaging
        >>> q_inputs = [centroids2model_input(centroids)]
        >>> for _ in range(100):
        ...     q_inputs.append(centroids2model_input(rotate_clip_centroids(centroids)))
        >>> query_embeds = model.net(torch.stack(q_inputs).to(device))
        >>> query_embed_avg = torch.mean(query_embeds, dim=0)
    """
    # Load the trajectory data
    trajectory_data = load_trajectory_file(filename)
    trajectories = trajectory_data["trajectories"]
    
    # Extract position sequences for each object
    position_sequences = []
    for obj_id in range(4):  # Support up to 4 objects
        obj_key = f"object_{obj_id}"
        if obj_key in trajectories:
            positions = trajectories[obj_key]["positions"]
            position_sequences.append(positions)
        else:
            position_sequences.append([])
    
    # Filter out empty sequences
    valid_sequences = [seq for seq in position_sequences if seq]
    if not valid_sequences:
        raise ValueError("No valid trajectory sequences found")
    
    # Convert to numpy array format
    centroids = np.array(valid_sequences)
    return centroids


def compare_trajectories(trajectory1_file: str, trajectory2_file: str, model) -> float:
    """
    Compare two trajectory files and return their similarity score.
    
    Args:
        trajectory1_file: Path to first trajectory file
        trajectory2_file: Path to second trajectory file
        model: The EncoderModelWrapper instance
    
    Returns:
        float: Cosine similarity score between 0 and 1 (1 = identical)
    
    Example:
        >>> similarity = compare_trajectories("traj1.json", "traj2.json", model)
        >>> print(f"Similarity: {similarity:.3f}")
    """
    # Load and convert both trajectories
    data1 = load_trajectory_file(trajectory1_file)
    data2 = load_trajectory_file(trajectory2_file)
    
    input1 = trajectory_to_model_input(data1)
    input2 = trajectory_to_model_input(data2)
    
    # Get embeddings
    from handle_input import get_embedding
    embed1 = get_embedding(model, input1)
    embed2 = get_embedding(model, input2)
    
    # Calculate cosine similarity
    similarity = torch.nn.functional.cosine_similarity(embed1, embed2, dim=1)
    return similarity.item()


def batch_compare_trajectories(query_file: str, trajectory_files: List[str], model, use_rotation_averaged_embedding=False) -> List[Tuple[str, float]]:
    """
    Compare a query trajectory against multiple trajectory files.
    
    Args:
        query_file: Path to the query trajectory file
        trajectory_files: List of paths to trajectory files to compare against
        model: The EncoderModelWrapper instance
    
    Returns:
        List of tuples (filename, similarity_score) sorted by similarity (highest first)
    
    Example:
        >>> files = ["traj1.json", "traj2.json", "traj3.json"]
        >>> results = batch_compare_trajectories("query.json", files, model)
        >>> for filename, score in results:
        ...     print(f"{filename}: {score:.3f}")
    """
    # Load query trajectory
    query_data = load_trajectory_file(query_file)
    query_input = trajectory_to_model_input(query_data)
    
    from handle_input import get_embedding
    query_embedding = get_embedding(model, query_input)
    
    # Compare against all trajectories
    results = []
    for traj_file in trajectory_files:
        try:
            traj_data = load_trajectory_file(traj_file)
            traj_input = trajectory_to_model_input(traj_data)
            traj_embedding = get_embedding(model, traj_input)
            
            similarity = torch.nn.functional.cosine_similarity(
                query_embedding, traj_embedding, dim=1
            ).item()
            
            results.append((traj_file, similarity))
            
        except Exception as e:
            print(f"Error processing {traj_file}: {e}")
            results.append((traj_file, 0.0))
    
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def visualize_trajectory(trajectory_file: str, ax=None, title=None, save_path: Optional[str] = None):
    """
    Create a visualization of a single trajectory with start/end indicators.
    
    Args:
        trajectory_file: Path to the trajectory file
        ax: Optional matplotlib axis to plot on (if None, creates new figure)
        title: Optional title for the plot
        save_path: Optional path to save the visualization image
    
    Returns:
        matplotlib axis object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return None
    
    # Load trajectory data
    data = load_trajectory_file(trajectory_file)
    
    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        standalone = True
    else:
        standalone = False
    
    # Plot trajectory
    colors = ['red', 'green', 'blue', 'yellow']
    for obj_id in range(4):
        obj_key = f"object_{obj_id}"
        if obj_key in data["trajectories"]:
            positions = data["trajectories"][obj_key]["positions"]
            if positions:
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                ax.plot(x_coords, y_coords, color=colors[obj_id], linewidth=2, 
                       label=f"Object {obj_id+1}")
                ax.scatter(x_coords, y_coords, color=colors[obj_id], s=20)
                # Start/end indicators
                ax.scatter(x_coords[0], y_coords[0], color=colors[obj_id], s=100, 
                          marker='o', edgecolors='white', linewidth=2)
                ax.scatter(x_coords[-1], y_coords[-1], color=colors[obj_id], s=100, 
                          marker='s', edgecolors='white', linewidth=2)

                # Add start/end labels
                ax.annotate('START', (x_coords[0], y_coords[0]), 
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, fontweight='bold', color='black')
                ax.annotate('END', (x_coords[-1], y_coords[-1]), 
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, fontweight='bold', color='black')
    
    # Set up the plot
    if title is None:
        title = f"Trajectory: {Path(trajectory_file).name}"
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Invert Y axis to match screen coordinates
    
    if standalone:
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        plt.show()
    
    return ax


def visualize_trajectory_comparison(trajectory1_file: str, trajectory2_file: str, 
                                  similarity_score: float, save_path: Optional[str] = None):
    """
    Create a visualization comparing two trajectories.
    
    Args:
        trajectory1_file: Path to first trajectory file
        trajectory2_file: Path to second trajectory file
        similarity_score: Similarity score between the trajectories
        save_path: Optional path to save the visualization image
    
    Note:
        This function requires matplotlib. Install with: pip install matplotlib
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot both trajectories using the general function
    visualize_trajectory(trajectory1_file, ax1, f"Trajectory 1: {Path(trajectory1_file).name}")
    visualize_trajectory(trajectory2_file, ax2, f"Trajectory 2: {Path(trajectory2_file).name}")
    
    # Add similarity score
    fig.suptitle(f"Trajectory Comparison - Similarity: {similarity_score:.3f}", 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def list_trajectory_files(directory: str = ".") -> List[str]:
    """
    List all trajectory JSON files in a directory.
    
    Args:
        directory: Directory to search for trajectory files
    
    Returns:
        List of trajectory file paths
    """
    directory = Path(directory)
    trajectory_files = list(directory.glob("trajectory_*.json"))
    return [str(f) for f in sorted(trajectory_files)]