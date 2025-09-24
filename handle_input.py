from utils import convert2centroids, centroids2model_input, normalize_flatten_data_centroid
import torch
import numpy as np

def bboxes_smoothing(boxes):
    """
    Smooth bounding boxes using a moving median filter.
    """
    boxes = np.array(boxes)
    boxes_smoothed = np.copy(boxes)
    smoothing_window = 10
    for i in range(len(boxes)):
        l = max(0, i-smoothing_window//2)
        r = min(len(boxes), i+smoothing_window//2)
        boxes_smoothed[i] = np.median(boxes[l:r], axis=0)
    return boxes_smoothed

def bboxes_to_model_input(bbox_sequences):
    """
    Convert bounding box sequences to model input format.
    
    Args:
        bbox_sequences: List of bbox sequences, where each sequence is a list of [x1, y1, x2, y2]
                       Example: [[[100, 200, 150, 250], [105, 205, 155, 255], ...],  # Car 1
                                [[300, 400, 350, 450], [305, 405, 355, 455], ...]]  # Car 2
    
    Returns:
        torch.Tensor: Model input of shape (1, 128, 8) ready for the encoder
    """
    # Convert bboxes to centroids
    centroids = convert2centroids(bbox_sequences)
    
    # Convert to model input format
    model_input = centroids2model_input(centroids)
    
    # Add batch dimension
    return model_input.unsqueeze(0)

def positions_to_model_input(position_sequences):
    """
    Convert position sequences to model input format.
    
    Args:
        position_sequences: List of position sequences, where each sequence is a list of [x, y]
                           Example: [[[100, 200], [105, 205], [110, 210], ...],  # Car 1
                                    [[300, 400], [305, 405], [310, 410], ...]]  # Car 2
    
    Returns:
        torch.Tensor: Model input of shape (1, 128, 8) ready for the encoder
    """
    centroids = np.array(position_sequences)
    
    # Normalize and format for model
    model_input = normalize_flatten_data_centroid(centroids)
    model_input = torch.from_numpy(model_input).float()
    
    # Add batch dimension
    return model_input.unsqueeze(0)

def get_embedding(model, model_input):
    """
    Get embedding from the model for a given input.
    
    Args:
        model: The EncoderModelWrapper instance
        model_input: Tensor of shape (1, 128, 8)
    
    Returns:
        torch.Tensor: Embedding vector
    """
    with torch.no_grad():
        embedding = model.net(model_input.to(model.device))
    return embedding

def get_rotation_averaged_embedding(model, position_sequences):
    """
    Get rotation averaged embedding from the model for a given input.
    """
    from utils import rotate_clip_centroids

    q_inputs = [centroids2model_input(position_sequences)]
    for _ in range(100):
        q_inputs.append(centroids2model_input(rotate_clip_centroids(position_sequences)))
    query_embeds = model.net(torch.stack(q_inputs).to(model.device))
    query_embed_avg = torch.mean(query_embeds,dim=0)
    return query_embed_avg
