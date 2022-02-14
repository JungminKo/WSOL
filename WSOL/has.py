"""
Original repository: https://github.com/kkanshul/Hide-and-Seek
Reference repository : https://github.com/GenDisc/IVR
"""
import random

def has(image, drop_rate, grid_size=None):
    """
    Args:
        image: torch.Tensor, N x C x H x W, float32.
        drop_rate: float (0~1)
        grid_size: int or None (if None, it will be randomly chosen)
    Returns:
        image: torch.Tensor, N x C x H x W, float32.
    """
    if grid_size == None:
        grid_size = random.choice([0, 16, 32, 44, 56])
        
    if grid_size == 0:
        return image

    batch_size, n_channels, height, width = image.size()

    for batch_idx in range(batch_size):
        for x in range(0, width, grid_size):
            for y in range(0, height, grid_size):
                x_end = min(height, x + grid_size)
                y_end = min(height, y + grid_size)
                if random.random() <= drop_rate:
                    image[batch_idx, :, x:x_end, y:y_end] = 0.
    return image