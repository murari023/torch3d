import torch


__all__ = [
    'furthest_point_sample'
]


def furthest_point_sample(points, num_samples):
    device = points.device
    batch_size = points.shape[0]
    num_points = points.shape[1]
    channels = points.shape[2]

    indices = torch.zeros(batch_size, num_samples, dtype=torch.int64).to(device)
    sqdists = torch.ones(batch_size, num_points).to(device) * 1e10
    furthest = torch.randint(0, num_points, (batch_size,), dtype=torch.int64).to(device)
    batches = torch.arange(batch_size, dtype=torch.int64).to(device)

    for i in range(num_samples):
        indices[:, i] = furthest
        centroid = points[batches, furthest, :].view(batch_size, 1, channels)
        dist = torch.sum((points - centroid) ** 2, dim=-1)
        mask = dist < sqdists
        sqdists[mask] = dist[mask]
        furthest = torch.argmax(sqdists, dim=-1)
    return indices
