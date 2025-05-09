import torch

def trilinear_interpolation_torch(
    density_map: torch.Tensor, points_zyx: torch.Tensor
) -> torch.Tensor:
    """
    Trilinear interpolation on a 3D map.

    Parameters:
    ----------
    density_map : torch.Tensor
        The 3D density map, expected to have shape [Mz, My, Mx],
        where Mz, My, Mx are the grid dimensions along z, y, and x axes.
    points_zyx : torch.Tensor
        A tensor of points at which to interpolate values, with shape
        [..., N_points, 3], where "..." represents any number of leading
        batch dimensions. Each innermost [N_points, 3] slice represents
        a set of points, with each point having (z, y, x) coordinates
        in grid units.
        It is assumed that any necessary periodic boundary conditions
        (modulo operations) have already been applied to these points
        if the map represents a unit cell.

    Returns:
    -------
    torch.Tensor
        A tensor of shape [..., N_points] containing the interpolated values,
        matching the leading dimensions of `points_zyx`.

    Notes:
    -----
    - Assumes integer grid coordinates correspond to voxel corners.
    - For points on a grid line, torch.floor determines the lower indexed voxel.
    - The device and dtype of the output will match `density_map`.
    """
    if not isinstance(density_map, torch.Tensor) or not isinstance(
        points_zyx, torch.Tensor
    ):
        raise TypeError("Inputs 'density_map' and 'points_zyx' must be torch.Tensors.")
    if density_map.ndim != 3:
        raise ValueError(
            f"Input 'density_map' must be a 3D tensor (shape [Mz, My, Mx]), "
            f"got shape {density_map.shape}."
        )
    if points_zyx.shape[-1] != 3:
        raise ValueError(
            f"The last dimension of 'points_zyx' must be 3 (for z,y,x coordinates), "
            f"got shape {points_zyx.shape}."
        )

    Mz, My, Mx = density_map.shape
    device = density_map.device
    dtype = density_map.dtype

    # Store original shape for reshaping the output
    original_shape = points_zyx.shape
    num_points_total = original_shape[:-1].numel() # Product of all dimensions except the last one

    # Reshape points_zyx to [N_total_points, 3] for processing
    points_zyx_flat = points_zyx.reshape(num_points_total, 3)
    points_zyx_flat = points_zyx_flat.to(device=device, dtype=dtype)

    z, y, x = points_zyx_flat[:, 0], points_zyx_flat[:, 1], points_zyx_flat[:, 2]

    # Get the integer part of the coordinates
    z0 = torch.floor(z).long()
    y0 = torch.floor(y).long()
    x0 = torch.floor(x).long()

    # Calculate the fractional parts (weights for the '+1' index)
    zd = z - z0.to(dtype)
    yd = y - y0.to(dtype)
    xd = x - x0.to(dtype)

    # Clamp base indices to be within the valid map range [0, Dim-1]
    z0_clamped = torch.clamp(z0, 0, Mz - 1)
    y0_clamped = torch.clamp(y0, 0, My - 1)
    x0_clamped = torch.clamp(x0, 0, Mx - 1)

    # Calculate the 'upper' voxel indices using modulo for periodic boundaries
    z1 = (z0_clamped + 1) % Mz
    y1 = (y0_clamped + 1) % My
    x1 = (x0_clamped + 1) % Mx

    # Gather the density values at the 8 corners of the voxel
    v000 = density_map[z0_clamped, y0_clamped, x0_clamped]
    v001 = density_map[z0_clamped, y0_clamped, x1]
    v010 = density_map[z0_clamped, y1, x0_clamped]
    v011 = density_map[z0_clamped, y1, x1]
    v100 = density_map[z1, y0_clamped, x0_clamped]
    v101 = density_map[z1, y0_clamped, x1]
    v110 = density_map[z1, y1, x0_clamped]
    v111 = density_map[z1, y1, x1]

    # Interpolate along x-axis
    c00 = v000 * (1 - xd) + v001 * xd
    c01 = v010 * (1 - xd) + v011 * xd
    c10 = v100 * (1 - xd) + v101 * xd
    c11 = v110 * (1 - xd) + v111 * xd

    # Interpolate along y-axis
    c0 = c00 * (1 - yd) + c01 * yd
    c1 = c10 * (1 - yd) + c11 * yd

    # Interpolate along z-axis
    interpolated_values_flat = c0 * (1 - zd) + c1 * zd

    # Reshape the output to match the leading dimensions of points_zyx,
    # with the last dimension being the number of points.
    output_shape = list(original_shape[:-1])
    interpolated_values = interpolated_values_flat.reshape(output_shape)

    return interpolated_values