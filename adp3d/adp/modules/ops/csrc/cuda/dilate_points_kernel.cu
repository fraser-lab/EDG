// atomdensity_cuda_kernels.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

const int THREADS_PER_BLOCK = 256;

// Forward Pass CUDA Kernel
__global__ void dilate_atom_centric_forward_kernel(
    const float* atom_coords_grid,       // [batch_size, N_atoms, 3]
    const float* atom_occupancies,       // [batch_size, N_atoms]
    const float* radial_profiles,        // [batch_size, N_atoms, N_radial_points]
    const float r_step,                  // Scalar
    const float rmax_cartesian,          // Scalar
    const int* lmax_grid_units,          // [3] - [lx, ly, lz]
    const int* grid_dims,                // [3] - [Dz, Dy, Dx]
    const float* grid_to_cartesian_matrix, // [3, 3]
    float* output_density_grid,          // [batch_size, Dz, Dy, Dx]
    const int batch_size,
    const int N_atoms,
    const int N_radial_points) 
{
    // Thread identification
    int global_atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_atom_idx >= batch_size * N_atoms) return;
    
    int b_idx = global_atom_idx / N_atoms;
    int atom_i = global_atom_idx % N_atoms;
    
    // Load atom-specific data
    float ax = atom_coords_grid[b_idx * N_atoms * 3 + atom_i * 3 + 0];
    float ay = atom_coords_grid[b_idx * N_atoms * 3 + atom_i * 3 + 1];
    float az = atom_coords_grid[b_idx * N_atoms * 3 + atom_i * 3 + 2];
    float occ = atom_occupancies[b_idx * N_atoms + atom_i];
    
    const float* current_radial_profile = &radial_profiles[b_idx * N_atoms * N_radial_points + atom_i * N_radial_points];
    
    // Grid dimensions
    int Dx = grid_dims[2];
    int Dy = grid_dims[1];
    int Dz = grid_dims[0];
    
    // Determine voxel iteration box
    int ax_floor = floorf(ax);
    int ay_floor = floorf(ay);
    int az_floor = floorf(az);
    
    int lmax_gx = lmax_grid_units[0];
    int lmax_gy = lmax_grid_units[1];
    int lmax_gz = lmax_grid_units[2];
    
    // Loop over nearby voxels
    for (int gz_offset = -lmax_gz; gz_offset <= lmax_gz; ++gz_offset) {
        for (int gy_offset = -lmax_gy; gy_offset <= lmax_gy; ++gy_offset) {
            for (int gx_offset = -lmax_gx; gx_offset <= lmax_gx; ++gx_offset) {
                // Target voxel absolute grid index
                int Gx_abs = ax_floor + gx_offset;
                int Gy_abs = ay_floor + gy_offset;
                int Gz_abs = az_floor + gz_offset;
                
                // Vector from atom to target voxel origin (in grid units)
                float delta_gx = (float)Gx_abs - ax;
                float delta_gy = (float)Gy_abs - ay;
                float delta_gz = (float)Gz_abs - az;
                
                // Transform delta vector to Cartesian
                float delta_cx = grid_to_cartesian_matrix[0*3+0]*delta_gx + 
                                 grid_to_cartesian_matrix[0*3+1]*delta_gy + 
                                 grid_to_cartesian_matrix[0*3+2]*delta_gz;
                
                float delta_cy = grid_to_cartesian_matrix[1*3+0]*delta_gx + 
                                 grid_to_cartesian_matrix[1*3+1]*delta_gy + 
                                 grid_to_cartesian_matrix[1*3+2]*delta_gz;
                
                float delta_cz = grid_to_cartesian_matrix[2*3+0]*delta_gx + 
                                 grid_to_cartesian_matrix[2*3+1]*delta_gy + 
                                 grid_to_cartesian_matrix[2*3+2]*delta_gz;
                
                // Calculate Cartesian distance & check rmax_cartesian
                float dist_sq_cart = delta_cx*delta_cx + delta_cy*delta_cy + delta_cz*delta_cz;
                if (dist_sq_cart >= rmax_cartesian * rmax_cartesian) continue;
                float dist_cart = sqrtf(dist_sq_cart);
                
                // Interpolate radial density (linear interpolation)
                float rad_continuous_idx = dist_cart / r_step;
                int r_idx_low = floorf(rad_continuous_idx);
                float weight_high = rad_continuous_idx - (float)r_idx_low;
                float weight_low = 1.0f - weight_high;
                
                // Clamp indices
                r_idx_low = max(0, min(r_idx_low, N_radial_points - 1));
                int r_idx_high = min(r_idx_low + 1, N_radial_points - 1);
                
                float density_at_r = weight_low * current_radial_profile[r_idx_low] + 
                                     weight_high * current_radial_profile[r_idx_high];
                
                // Apply occupancy
                float final_contrib = occ * density_at_r;
                
                // Determine final grid voxel (periodic boundary conditions)
                int Gx_final = ((Gx_abs % Dx) + Dx) % Dx;
                int Gy_final = ((Gy_abs % Dy) + Dy) % Dy;
                int Gz_final = ((Gz_abs % Dz) + Dz) % Dz;
                
                // Atomic add to output grid
                unsigned long long flat_idx_grid = 
                    (unsigned long long)b_idx * Dz * Dy * Dx + 
                    (unsigned long long)Gz_final * Dy * Dx + 
                    (unsigned long long)Gy_final * Dx + Gx_final;
                
                atomicAdd(&output_density_grid[flat_idx_grid], final_contrib);
            }
        }
    }
}

// Backward Pass CUDA Kernel
__global__ void dilate_atom_centric_backward_kernel(
    const float* grad_output_density_grid, // [batch_size, Dz, Dy, Dx]
    const float* atom_coords_grid,         // [batch_size, N_atoms, 3]
    const float* atom_occupancies,         // [batch_size, N_atoms]
    const float* radial_profiles,          // [batch_size, N_atoms, N_radial_points]
    const float* radial_profiles_derivatives, // [batch_size, N_atoms, N_radial_points]
    const float r_step,                    // Scalar
    const float rmax_cartesian,            // Scalar
    const int* lmax_grid_units,            // [3] - [lx, ly, lz]
    const int* grid_dims,                  // [3] - [Dz, Dy, Dx]
    const float* grid_to_cartesian_matrix, // [3, 3]
    float* grad_atom_coords_grid,          // [batch_size, N_atoms, 3]
    float* grad_atom_occupancies,          // [batch_size, N_atoms]
    float* grad_radial_profiles,           // [batch_size, N_atoms, N_radial_points]
    const int batch_size,
    const int N_atoms,
    const int N_radial_points)
{
    // Thread identification
    int global_atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_atom_idx >= batch_size * N_atoms) return;
    
    int b_idx = global_atom_idx / N_atoms;
    int atom_i = global_atom_idx % N_atoms;
    
    // Load atom-specific data
    float ax = atom_coords_grid[b_idx * N_atoms * 3 + atom_i * 3 + 0];
    float ay = atom_coords_grid[b_idx * N_atoms * 3 + atom_i * 3 + 1];
    float az = atom_coords_grid[b_idx * N_atoms * 3 + atom_i * 3 + 2];
    float occ = atom_occupancies[b_idx * N_atoms + atom_i];
    
    const float* current_radial_profile = &radial_profiles[b_idx * N_atoms * N_radial_points + atom_i * N_radial_points];
    const float* current_radial_profile_deriv = &radial_profiles_derivatives[b_idx * N_atoms * N_radial_points + atom_i * N_radial_points];
    
    // Grid dimensions
    int Dx = grid_dims[2];
    int Dy = grid_dims[1];
    int Dz = grid_dims[0];
    
    // Determine voxel iteration box
    int ax_floor = floorf(ax);
    int ay_floor = floorf(ay);
    int az_floor = floorf(az);
    
    int lmax_gx = lmax_grid_units[0];
    int lmax_gy = lmax_grid_units[1];
    int lmax_gz = lmax_grid_units[2];
    
    // Loop over nearby voxels
    for (int gz_offset = -lmax_gz; gz_offset <= lmax_gz; ++gz_offset) {
        for (int gy_offset = -lmax_gy; gy_offset <= lmax_gy; ++gy_offset) {
            for (int gx_offset = -lmax_gx; gx_offset <= lmax_gx; ++gx_offset) {
                // Target voxel absolute grid index
                int Gx_abs = ax_floor + gx_offset;
                int Gy_abs = ay_floor + gy_offset;
                int Gz_abs = az_floor + gz_offset;
                
                // Vector from atom to target voxel origin (in grid units)
                float delta_gx = (float)Gx_abs - ax;
                float delta_gy = (float)Gy_abs - ay;
                float delta_gz = (float)Gz_abs - az;
                
                // Transform delta vector to Cartesian
                float delta_cx = grid_to_cartesian_matrix[0*3+0]*delta_gx + 
                                 grid_to_cartesian_matrix[0*3+1]*delta_gy + 
                                 grid_to_cartesian_matrix[0*3+2]*delta_gz;
                
                float delta_cy = grid_to_cartesian_matrix[1*3+0]*delta_gx + 
                                 grid_to_cartesian_matrix[1*3+1]*delta_gy + 
                                 grid_to_cartesian_matrix[1*3+2]*delta_gz;
                
                float delta_cz = grid_to_cartesian_matrix[2*3+0]*delta_gx + 
                                 grid_to_cartesian_matrix[2*3+1]*delta_gy + 
                                 grid_to_cartesian_matrix[2*3+2]*delta_gz;
                
                // Calculate Cartesian distance & check rmax_cartesian
                float dist_sq_cart = delta_cx*delta_cx + delta_cy*delta_cy + delta_cz*delta_cz;
                if (dist_sq_cart >= rmax_cartesian * rmax_cartesian) continue;
                float dist_cart = sqrtf(dist_sq_cart);
                
                // Interpolate radial density (linear interpolation)
                float rad_continuous_idx = dist_cart / r_step;
                int r_idx_low = floorf(rad_continuous_idx);
                float weight_high = rad_continuous_idx - (float)r_idx_low;
                float weight_low = 1.0f - weight_high;
                
                // Clamp indices
                r_idx_low = max(0, min(r_idx_low, N_radial_points - 1));
                int r_idx_high = min(r_idx_low + 1, N_radial_points - 1);
                
                // Determine final grid voxel (periodic boundary conditions)
                int Gx_final = ((Gx_abs % Dx) + Dx) % Dx;
                int Gy_final = ((Gy_abs % Dy) + Dy) % Dy;
                int Gz_final = ((Gz_abs % Dz) + Dz) % Dz;
                
                // Get upstream gradient
                unsigned long long flat_idx_grid = 
                    (unsigned long long)b_idx * Dz * Dy * Dx + 
                    (unsigned long long)Gz_final * Dy * Dx + 
                    (unsigned long long)Gy_final * Dx + Gx_final;
                float dL_dRho_voxel = grad_output_density_grid[flat_idx_grid];
                
                // Recompute density contribution (before occupancy)
                float density_at_r = weight_low * current_radial_profile[r_idx_low] + 
                                     weight_high * current_radial_profile[r_idx_high];
                
                // Gradient w.r.t. Occupancy (dL/d_occ)
                float dL_dOcc_contrib = dL_dRho_voxel * density_at_r;
                atomicAdd(&grad_atom_occupancies[b_idx * N_atoms + atom_i], dL_dOcc_contrib);
                
                // Gradient w.r.t. Radial Profile Points (dL/d_radial_profile_points)
                float dL_dRadProfLow_contrib = dL_dRho_voxel * occ * weight_low;
                float dL_dRadProfHigh_contrib = dL_dRho_voxel * occ * weight_high;
                atomicAdd(&grad_radial_profiles[b_idx * N_atoms * N_radial_points + atom_i * N_radial_points + r_idx_low], 
                          dL_dRadProfLow_contrib);
                atomicAdd(&grad_radial_profiles[b_idx * N_atoms * N_radial_points + atom_i * N_radial_points + r_idx_high], 
                          dL_dRadProfHigh_contrib);
                
                // Gradient w.r.t. Atom Coordinates (dL/d_atom_coords_grid)
                
                // A. Derivative of Interpolated Density w.r.t. Cartesian Distance
                float deriv_at_r_low = current_radial_profile_deriv[r_idx_low];
                float deriv_at_r_high = current_radial_profile_deriv[r_idx_high];
                float d_density_at_r_d_dist_cart = weight_low * deriv_at_r_low + weight_high * deriv_at_r_high;
                
                // B. Derivative of Cartesian Distance w.r.t. Atom Grid Coordinate
                float common_dist_term = (dist_cart > 1e-9f) ? (1.0f / dist_cart) : 0.0f;
                
                // Derivatives with respect to atom coordinates
                float d_dist_cart_d_ax = common_dist_term * (
                    delta_cx * (-grid_to_cartesian_matrix[0*3+0]) + 
                    delta_cy * (-grid_to_cartesian_matrix[1*3+0]) + 
                    delta_cz * (-grid_to_cartesian_matrix[2*3+0])
                );
                
                float d_dist_cart_d_ay = common_dist_term * (
                    delta_cx * (-grid_to_cartesian_matrix[0*3+1]) + 
                    delta_cy * (-grid_to_cartesian_matrix[1*3+1]) + 
                    delta_cz * (-grid_to_cartesian_matrix[2*3+1])
                );
                
                float d_dist_cart_d_az = common_dist_term * (
                    delta_cx * (-grid_to_cartesian_matrix[0*3+2]) + 
                    delta_cy * (-grid_to_cartesian_matrix[1*3+2]) + 
                    delta_cz * (-grid_to_cartesian_matrix[2*3+2])
                );
                
                // C. Combine for Coordinate Gradients
                float dL_dAx_contrib = dL_dRho_voxel * occ * d_density_at_r_d_dist_cart * d_dist_cart_d_ax;
                float dL_dAy_contrib = dL_dRho_voxel * occ * d_density_at_r_d_dist_cart * d_dist_cart_d_ay;
                float dL_dAz_contrib = dL_dRho_voxel * occ * d_density_at_r_d_dist_cart * d_dist_cart_d_az;
                
                atomicAdd(&grad_atom_coords_grid[b_idx * N_atoms * 3 + atom_i * 3 + 0], dL_dAx_contrib);
                atomicAdd(&grad_atom_coords_grid[b_idx * N_atoms * 3 + atom_i * 3 + 1], dL_dAy_contrib);
                atomicAdd(&grad_atom_coords_grid[b_idx * N_atoms * 3 + atom_i * 3 + 2], dL_dAz_contrib);
            }
        }
    }
}