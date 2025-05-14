// dilate_atom_centric_forward_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int THREADS_PER_BLOCK = 256;

__device__ inline int modulo(int x, int N) {
    int ret = x % N;
    if (ret < 0)
        ret += N;
    return ret;
}

__global__ void dilate_atom_centric_forward_kernel(
    const float* atom_coords_grid,
    const float* atom_occupancies,
    const float* radial_profiles,
    const float r_step,
    const float rmax_cartesian,
    const int* lmax_grid_units,
    const int* grid_dims,
    const float* grid_to_cartesian_matrix,
    float* output_density_grid,
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
    
    // Precompute constants
    int out_slice = Dy * Dx;
    int out_size = Dz * out_slice;
    float rmax2 = rmax_cartesian * rmax_cartesian;
    
    // Loop over nearby voxels (optimized order like CPU implementation)
    for (int gz_offset = -lmax_gz; gz_offset <= lmax_gz; ++gz_offset) {
        int Gz_abs = az_floor + gz_offset;
        float delta_gz = az - (float)Gz_abs;  // FIXED SIGN
        float dz = grid_to_cartesian_matrix[2*3+0]*delta_gz + 
                   grid_to_cartesian_matrix[2*3+1]*delta_gz + 
                   grid_to_cartesian_matrix[2*3+2]*delta_gz;
        float dz2 = dz * dz;
        
        for (int gy_offset = -lmax_gy; gy_offset <= lmax_gy; ++gy_offset) {
            int Gy_abs = ay_floor + gy_offset;
            float delta_gy = ay - (float)Gy_abs;  // FIXED SIGN
            float dy = grid_to_cartesian_matrix[1*3+0]*delta_gy + 
                       grid_to_cartesian_matrix[1*3+1]*delta_gy + 
                       grid_to_cartesian_matrix[1*3+2]*delta_gy;
            float dz2_dy2 = dz2 + dy * dy;
            
            for (int gx_offset = -lmax_gx; gx_offset <= lmax_gx; ++gx_offset) {
                int Gx_abs = ax_floor + gx_offset;
                float delta_gx = ax - (float)Gx_abs;  // FIXED SIGN
                float dx = grid_to_cartesian_matrix[0*3+0]*delta_gx + 
                           grid_to_cartesian_matrix[0*3+1]*delta_gx + 
                           grid_to_cartesian_matrix[0*3+2]*delta_gx;
                float d2_zyx = dz2_dy2 + dx * dx;
                
                if (d2_zyx <= rmax2) {
                    // Calculate Cartesian distance
                    float dist_cart = sqrtf(d2_zyx);
                    
                    // Compute radial density (using nearest int approach from CPU code for speed)
                    int r_idx = __float2int_rd(dist_cart / r_step + 0.5f);
                    r_idx = min(r_idx, N_radial_points - 1);
                    float density_value = current_radial_profile[r_idx];
                    
                    // Apply occupancy
                    float final_contrib = occ * density_value;
                    
                    // Determine final grid voxel (periodic boundary conditions)
                    int Gx_final = modulo(Gx_abs, Dx);
                    int Gy_final = modulo(Gy_abs, Dy);
                    int Gz_final = modulo(Gz_abs, Dz);
                    
                    // Flat index into output array with batch
                    unsigned long long flat_idx_grid = 
                        (unsigned long long)b_idx * Dz * Dy * Dx + 
                        (unsigned long long)Gz_final * Dy * Dx + 
                        (unsigned long long)Gy_final * Dx + Gx_final;
                    
                    // Atomic add to output grid
                    atomicAdd(&output_density_grid[flat_idx_grid], final_contrib);
                }
            }
        }
    }
}

__global__ void dilate_atom_centric_backward_kernel(
    const float* grad_output_density_grid,
    const float* atom_coords_grid,
    const float* atom_occupancies,
    const float* radial_profiles,
    const float* radial_profiles_derivatives,
    const float r_step,
    const float rmax_cartesian,
    const int* lmax_grid_units,
    const int* grid_dims,
    const float* grid_to_cartesian_matrix,
    float* grad_atom_coords_grid,
    float* grad_atom_occupancies,
    float* grad_radial_profiles,
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
    
    // Precompute constants
    int out_slice = Dy * Dx;
    int out_size = Dz * out_slice;
    float rmax2 = rmax_cartesian * rmax_cartesian;
    
    // Accumulate gradients for this atom
    float grad_ax = 0.0f;
    float grad_ay = 0.0f;
    float grad_az = 0.0f;
    float grad_occ = 0.0f;
    
    // Loop over nearby voxels in optimized order
    for (int gz_offset = -lmax_gz; gz_offset <= lmax_gz; ++gz_offset) {
        int Gz_abs = az_floor + gz_offset;
        float delta_gz = az - (float)Gz_abs;  // FIXED SIGN
        float dz = grid_to_cartesian_matrix[2*3+0]*delta_gz + 
                   grid_to_cartesian_matrix[2*3+1]*delta_gz + 
                   grid_to_cartesian_matrix[2*3+2]*delta_gz;
        float dz2 = dz * dz;
        
        for (int gy_offset = -lmax_gy; gy_offset <= lmax_gy; ++gy_offset) {
            int Gy_abs = ay_floor + gy_offset;
            float delta_gy = ay - (float)Gy_abs;  // FIXED SIGN
            float dy = grid_to_cartesian_matrix[1*3+0]*delta_gy + 
                       grid_to_cartesian_matrix[1*3+1]*delta_gy + 
                       grid_to_cartesian_matrix[1*3+2]*delta_gy;
            float dz2_dy2 = dz2 + dy * dy;
            
            for (int gx_offset = -lmax_gx; gx_offset <= lmax_gx; ++gx_offset) {
                int Gx_abs = ax_floor + gx_offset;
                float delta_gx = ax - (float)Gx_abs;  // FIXED SIGN
                float dx = grid_to_cartesian_matrix[0*3+0]*delta_gx + 
                           grid_to_cartesian_matrix[0*3+1]*delta_gx + 
                           grid_to_cartesian_matrix[0*3+2]*delta_gx;
                float d2_zyx = dz2_dy2 + dx * dx;
                
                if (d2_zyx <= rmax2) {
                    // Calculate Cartesian distance
                    float dist_cart = sqrtf(d2_zyx);
                    
                    // Get the index into radial profiles using the nearest int approach
                    int r_idx = __float2int_rd(dist_cart / r_step + 0.5f);
                    r_idx = min(r_idx, N_radial_points - 1);
                    
                    // Get density value
                    float density_value = current_radial_profile[r_idx];
                    
                    // Determine final grid voxel (periodic boundary conditions)
                    int Gx_final = modulo(Gx_abs, Dx);
                    int Gy_final = modulo(Gy_abs, Dy);
                    int Gz_final = modulo(Gz_abs, Dz);
                    
                    // Flat index into output array
                    unsigned long long flat_idx_grid = 
                        (unsigned long long)b_idx * Dz * Dy * Dx + 
                        (unsigned long long)Gz_final * Dy * Dx + 
                        (unsigned long long)Gy_final * Dx + Gx_final;
                    
                    // Get upstream gradient
                    float dL_dRho_voxel = grad_output_density_grid[flat_idx_grid];
                    
                    // Gradient w.r.t. occupancy
                    grad_occ += dL_dRho_voxel * density_value;
                    
                    // Gradient w.r.t. Radial Profile Points
                    atomicAdd(&grad_radial_profiles[b_idx * N_atoms * N_radial_points + atom_i * N_radial_points + r_idx], 
                              dL_dRho_voxel * occ);
                    
                    // Gradient w.r.t. coordinates requires derivative w.r.t. radius
                    float deriv_at_r = current_radial_profile_deriv[r_idx];
                    
                    // Only compute coordinate gradients if distance is non-zero
                    if (dist_cart > 1e-9f) {
                        float inv_dist = 1.0f / dist_cart;
                        float common_factor = dL_dRho_voxel * occ * deriv_at_r * inv_dist;
                        
                        // Coordinate gradients
                        grad_ax += common_factor * dx * grid_to_cartesian_matrix[0*3+0];
                        grad_ay += common_factor * dy * grid_to_cartesian_matrix[1*3+1];
                        grad_az += common_factor * dz * grid_to_cartesian_matrix[2*3+2];
                    }
                }
            }
        }
    }
    
    // Atomically update the gradients for this atom
    atomicAdd(&grad_atom_coords_grid[b_idx * N_atoms * 3 + atom_i * 3 + 0], grad_ax);
    atomicAdd(&grad_atom_coords_grid[b_idx * N_atoms * 3 + atom_i * 3 + 1], grad_ay);
    atomicAdd(&grad_atom_coords_grid[b_idx * N_atoms * 3 + atom_i * 3 + 2], grad_az);
    atomicAdd(&grad_atom_occupancies[b_idx * N_atoms + atom_i], grad_occ);
}