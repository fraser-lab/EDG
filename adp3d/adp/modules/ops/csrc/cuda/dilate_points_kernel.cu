// dilate_atom_centric_corrected_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int THREADS_PER_BLOCK = 256;

__device__ inline int modulo(int x, int N)
{
    int ret = x % N;
    if (ret < 0)
        ret += N;
    return ret;
}

__device__ inline int NEAREST_INT(float a)
{
    return (int)(a + 0.5f);
}

__global__ void dilate_atom_centric_forward_kernel(
    const float *points,
    const float *occupancies,
    const float *radial_densities,
    const float rstep,
    const float rmax,
    const int *lmax,
    const int *grid_dims,
    const float *grid_to_cartesian,
    float *out,
    const int batch_size,
    const int N_atoms,
    const int N_radial_points)
{
    int global_atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_atom_idx >= batch_size * N_atoms)
        return;

    int b_idx = global_atom_idx / N_atoms;
    int atom_i = global_atom_idx % N_atoms;

    float center_a = points[b_idx * N_atoms * 3 + atom_i * 3 + 0];
    float center_b = points[b_idx * N_atoms * 3 + atom_i * 3 + 1];
    float center_c = points[b_idx * N_atoms * 3 + atom_i * 3 + 2];
    float q = occupancies[b_idx * N_atoms + atom_i];

    const float *curr_radial_densities = &radial_densities[b_idx * N_atoms * N_radial_points + atom_i * N_radial_points];

    int Dx = grid_dims[2];
    int Dy = grid_dims[1];
    int Dz = grid_dims[0];

    int amin = (int)floorf(center_a - lmax[0]);
    int bmin = (int)floorf(center_b - lmax[1]);
    int cmin = (int)floorf(center_c - lmax[2]);

    int amax = (int)floorf(center_a + lmax[0]);
    int bmax = (int)floorf(center_b + lmax[1]);
    int cmax = (int)floorf(center_c + lmax[2]);

    int out_slice = Dy * Dx;
    float rmax2 = rmax * rmax;

    for (int c = cmin; c <= cmax; ++c)
    {
        int ind_c = modulo(c * out_slice, Dz * out_slice);
        float dc = center_c - c;
        float dz = grid_to_cartesian[8] * dc;
        float dz2 = dz * dz;
        float dy_c = grid_to_cartesian[5] * dc;
        float dx_c = grid_to_cartesian[2] * dc;

        for (int b = bmin; b <= bmax; ++b)
        {
            int ind_cb = modulo(b * Dx, out_slice) + ind_c;
            float db = center_b - b;
            float dy = dy_c + grid_to_cartesian[4] * db;
            float d2_zy = dz2 + dy * dy;
            float dx_cb = dx_c + grid_to_cartesian[1] * db;

            for (int a = amin; a <= amax; ++a)
            {
                float da = center_a - a;
                float dx = dx_cb + grid_to_cartesian[0] * da;
                float d2_zyx = d2_zy + dx * dx;

                if (d2_zyx <= rmax2)
                {
                    float r = sqrtf(d2_zyx);

                    int index = NEAREST_INT(r / rstep);
                    index = min(index, N_radial_points - 1);

                    int out_index = ind_cb + modulo(a, Dx);
                    atomicAdd(&out[b_idx * Dz * Dy * Dx + out_index],
                              q * curr_radial_densities[index]);
                }
            }
        }
    }
}

__global__ void dilate_atom_centric_backward_kernel(
    const float *grad_out,
    const float *points,
    const float *occupancies,
    const float *radial_densities,
    const float *derivatives,
    const float rstep,
    const float rmax,
    const int *lmax,
    const int *grid_dims,
    const float *grid_to_cartesian,
    float *grad_points,
    float *grad_occupancies,
    float *grad_radial_densities,
    const int batch_size,
    const int N_atoms,
    const int N_radial_points)
{
    int global_atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_atom_idx >= batch_size * N_atoms)
        return;

    int b_idx = global_atom_idx / N_atoms;
    int atom_i = global_atom_idx % N_atoms;

    float center_a = points[b_idx * N_atoms * 3 + atom_i * 3 + 0];
    float center_b = points[b_idx * N_atoms * 3 + atom_i * 3 + 1];
    float center_c = points[b_idx * N_atoms * 3 + atom_i * 3 + 2];
    float q = occupancies[b_idx * N_atoms + atom_i];

    int Dx = grid_dims[2];
    int Dy = grid_dims[1];
    int Dz = grid_dims[0];

    int amin = (int)ceilf(center_a - lmax[0]);
    int bmin = (int)ceilf(center_b - lmax[1]);
    int cmin = (int)ceilf(center_c - lmax[2]);

    int amax = (int)floorf(center_a + lmax[0]);
    int bmax = (int)floorf(center_b + lmax[1]);
    int cmax = (int)floorf(center_c + lmax[2]);

    int out_slice = Dy * Dx;
    float rmax2 = rmax * rmax;

    float grad_center_a = 0.0f;
    float grad_center_b = 0.0f;
    float grad_center_c = 0.0f;
    float grad_q = 0.0f;

    int derivatives_ind = b_idx * N_atoms * N_radial_points + atom_i * N_radial_points;

    for (int c = cmin; c <= cmax; ++c)
    {
        int ind_c = modulo(c * out_slice, Dz * out_slice);
        float dc = center_c - c;
        float dz = grid_to_cartesian[8] * dc;
        float dz2 = dz * dz;
        float dy_c = grid_to_cartesian[5] * dc;
        float dx_c = grid_to_cartesian[2] * dc;

        int gradient_z_idx = NEAREST_INT(fabsf(dz) / rstep);
        gradient_z_idx = min(gradient_z_idx, N_radial_points - 1);
        float gradient_z = derivatives[derivatives_ind + gradient_z_idx];
        if (dz < 0)
        {
            gradient_z *= -1;
        }

        for (int b = bmin; b <= bmax; ++b)
        {
            int ind_cb = modulo(b * Dx, out_slice) + ind_c;
            float db = center_b - b;
            float dy = dy_c + grid_to_cartesian[4] * db;
            float d2_zy = dz2 + dy * dy;
            float dx_cb = dx_c + grid_to_cartesian[1] * db;

            int gradient_y_idx = NEAREST_INT(fabsf(dy) / rstep);
            gradient_y_idx = min(gradient_y_idx, N_radial_points - 1);
            float gradient_y = derivatives[derivatives_ind + gradient_y_idx];
            if (dy < 0)
            {
                gradient_y *= -1;
            }

            for (int a = amin; a <= amax; ++a)
            {
                float da = center_a - a;
                float dx = dx_cb + grid_to_cartesian[0] * da;
                float d2_zyx = d2_zy + dx * dx;

                if (d2_zyx <= rmax2)
                {
                    int gradient_x_idx = NEAREST_INT(fabsf(dx) / rstep);
                    gradient_x_idx = min(gradient_x_idx, N_radial_points - 1);
                    float gradient_x = derivatives[derivatives_ind + gradient_x_idx];
                    if (dx < 0)
                    {
                        gradient_x *= -1;
                    }

                    float r = sqrtf(d2_zyx);

                    int index = NEAREST_INT(r / rstep);
                    index = min(index, N_radial_points - 1);

                    float density_value = radial_densities[b_idx * N_atoms * N_radial_points + atom_i * N_radial_points + index];

                    // Get upstream gradient from output
                    int out_index = ind_cb + modulo(a, Dx);
                    float dL_dRho = grad_out[b_idx * Dz * Dy * Dx + out_index];

                    // Gradient w.r.t. occupancy
                    grad_q += dL_dRho * density_value;

                    // Gradient w.r.t. radial profile points
                    atomicAdd(&grad_radial_densities[b_idx * N_atoms * N_radial_points + atom_i * N_radial_points + index],
                              dL_dRho * q);

                    // Gradient w.r.t. coordinates
                    grad_center_a += dL_dRho * q * gradient_x;
                    grad_center_b += dL_dRho * q * gradient_y;
                    grad_center_c += dL_dRho * q * gradient_z;
                }
            }
        }
    }

    atomicAdd(&grad_points[b_idx * N_atoms * 3 + atom_i * 3 + 0], grad_center_a);
    atomicAdd(&grad_points[b_idx * N_atoms * 3 + atom_i * 3 + 1], grad_center_b);
    atomicAdd(&grad_points[b_idx * N_atoms * 3 + atom_i * 3 + 2], grad_center_c);
    atomicAdd(&grad_occupancies[b_idx * N_atoms + atom_i], grad_q);
}