// atomdensity_cuda_impl.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Forward declaration of CUDA kernels (defined in atomdensity_cuda_kernels.cu)
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
    const int N_radial_points);

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
    const int N_radial_points);

// Implementation of forward pass
torch::Tensor dilate_atom_centric_forward_cuda(
    const torch::Tensor& atom_coords_grid,
    const torch::Tensor& atom_occupancies,
    const torch::Tensor& radial_profiles,
    const float r_step,
    const float rmax_cartesian,
    const torch::Tensor& lmax_grid_units,
    const torch::Tensor& grid_dims,
    const torch::Tensor& grid_to_cartesian_matrix) {
    
    auto batch_size = atom_coords_grid.size(0);
    auto N_atoms = atom_coords_grid.size(1);
    auto N_radial_points = radial_profiles.size(2);
    
    auto options = torch::TensorOptions()
        .dtype(atom_coords_grid.dtype())
        .device(atom_coords_grid.device());
    
    auto output_density_grid = torch::zeros({
        batch_size,
        grid_dims.index({0}).item<int>(), // Dz
        grid_dims.index({1}).item<int>(), // Dy
        grid_dims.index({2}).item<int>()  // Dx
    }, options);
    
    const int threads_per_block = 256;
    const int total_atoms = batch_size * N_atoms;
    const int blocks = (total_atoms + threads_per_block - 1) / threads_per_block;
    
    dilate_atom_centric_forward_kernel<<<blocks, threads_per_block>>>(
        atom_coords_grid.data_ptr<float>(),
        atom_occupancies.data_ptr<float>(),
        radial_profiles.data_ptr<float>(),
        r_step,
        rmax_cartesian,
        lmax_grid_units.data_ptr<int>(),
        grid_dims.data_ptr<int>(),
        grid_to_cartesian_matrix.data_ptr<float>(),
        output_density_grid.data_ptr<float>(),
        batch_size,
        N_atoms,
        N_radial_points
    );
    
    return output_density_grid;
}

// Implementation of backward pass
std::vector<torch::Tensor> dilate_atom_centric_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& atom_coords_grid,
    const torch::Tensor& atom_occupancies,
    const torch::Tensor& radial_profiles,
    const torch::Tensor& radial_profiles_derivatives,
    const float r_step,
    const float rmax_cartesian,
    const torch::Tensor& lmax_grid_units,
    const torch::Tensor& grid_dims,
    const torch::Tensor& grid_to_cartesian_matrix) {
    
    auto batch_size = atom_coords_grid.size(0);
    auto N_atoms = atom_coords_grid.size(1);
    auto N_radial_points = radial_profiles.size(2);
    
    auto options = torch::TensorOptions()
        .dtype(atom_coords_grid.dtype())
        .device(atom_coords_grid.device());
    
    auto grad_atom_coords_grid = torch::zeros_like(atom_coords_grid);
    auto grad_atom_occupancies = torch::zeros_like(atom_occupancies);
    auto grad_radial_profiles = torch::zeros_like(radial_profiles);
    
    const int threads_per_block = 256;
    const int total_atoms = batch_size * N_atoms;
    const int blocks = (total_atoms + threads_per_block - 1) / threads_per_block;
    
    dilate_atom_centric_backward_kernel<<<blocks, threads_per_block>>>(
        grad_output.data_ptr<float>(),
        atom_coords_grid.data_ptr<float>(),
        atom_occupancies.data_ptr<float>(),
        radial_profiles.data_ptr<float>(),
        radial_profiles_derivatives.data_ptr<float>(),
        r_step,
        rmax_cartesian,
        lmax_grid_units.data_ptr<int>(),
        grid_dims.data_ptr<int>(),
        grid_to_cartesian_matrix.data_ptr<float>(),
        grad_atom_coords_grid.data_ptr<float>(),
        grad_atom_occupancies.data_ptr<float>(),
        grad_radial_profiles.data_ptr<float>(),
        batch_size,
        N_atoms,
        N_radial_points
    );
    
    return {grad_atom_coords_grid, grad_atom_occupancies, grad_radial_profiles};
}