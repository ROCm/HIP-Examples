// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

#include <cutil.h>
#include <iostream>
#include <fstream>

 
 
/*
 * Options 
 * 
 */ 
#define GAMMA 1.4
#define iterations 2000
#ifndef block_length
	#define block_length 192
#endif

#define NDIM 3
#define NNB 4

#define RK 3	// 3rd order RK
#define ff_mach 1.2
#define deg_angle_of_attack 0.0f

/*
 * not options
 */


#if block_length > 128
#warning "the kernels may fail too launch on some systems if the block length is too large"
#endif


#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)


/*
 * Generic functions
 */
template <typename T>
T* alloc(int N)
{
	T* t;
	CUDA_SAFE_CALL(cudaMalloc((void**)&t, sizeof(T)*N));
	return t;
}

template <typename T>
void dealloc(T* array)
{
	CUDA_SAFE_CALL(cudaFree((void*)array));
}

template <typename T>
void copy(T* dst, T* src, int N)
{
	CUDA_SAFE_CALL(cudaMemcpy((void*)dst, (void*)src, N*sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T>
void upload(T* dst, T* src, int N)
{
	CUDA_SAFE_CALL(cudaMemcpy((void*)dst, (void*)src, N*sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void download(T* dst, T* src, int N)
{
	CUDA_SAFE_CALL(cudaMemcpy((void*)dst, (void*)src, N*sizeof(T), cudaMemcpyDeviceToHost));
}

void dump(float* variables, int nel, int nelr)
{
	float* h_variables = new float[nelr*NVAR];
	download(h_variables, variables, nelr*NVAR);

	{
		std::ofstream file("density");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY*nelr] << std::endl;
	}


	{
		std::ofstream file("momentum");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++)
		{
			for(int j = 0; j != NDIM; j++)
				file << h_variables[i + (VAR_MOMENTUM+j)*nelr] << " ";
			file << std::endl;
		}
	}
	
	{
		std::ofstream file("density_energy");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY_ENERGY*nelr] << std::endl;
	}
	delete[] h_variables;
}

/*
 * Element-based Cell-centered FVM solver functions
 */
__constant__ float ff_variable[NVAR];
__constant__ float3 ff_fc_momentum_x[1];
__constant__ float3 ff_fc_momentum_y[1];
__constant__ float3 ff_fc_momentum_z[1];
__constant__ float3 ff_fc_density_energy[1];

__global__ void cuda_initialize_variables(int nelr, float* variables)
{
	const int i = (blockDim.x*blockIdx.x + threadIdx.x);
	for(int j = 0; j < NVAR; j++)
		variables[i + j*nelr] = ff_variable[j];
}
void initialize_variables(int nelr, float* variables)
{
	dim3 Dg(nelr / block_length), Db(block_length);
	cuda_initialize_variables<<<Dg, Db>>>(nelr, variables);
	CUT_CHECK_ERROR("initialize_variables failed");
}

__device__ __host__ inline void compute_flux_contribution(float& density, float3& momentum, float& density_energy, float& pressure, float3& velocity, float3& fc_momentum_x, float3& fc_momentum_y, float3& fc_momentum_z, float3& fc_density_energy)
{
	fc_momentum_x.x = velocity.x*momentum.x + pressure;
	fc_momentum_x.y = velocity.x*momentum.y;
	fc_momentum_x.z = velocity.x*momentum.z;
	
	
	fc_momentum_y.x = fc_momentum_x.y;
	fc_momentum_y.y = velocity.y*momentum.y + pressure;
	fc_momentum_y.z = velocity.y*momentum.z;

	fc_momentum_z.x = fc_momentum_x.z;
	fc_momentum_z.y = fc_momentum_y.z;
	fc_momentum_z.z = velocity.z*momentum.z + pressure;

	float de_p = density_energy+pressure;
	fc_density_energy.x = velocity.x*de_p;
	fc_density_energy.y = velocity.y*de_p;
	fc_density_energy.z = velocity.z*de_p;
}

__device__ inline void compute_velocity(float& density, float3& momentum, float3& velocity)
{
	velocity.x = momentum.x / density;
	velocity.y = momentum.y / density;
	velocity.z = momentum.z / density;
}
	
__device__ inline float compute_speed_sqd(float3& velocity)
{
	return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}

__device__ inline float compute_pressure(float& density, float& density_energy, float& speed_sqd)
{
	return (float(GAMMA)-float(1.0f))*(density_energy - float(0.5f)*density*speed_sqd);
}

__device__ inline float compute_speed_of_sound(float& density, float& pressure)
{
	return sqrtf(float(GAMMA)*pressure/density);
}

__global__ void cuda_compute_step_factor(int nelr, float* variables, float* areas, float* step_factors)
{
	const int i = (blockDim.x*blockIdx.x + threadIdx.x);

	float density = variables[i + VAR_DENSITY*nelr];
	float3 momentum;
	momentum.x = variables[i + (VAR_MOMENTUM+0)*nelr];
	momentum.y = variables[i + (VAR_MOMENTUM+1)*nelr];
	momentum.z = variables[i + (VAR_MOMENTUM+2)*nelr];
	
	float density_energy = variables[i + VAR_DENSITY_ENERGY*nelr];
	
	float3 velocity;       compute_velocity(density, momentum, velocity);
	float speed_sqd      = compute_speed_sqd(velocity);
	float pressure       = compute_pressure(density, density_energy, speed_sqd);
	float speed_of_sound = compute_speed_of_sound(density, pressure);

	// dt = float(0.5f) * sqrtf(areas[i]) /  (||v|| + c).... but when we do time stepping, this later would need to be divided by the area, so we just do it all at once
	step_factors[i] = float(0.5f) / (sqrtf(areas[i]) * (sqrtf(speed_sqd) + speed_of_sound));
}
void compute_step_factor(int nelr, float* variables, float* areas, float* step_factors)
{
	dim3 Dg(nelr / block_length), Db(block_length);
	cuda_compute_step_factor<<<Dg, Db>>>(nelr, variables, areas, step_factors);		
	CUT_CHECK_ERROR("compute_step_factor failed");
}

__global__ void cuda_compute_flux_contributions(int nelr, float* variables, float* fc_momentum_x, float* fc_momentum_y, float* fc_momentum_z, float* fc_density_energy)
{
	const int i = (blockDim.x*blockIdx.x + threadIdx.x);

	float density_i = variables[i + VAR_DENSITY*nelr];
	float3 momentum_i;
	momentum_i.x = variables[i + (VAR_MOMENTUM+0)*nelr];
	momentum_i.y = variables[i + (VAR_MOMENTUM+1)*nelr];
	momentum_i.z = variables[i + (VAR_MOMENTUM+2)*nelr];
	float density_energy_i = variables[i + VAR_DENSITY_ENERGY*nelr];

	float3 velocity_i;             				compute_velocity(density_i, momentum_i, velocity_i);
	float speed_sqd_i                          = compute_speed_sqd(velocity_i);
	float speed_i                              = sqrtf(speed_sqd_i);
	float pressure_i                           = compute_pressure(density_i, density_energy_i, speed_sqd_i);
	float speed_of_sound_i                     = compute_speed_of_sound(density_i, pressure_i);
	float3 fc_i_momentum_x, fc_i_momentum_y, fc_i_momentum_z;
	float3 fc_i_density_energy;	
	compute_flux_contribution(density_i, momentum_i, density_energy_i, pressure_i, velocity_i, fc_i_momentum_x, fc_i_momentum_y, fc_i_momentum_z, fc_i_density_energy);

	fc_momentum_x[i + 0*nelr] = fc_i_momentum_x.x;
	fc_momentum_x[i + 1*nelr] = fc_i_momentum_x.y;
	fc_momentum_x[i + 2*nelr] = fc_i_momentum_x.z;

	fc_momentum_y[i + 0*nelr] = fc_i_momentum_y.x;
	fc_momentum_y[i + 1*nelr] = fc_i_momentum_y.y;
	fc_momentum_y[i + 2*nelr] = fc_i_momentum_y.z;


	fc_momentum_z[i + 0*nelr] = fc_i_momentum_z.x;
	fc_momentum_z[i + 1*nelr] = fc_i_momentum_z.y;
	fc_momentum_z[i + 2*nelr] = fc_i_momentum_z.z;

	fc_density_energy[i + 0*nelr] = fc_i_density_energy.x;
	fc_density_energy[i + 1*nelr] = fc_i_density_energy.y;
	fc_density_energy[i + 2*nelr] = fc_i_density_energy.z;

}
void compute_flux_contributions(int nelr, float* variables, float* fc_momentum_x, float* fc_momentum_y, float* fc_momentum_z, float* fc_density_energy)
{
	dim3 Dg(nelr / block_length), Db(block_length);
	cuda_compute_flux_contributions<<<Dg,Db>>>(nelr, variables, fc_momentum_x, fc_momentum_y, fc_momentum_z, fc_density_energy);
	CUT_CHECK_ERROR("compute_flux_contributions failed");
}


__global__ void cuda_compute_flux(int nelr, int* elements_surrounding_elements, float* normals, float* variables, float* fc_momentum_x, float* fc_momentum_y, float* fc_momentum_z, float* fc_density_energy, float* fluxes)
{
	const float smoothing_coefficient = float(0.2f);
	const int i = (blockDim.x*blockIdx.x + threadIdx.x);
	
	int j, nb;
	float3 normal; float normal_len;
	float factor;
	
	float density_i = variables[i + VAR_DENSITY*nelr];
	float3 momentum_i;
	momentum_i.x = variables[i + (VAR_MOMENTUM+0)*nelr];
	momentum_i.y = variables[i + (VAR_MOMENTUM+1)*nelr];
	momentum_i.z = variables[i + (VAR_MOMENTUM+2)*nelr];

	float density_energy_i = variables[i + VAR_DENSITY_ENERGY*nelr];

	float3 velocity_i;             	           compute_velocity(density_i, momentum_i, velocity_i);
	float speed_sqd_i                          = compute_speed_sqd(velocity_i);
	float speed_i                              = sqrtf(speed_sqd_i);
	float pressure_i                           = compute_pressure(density_i, density_energy_i, speed_sqd_i);
	float speed_of_sound_i                     = compute_speed_of_sound(density_i, pressure_i);
	float3 fc_i_momentum_x, fc_i_momentum_y, fc_i_momentum_z;
	float3 fc_i_density_energy;	

	fc_i_momentum_x.x = fc_momentum_x[i + 0*nelr];
	fc_i_momentum_x.y = fc_momentum_x[i + 1*nelr];
	fc_i_momentum_x.z = fc_momentum_x[i + 2*nelr];

	fc_i_momentum_y.x = fc_momentum_y[i + 0*nelr];
	fc_i_momentum_y.y = fc_momentum_y[i + 1*nelr];
	fc_i_momentum_y.z = fc_momentum_y[i + 2*nelr];

	fc_i_momentum_z.x = fc_momentum_z[i + 0*nelr];
	fc_i_momentum_z.y = fc_momentum_z[i + 1*nelr];
	fc_i_momentum_z.z = fc_momentum_z[i + 2*nelr];

	fc_i_density_energy.x = fc_density_energy[i + 0*nelr];
	fc_i_density_energy.y = fc_density_energy[i + 1*nelr];
	fc_i_density_energy.z = fc_density_energy[i + 2*nelr];
	
	float flux_i_density = float(0.0f);
	float3 flux_i_momentum;
	flux_i_momentum.x = float(0.0f);
	flux_i_momentum.y = float(0.0f);
	flux_i_momentum.z = float(0.0f);
	float flux_i_density_energy = float(0.0f);
		
	float3 velocity_nb;
	float density_nb, density_energy_nb;
	float3 momentum_nb;
	float3 fc_nb_momentum_x, fc_nb_momentum_y, fc_nb_momentum_z;
	float3 fc_nb_density_energy;	
	float speed_sqd_nb, speed_of_sound_nb, pressure_nb;
	
	#pragma unroll
	for(j = 0; j < NNB; j++)
	{
		nb = elements_surrounding_elements[i + j*nelr];
		normal.x = normals[i + (j + 0*NNB)*nelr];
		normal.y = normals[i + (j + 1*NNB)*nelr];
		normal.z = normals[i + (j + 2*NNB)*nelr];
		normal_len = sqrtf(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
		
		if(nb >= 0) 	// a legitimate neighbor
		{
			density_nb = variables[nb + VAR_DENSITY*nelr];
			momentum_nb.x = variables[nb + (VAR_MOMENTUM+0)*nelr];
			momentum_nb.y = variables[nb + (VAR_MOMENTUM+1)*nelr];
			momentum_nb.z = variables[nb + (VAR_MOMENTUM+2)*nelr];
			density_energy_nb = variables[nb + VAR_DENSITY_ENERGY*nelr];
								  compute_velocity(density_nb, momentum_nb, velocity_nb);
			speed_sqd_nb                      = compute_speed_sqd(velocity_nb);
			pressure_nb                       = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
			speed_of_sound_nb                 = compute_speed_of_sound(density_nb, pressure_nb);
			                                   
			fc_nb_momentum_x.x = fc_momentum_x[nb + 0*nelr];
			fc_nb_momentum_x.y = fc_momentum_x[nb + 1*nelr];
			fc_nb_momentum_x.z = fc_momentum_x[nb + 2*nelr];

			fc_nb_momentum_y.x = fc_momentum_y[nb + 0*nelr];
			fc_nb_momentum_y.y = fc_momentum_y[nb + 1*nelr];
			fc_nb_momentum_y.z = fc_momentum_y[nb + 2*nelr];

			fc_nb_momentum_z.x = fc_momentum_z[nb + 0*nelr];
			fc_nb_momentum_z.y = fc_momentum_z[nb + 1*nelr];
			fc_nb_momentum_z.z = fc_momentum_z[nb + 2*nelr];

			fc_nb_density_energy.x = fc_density_energy[nb + 0*nelr];
			fc_nb_density_energy.y = fc_density_energy[nb + 1*nelr];
			fc_nb_density_energy.z = fc_density_energy[nb + 2*nelr];
			

			// artificial viscosity
			factor = -normal_len*smoothing_coefficient*float(0.5f)*(speed_i + sqrtf(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
			flux_i_density += factor*(density_i-density_nb);
			flux_i_density_energy += factor*(density_energy_i-density_energy_nb);
			flux_i_momentum.x += factor*(momentum_i.x-momentum_nb.x);
			flux_i_momentum.y += factor*(momentum_i.y-momentum_nb.y);
			flux_i_momentum.z += factor*(momentum_i.z-momentum_nb.z);

			// accumulate cell-centered fluxes
			factor = float(0.5f)*normal.x;
			flux_i_density += factor*(momentum_nb.x+momentum_i.x);
			flux_i_density_energy += factor*(fc_nb_density_energy.x+fc_i_density_energy.x);
			flux_i_momentum.x += factor*(fc_nb_momentum_x.x+fc_i_momentum_x.x);
			flux_i_momentum.y += factor*(fc_nb_momentum_y.x+fc_i_momentum_y.x);
			flux_i_momentum.z += factor*(fc_nb_momentum_z.x+fc_i_momentum_z.x);
			
			factor = float(0.5f)*normal.y;
			flux_i_density += factor*(momentum_nb.y+momentum_i.y);
			flux_i_density_energy += factor*(fc_nb_density_energy.y+fc_i_density_energy.y);
			flux_i_momentum.x += factor*(fc_nb_momentum_x.y+fc_i_momentum_x.y);
			flux_i_momentum.y += factor*(fc_nb_momentum_y.y+fc_i_momentum_y.y);
			flux_i_momentum.z += factor*(fc_nb_momentum_z.y+fc_i_momentum_z.y);
			
			factor = float(0.5f)*normal.z;
			flux_i_density += factor*(momentum_nb.z+momentum_i.z);
			flux_i_density_energy += factor*(fc_nb_density_energy.z+fc_i_density_energy.z);
			flux_i_momentum.x += factor*(fc_nb_momentum_x.z+fc_i_momentum_x.z);
			flux_i_momentum.y += factor*(fc_nb_momentum_y.z+fc_i_momentum_y.z);
			flux_i_momentum.z += factor*(fc_nb_momentum_z.z+fc_i_momentum_z.z);
		}
		else if(nb == -1)	// a wing boundary
		{
			flux_i_momentum.x += normal.x*pressure_i;
			flux_i_momentum.y += normal.y*pressure_i;
			flux_i_momentum.z += normal.z*pressure_i;
		}
		else if(nb == -2) // a far field boundary
		{
			factor = float(0.5f)*normal.x;
			flux_i_density += factor*(ff_variable[VAR_MOMENTUM+0]+momentum_i.x);
			flux_i_density_energy += factor*(ff_fc_density_energy[0].x+fc_i_density_energy.x);
			flux_i_momentum.x += factor*(ff_fc_momentum_x[0].x + fc_i_momentum_x.x);
			flux_i_momentum.y += factor*(ff_fc_momentum_y[0].x + fc_i_momentum_y.x);
			flux_i_momentum.z += factor*(ff_fc_momentum_z[0].x + fc_i_momentum_z.x);
			
			factor = float(0.5f)*normal.y;
			flux_i_density += factor*(ff_variable[VAR_MOMENTUM+1]+momentum_i.y);
			flux_i_density_energy += factor*(ff_fc_density_energy[0].y+fc_i_density_energy.y);
			flux_i_momentum.x += factor*(ff_fc_momentum_x[0].y + fc_i_momentum_x.y);
			flux_i_momentum.y += factor*(ff_fc_momentum_y[0].y + fc_i_momentum_y.y);
			flux_i_momentum.z += factor*(ff_fc_momentum_z[0].y + fc_i_momentum_z.y);

			factor = float(0.5f)*normal.z;
			flux_i_density += factor*(ff_variable[VAR_MOMENTUM+2]+momentum_i.z);
			flux_i_density_energy += factor*(ff_fc_density_energy[0].z+fc_i_density_energy.z);
			flux_i_momentum.x += factor*(ff_fc_momentum_x[0].z + fc_i_momentum_x.z);
			flux_i_momentum.y += factor*(ff_fc_momentum_y[0].z + fc_i_momentum_y.z);
			flux_i_momentum.z += factor*(ff_fc_momentum_z[0].z + fc_i_momentum_z.z);

		}
	}

	fluxes[i + VAR_DENSITY*nelr] = flux_i_density;
	fluxes[i + (VAR_MOMENTUM+0)*nelr] = flux_i_momentum.x;
	fluxes[i + (VAR_MOMENTUM+1)*nelr] = flux_i_momentum.y;
	fluxes[i + (VAR_MOMENTUM+2)*nelr] = flux_i_momentum.z;
	fluxes[i + VAR_DENSITY_ENERGY*nelr] = flux_i_density_energy;
}
void compute_flux(int nelr, int* elements_surrounding_elements, float* normals, float* variables, float* fc_momentum_x, float* fc_momentum_y, float* fc_momentum_z, float* fc_density_energy, float* fluxes)
{
	dim3 Dg(nelr / block_length), Db(block_length);
	cuda_compute_flux<<<Dg,Db>>>(nelr, elements_surrounding_elements, normals, variables, fc_momentum_x, fc_momentum_y, fc_momentum_z, fc_density_energy, fluxes);
	CUT_CHECK_ERROR("compute_flux failed");
}

__global__ void cuda_time_step(int j, int nelr, float* old_variables, float* variables, float* step_factors, float* fluxes)
{
	const int i = (blockDim.x*blockIdx.x + threadIdx.x);

	float factor = step_factors[i]/float(RK+1-j);

	variables[i + VAR_DENSITY*nelr] = old_variables[i + VAR_DENSITY*nelr] + factor*fluxes[i + VAR_DENSITY*nelr];
	variables[i + VAR_DENSITY_ENERGY*nelr] = old_variables[i + VAR_DENSITY_ENERGY*nelr] + factor*fluxes[i + VAR_DENSITY_ENERGY*nelr];
	variables[i + (VAR_MOMENTUM+0)*nelr] = old_variables[i + (VAR_MOMENTUM+0)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+0)*nelr];
	variables[i + (VAR_MOMENTUM+1)*nelr] = old_variables[i + (VAR_MOMENTUM+1)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+1)*nelr];	
	variables[i + (VAR_MOMENTUM+2)*nelr] = old_variables[i + (VAR_MOMENTUM+2)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+2)*nelr];	
}
void time_step(int j, int nelr, float* old_variables, float* variables, float* step_factors, float* fluxes)
{
	dim3 Dg(nelr / block_length), Db(block_length);
	cuda_time_step<<<Dg,Db>>>(j, nelr, old_variables, variables, step_factors, fluxes);
	CUT_CHECK_ERROR("update failed");
}

/*
 * Main function
 */
int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cout << "specify data file name" << std::endl;
		return 0;
	}
	const char* data_file_name = argv[1];
	
	cudaDeviceProp prop;
	int dev;
	
	CUDA_SAFE_CALL(cudaSetDevice(0));
	CUDA_SAFE_CALL(cudaGetDevice(&dev));
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, dev));
	
	printf("Name:                     %s\n", prop.name);

	// set far field conditions and load them into constant memory on the gpu
	{
		float h_ff_variable[NVAR];
		const float angle_of_attack = float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);
		
		h_ff_variable[VAR_DENSITY] = float(1.4);
		
		float ff_pressure = float(1.0f);
		float ff_speed_of_sound = sqrt(GAMMA*ff_pressure / h_ff_variable[VAR_DENSITY]);
		float ff_speed = float(ff_mach)*ff_speed_of_sound;
		
		float3 ff_velocity;
		ff_velocity.x = ff_speed*float(cos((float)angle_of_attack));
		ff_velocity.y = ff_speed*float(sin((float)angle_of_attack));
		ff_velocity.z = 0.0f;
		
		h_ff_variable[VAR_MOMENTUM+0] = h_ff_variable[VAR_DENSITY] * ff_velocity.x;
		h_ff_variable[VAR_MOMENTUM+1] = h_ff_variable[VAR_DENSITY] * ff_velocity.y;
		h_ff_variable[VAR_MOMENTUM+2] = h_ff_variable[VAR_DENSITY] * ff_velocity.z;
				
		h_ff_variable[VAR_DENSITY_ENERGY] = h_ff_variable[VAR_DENSITY]*(float(0.5f)*(ff_speed*ff_speed)) + (ff_pressure / float(GAMMA-1.0f));

		float3 h_ff_momentum;
		h_ff_momentum.x = *(h_ff_variable+VAR_MOMENTUM+0);
		h_ff_momentum.y = *(h_ff_variable+VAR_MOMENTUM+1);
		h_ff_momentum.z = *(h_ff_variable+VAR_MOMENTUM+2);
		float3 h_ff_fc_momentum_x;
		float3 h_ff_fc_momentum_y;
		float3 h_ff_fc_momentum_z;
		float3 h_ff_fc_density_energy;
		compute_flux_contribution(h_ff_variable[VAR_DENSITY], h_ff_momentum, h_ff_variable[VAR_DENSITY_ENERGY], ff_pressure, ff_velocity, h_ff_fc_momentum_x, h_ff_fc_momentum_y, h_ff_fc_momentum_z, h_ff_fc_density_energy);

		// copy far field conditions to the gpu
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(ff_variable,          h_ff_variable,          NVAR*sizeof(float)) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(ff_fc_momentum_x, &h_ff_fc_momentum_x, sizeof(float3)) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(ff_fc_momentum_y, &h_ff_fc_momentum_y, sizeof(float3)) );
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(ff_fc_momentum_z, &h_ff_fc_momentum_z, sizeof(float3)) );
		
		CUDA_SAFE_CALL( cudaMemcpyToSymbol(ff_fc_density_energy, &h_ff_fc_density_energy, sizeof(float3)) );		
	}
	int nel;
	int nelr;
	
	// read in domain geometry
	float* areas;
	int* elements_surrounding_elements;
	float* normals;
	{
		std::ifstream file(data_file_name);
	
		file >> nel;
		nelr = block_length*((nel / block_length )+ std::min(1, nel % block_length));

		float* h_areas = new float[nelr];
		int* h_elements_surrounding_elements = new int[nelr*NNB];
		float* h_normals = new float[nelr*NDIM*NNB];

				
		// read in data
		for(int i = 0; i < nel; i++)
		{
			file >> h_areas[i];
			for(int j = 0; j < NNB; j++)
			{
				file >> h_elements_surrounding_elements[i + j*nelr];
				if(h_elements_surrounding_elements[i+j*nelr] < 0) h_elements_surrounding_elements[i+j*nelr] = -1;
				h_elements_surrounding_elements[i + j*nelr]--; //it's coming in with Fortran numbering				
				
				for(int k = 0; k < NDIM; k++)
				{
					file >> h_normals[i + (j + k*NNB)*nelr];
					h_normals[i + (j + k*NNB)*nelr] = -h_normals[i + (j + k*NNB)*nelr];
				}
			}
		}
		
		// fill in remaining data
		int last = nel-1;
		for(int i = nel; i < nelr; i++)
		{
			h_areas[i] = h_areas[last];
			for(int j = 0; j < NNB; j++)
			{
				// duplicate the last element
				h_elements_surrounding_elements[i + j*nelr] = h_elements_surrounding_elements[last + j*nelr];	
				for(int k = 0; k < NDIM; k++) h_normals[last + (j + k*NNB)*nelr] = h_normals[last + (j + k*NNB)*nelr];
			}
		}
		
		areas = alloc<float>(nelr);
		upload<float>(areas, h_areas, nelr);

		elements_surrounding_elements = alloc<int>(nelr*NNB);
		upload<int>(elements_surrounding_elements, h_elements_surrounding_elements, nelr*NNB);

		normals = alloc<float>(nelr*NDIM*NNB);
		upload<float>(normals, h_normals, nelr*NDIM*NNB);
				
		delete[] h_areas;
		delete[] h_elements_surrounding_elements;
		delete[] h_normals;
	}

	// Create arrays and set initial conditions
	float* variables = alloc<float>(nelr*NVAR);
	initialize_variables(nelr, variables);

	float* old_variables = alloc<float>(nelr*NVAR);   	
	float* fluxes = alloc<float>(nelr*NVAR);
	float* step_factors = alloc<float>(nelr); 
	float* fc_momentum_x = alloc<float>(nelr*NDIM); 
	float* fc_momentum_y = alloc<float>(nelr*NDIM);
	float* fc_momentum_z = alloc<float>(nelr*NDIM);
	float* fc_density_energy = alloc<float>(nelr*NDIM);


	// make sure all memory is floatly allocated before we start timing
	initialize_variables(nelr, old_variables);
	initialize_variables(nelr, fluxes);
	cudaMemset( (void*) step_factors, 0, sizeof(float)*nelr );
	// make sure CUDA isn't still doing something before we start timing
	cudaThreadSynchronize();

	// these need to be computed the first time in order to compute time step
	std::cout << "Starting..." << std::endl;


	unsigned int timer = 0;
	CUT_SAFE_CALL( cutCreateTimer( &timer));
	CUT_SAFE_CALL( cutStartTimer( timer));

	// Begin iterations
	for(int i = 0; i < iterations; i++)
	{
		copy<float>(old_variables, variables, nelr*NVAR);
		
		// for the first iteration we compute the time step
		compute_step_factor(nelr, variables, areas, step_factors);
		
		for(int j = 0; j < RK; j++)
		{
			compute_flux_contributions(nelr, variables, fc_momentum_x, fc_momentum_y, fc_momentum_z, fc_density_energy);
			compute_flux(nelr, elements_surrounding_elements, normals, variables, fc_momentum_x, fc_momentum_y, fc_momentum_z, fc_density_energy, fluxes);
			time_step(j, nelr, old_variables, variables, step_factors, fluxes);
		}
	}

	cudaThreadSynchronize();
	CUT_SAFE_CALL( cutStopTimer(timer) );  

	std::cout  << (cutGetAverageTimerValue(timer)/1000.0)  / iterations << " seconds per iteration" << std::endl;

	std::cout << "Saving solution..." << std::endl;
	dump(variables, nel, nelr);
	std::cout << "Saved solution..." << std::endl;

	
	std::cout << "Cleaning up..." << std::endl;
	dealloc<float>(areas);
	dealloc<int>(elements_surrounding_elements);
	dealloc<float>(normals);
	
	dealloc<float>(variables);
	dealloc<float>(old_variables);
	dealloc<float>(fluxes);
	dealloc<float>(step_factors);
	dealloc<float>(fc_momentum_x); 
	dealloc<float>(fc_momentum_y);
	dealloc<float>(fc_momentum_z);
	dealloc<float>(fc_density_energy);


	std::cout << "Done..." << std::endl;

	return 0;
}
