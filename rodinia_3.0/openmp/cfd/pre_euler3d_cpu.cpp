// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>

struct float3 { float x, y, z; };

#ifndef block_length
#error "you need to define block_length"
#endif

/*
 * Options
 *
 */
#define GAMMA 1.4
#define iterations 2000

#define NDIM 3
#define NNB 4

#define RK 3	// 3rd order RK
#define ff_mach 1.2
#define deg_angle_of_attack 0.0f

/*
 * not options
 */
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
	return new T[N];
}

template <typename T>
void dealloc(T* array)
{
	delete[] array;
}

template <typename T>
void copy(T* dst, T* src, int N)
{
	#pragma omp parallel for default(shared) schedule(static)
	for(int i = 0; i < N; i++)
	{
		dst[i] = src[i];
	}
}


void dump(float* variables, int nel, int nelr)
{


	{
		std::ofstream file("density");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++) file << variables[i*NVAR + VAR_DENSITY] << std::endl;
	}


	{
		std::ofstream file("momentum");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++)
		{
			for(int j = 0; j != NDIM; j++) file << variables[i*NVAR + (VAR_MOMENTUM+j)] << " ";
			file << std::endl;
		}
	}

	{
		std::ofstream file("density_energy");
		file << nel << " " << nelr << std::endl;
		for(int i = 0; i < nel; i++) file << variables[i*NVAR + VAR_DENSITY_ENERGY] << std::endl;
	}

}

/*
 * Element-based Cell-centered FVM solver functions
 */
float ff_variable[NVAR];
float3 ff_fc_momentum_x;
float3 ff_fc_momentum_y;
float3 ff_fc_momentum_z;
float3 ff_fc_density_energy;


void initialize_variables(int nelr, float* variables)
{
	#pragma omp parallel for default(shared) schedule(static)
	for(int i = 0; i < nelr; i++)
	{
		for(int j = 0; j < NVAR; j++) variables[i*NVAR + j] = ff_variable[j];
	}
}

inline void compute_flux_contribution(float& density, float3& momentum, float& density_energy, float& pressure, float3& velocity, float3& fc_momentum_x, float3& fc_momentum_y, float3& fc_momentum_z, float3& fc_density_energy)
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

inline void compute_velocity(float& density, float3& momentum, float3& velocity)
{
	velocity.x = momentum.x / density;
	velocity.y = momentum.y / density;
	velocity.z = momentum.z / density;
}

inline float compute_speed_sqd(float3& velocity)
{
	return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}

inline float compute_pressure(float& density, float& density_energy, float& speed_sqd)
{
	return (float(GAMMA)-float(1.0f))*(density_energy - float(0.5f)*density*speed_sqd);
}

inline float compute_speed_of_sound(float& density, float& pressure)
{
	return std::sqrt(float(GAMMA)*pressure/density);
}



void compute_step_factor(int nelr, float* variables, float* areas, float* step_factors)
{
	#pragma omp parallel for default(shared) schedule(static)
	for(int i = 0; i < nelr; i++)
	{
		float density = variables[NVAR*i + VAR_DENSITY];

		float3 momentum;
		momentum.x = variables[NVAR*i + (VAR_MOMENTUM+0)];
		momentum.y = variables[NVAR*i + (VAR_MOMENTUM+1)];
		momentum.z = variables[NVAR*i + (VAR_MOMENTUM+2)];

		float density_energy = variables[NVAR*i + VAR_DENSITY_ENERGY];
		float3 velocity;	   compute_velocity(density, momentum, velocity);
		float speed_sqd      = compute_speed_sqd(velocity);
		float pressure       = compute_pressure(density, density_energy, speed_sqd);
		float speed_of_sound = compute_speed_of_sound(density, pressure);

		// dt = float(0.5f) * std::sqrt(areas[i]) /  (||v|| + c).... but when we do time stepping, this later would need to be divided by the area, so we just do it all at once
		step_factors[i] = float(0.5f) / (std::sqrt(areas[i]) * (std::sqrt(speed_sqd) + speed_of_sound));
	}
}

void compute_flux_contributions(int nelr, float* variables, float* fc_momentum_x, float* fc_momentum_y, float* fc_momentum_z, float* fc_density_energy)
{
	#pragma omp parallel for default(shared) schedule(static)
	for(int i = 0; i < nelr; i++)
	{
		float density_i = variables[NVAR*i + VAR_DENSITY];
		float3 momentum_i;
		momentum_i.x = variables[NVAR*i + (VAR_MOMENTUM+0)];
		momentum_i.y = variables[NVAR*i + (VAR_MOMENTUM+1)];
		momentum_i.z = variables[NVAR*i + (VAR_MOMENTUM+2)];
		float density_energy_i = variables[NVAR*i + VAR_DENSITY_ENERGY];

		float3 velocity_i;             				compute_velocity(density_i, momentum_i, velocity_i);
		float speed_sqd_i                          = compute_speed_sqd(velocity_i);
		float speed_i                              = sqrtf(speed_sqd_i);
		float pressure_i                           = compute_pressure(density_i, density_energy_i, speed_sqd_i);
		float speed_of_sound_i                     = compute_speed_of_sound(density_i, pressure_i);
		float3 fc_i_momentum_x, fc_i_momentum_y, fc_i_momentum_z;
		float3 fc_i_density_energy;	
		compute_flux_contribution(density_i, momentum_i, density_energy_i, pressure_i, velocity_i, fc_i_momentum_x, fc_i_momentum_y, fc_i_momentum_z, fc_i_density_energy);

		fc_momentum_x[i*NDIM + 0] = fc_i_momentum_x.x;
		fc_momentum_x[i*NDIM + 1] = fc_i_momentum_x.y;
		fc_momentum_x[i*NDIM+  2] = fc_i_momentum_x.z;

		fc_momentum_y[i*NDIM+ 0] = fc_i_momentum_y.x;
		fc_momentum_y[i*NDIM+ 1] = fc_i_momentum_y.y;
		fc_momentum_y[i*NDIM+ 2] = fc_i_momentum_y.z;


		fc_momentum_z[i*NDIM+ 0] = fc_i_momentum_z.x;
		fc_momentum_z[i*NDIM+ 1] = fc_i_momentum_z.y;
		fc_momentum_z[i*NDIM+ 2] = fc_i_momentum_z.z;

		fc_density_energy[i*NDIM+ 0] = fc_i_density_energy.x;
		fc_density_energy[i*NDIM+ 1] = fc_i_density_energy.y;
		fc_density_energy[i*NDIM+ 2] = fc_i_density_energy.z;
	}

}

/*
 *
 *
*/

void compute_flux(int nelr, int* elements_surrounding_elements, float* normals, float* variables, float* fc_momentum_x, float* fc_momentum_y, float* fc_momentum_z, float* fc_density_energy, float* fluxes)
{
	const float smoothing_coefficient = float(0.2f);

	#pragma omp parallel for default(shared) schedule(static)
	for(int i = 0; i < nelr; i++)
	{
		int j, nb;
		float3 normal; float normal_len;
		float factor;

		float density_i = variables[NVAR*i + VAR_DENSITY];
		float3 momentum_i;
		momentum_i.x = variables[NVAR*i + (VAR_MOMENTUM+0)];
		momentum_i.y = variables[NVAR*i + (VAR_MOMENTUM+1)];
		momentum_i.z = variables[NVAR*i + (VAR_MOMENTUM+2)];
		float density_energy_i = variables[NVAR*i + VAR_DENSITY_ENERGY];

		float3 velocity_i;             				 compute_velocity(density_i, momentum_i, velocity_i);
		float speed_sqd_i                          = compute_speed_sqd(velocity_i);
		float speed_i                              = std::sqrt(speed_sqd_i);
		float pressure_i                           = compute_pressure(density_i, density_energy_i, speed_sqd_i);
		float speed_of_sound_i                     = compute_speed_of_sound(density_i, pressure_i);
		float3 fc_i_momentum_x, fc_i_momentum_y, fc_i_momentum_z;
		float3 fc_i_density_energy;

		fc_i_momentum_x.x = fc_momentum_x[i*NDIM + 0];
		fc_i_momentum_x.y = fc_momentum_x[i*NDIM + 1];
		fc_i_momentum_x.z = fc_momentum_x[i*NDIM + 2];

		fc_i_momentum_y.x = fc_momentum_y[i*NDIM + 0];
		fc_i_momentum_y.y = fc_momentum_y[i*NDIM + 1];
		fc_i_momentum_y.z = fc_momentum_y[i*NDIM + 2];

		fc_i_momentum_z.x = fc_momentum_z[i*NDIM + 0];
		fc_i_momentum_z.y = fc_momentum_z[i*NDIM + 1];
		fc_i_momentum_z.z = fc_momentum_z[i*NDIM + 2];

		fc_i_density_energy.x = fc_density_energy[i*NDIM + 0];
		fc_i_density_energy.y = fc_density_energy[i*NDIM + 1];
		fc_i_density_energy.z = fc_density_energy[i*NDIM + 2];

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

		for(j = 0; j < NNB; j++)
		{
			nb = elements_surrounding_elements[i*NNB + j];
			normal.x = normals[(i*NNB + j)*NDIM + 0];
			normal.y = normals[(i*NNB + j)*NDIM + 1];
			normal.z = normals[(i*NNB + j)*NDIM + 2];
			normal_len = std::sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);

			if(nb >= 0) 	// a legitimate neighbor
			{
				density_nb =        variables[nb*NVAR + VAR_DENSITY];
				momentum_nb.x =     variables[nb*NVAR + (VAR_MOMENTUM+0)];
				momentum_nb.y =     variables[nb*NVAR + (VAR_MOMENTUM+1)];
				momentum_nb.z =     variables[nb*NVAR + (VAR_MOMENTUM+2)];
				density_energy_nb = variables[nb*NVAR + VAR_DENSITY_ENERGY];
													compute_velocity(density_nb, momentum_nb, velocity_nb);
				speed_sqd_nb                      = compute_speed_sqd(velocity_nb);
				pressure_nb                       = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
				speed_of_sound_nb                 = compute_speed_of_sound(density_nb, pressure_nb);
				fc_nb_momentum_x.x = fc_momentum_x[nb*NDIM + 0];
				fc_nb_momentum_x.y = fc_momentum_x[nb*NDIM + 1];
				fc_nb_momentum_x.z = fc_momentum_x[nb*NDIM + 2];

				fc_nb_momentum_y.x = fc_momentum_y[nb*NDIM + 0];
				fc_nb_momentum_y.y = fc_momentum_y[nb*NDIM + 1];
				fc_nb_momentum_y.z = fc_momentum_y[nb*NDIM + 2];

				fc_nb_momentum_z.x = fc_momentum_z[nb*NDIM + 0];
				fc_nb_momentum_z.y = fc_momentum_z[nb*NDIM + 1];
				fc_nb_momentum_z.z = fc_momentum_z[nb*NDIM + 2];

				fc_nb_density_energy.x = fc_density_energy[nb*NDIM + 0];
				fc_nb_density_energy.y = fc_density_energy[nb*NDIM + 1];
				fc_nb_density_energy.z = fc_density_energy[nb*NDIM + 2];

				// artificial viscosity
				factor = -normal_len*smoothing_coefficient*float(0.5f)*(speed_i + std::sqrt(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
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
				flux_i_density_energy += factor*(ff_fc_density_energy.x+fc_i_density_energy.x);
				flux_i_momentum.x += factor*(ff_fc_momentum_x.x + fc_i_momentum_x.x);
				flux_i_momentum.y += factor*(ff_fc_momentum_y.x + fc_i_momentum_y.x);
				flux_i_momentum.z += factor*(ff_fc_momentum_z.x + fc_i_momentum_z.x);

				factor = float(0.5f)*normal.y;
				flux_i_density += factor*(ff_variable[VAR_MOMENTUM+1]+momentum_i.y);
				flux_i_density_energy += factor*(ff_fc_density_energy.y+fc_i_density_energy.y);
				flux_i_momentum.x += factor*(ff_fc_momentum_x.y + fc_i_momentum_x.y);
				flux_i_momentum.y += factor*(ff_fc_momentum_y.y + fc_i_momentum_y.y);
				flux_i_momentum.z += factor*(ff_fc_momentum_z.y + fc_i_momentum_z.y);

				factor = float(0.5f)*normal.z;
				flux_i_density += factor*(ff_variable[VAR_MOMENTUM+2]+momentum_i.z);
				flux_i_density_energy += factor*(ff_fc_density_energy.z+fc_i_density_energy.z);
				flux_i_momentum.x += factor*(ff_fc_momentum_x.z + fc_i_momentum_x.z);
				flux_i_momentum.y += factor*(ff_fc_momentum_y.z + fc_i_momentum_y.z);
				flux_i_momentum.z += factor*(ff_fc_momentum_z.z + fc_i_momentum_z.z);

			}
		}

		fluxes[i*NVAR + VAR_DENSITY] = flux_i_density;
		fluxes[i*NVAR + (VAR_MOMENTUM+0)] = flux_i_momentum.x;
		fluxes[i*NVAR + (VAR_MOMENTUM+1)] = flux_i_momentum.y;
		fluxes[i*NVAR + (VAR_MOMENTUM+2)] = flux_i_momentum.z;
		fluxes[i*NVAR + VAR_DENSITY_ENERGY] = flux_i_density_energy;
	}
}

void time_step(int j, int nelr, float* old_variables, float* variables, float* step_factors, float* fluxes)
{
	#pragma omp parallel for  default(shared) schedule(static)
	for(int i = 0; i < nelr; i++)
	{
		float factor = step_factors[i]/float(RK+1-j);

		variables[NVAR*i + VAR_DENSITY] = old_variables[NVAR*i + VAR_DENSITY] + factor*fluxes[NVAR*i + VAR_DENSITY];
		variables[NVAR*i + VAR_DENSITY_ENERGY] = old_variables[NVAR*i + VAR_DENSITY_ENERGY] + factor*fluxes[NVAR*i + VAR_DENSITY_ENERGY];
		variables[NVAR*i + (VAR_MOMENTUM+0)] = old_variables[NVAR*i + (VAR_MOMENTUM+0)] + factor*fluxes[NVAR*i + (VAR_MOMENTUM+0)];
		variables[NVAR*i + (VAR_MOMENTUM+1)] = old_variables[NVAR*i + (VAR_MOMENTUM+1)] + factor*fluxes[NVAR*i + (VAR_MOMENTUM+1)];
		variables[NVAR*i + (VAR_MOMENTUM+2)] = old_variables[NVAR*i + (VAR_MOMENTUM+2)] + factor*fluxes[NVAR*i + (VAR_MOMENTUM+2)];
	}
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

	// set far field conditions
	{
		const float angle_of_attack = float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);

		ff_variable[VAR_DENSITY] = float(1.4);

		float ff_pressure = float(1.0f);
		float ff_speed_of_sound = sqrt(GAMMA*ff_pressure / ff_variable[VAR_DENSITY]);
		float ff_speed = float(ff_mach)*ff_speed_of_sound;

		float3 ff_velocity;
		ff_velocity.x = ff_speed*float(cos((float)angle_of_attack));
		ff_velocity.y = ff_speed*float(sin((float)angle_of_attack));
		ff_velocity.z = 0.0f;

		ff_variable[VAR_MOMENTUM+0] = ff_variable[VAR_DENSITY] * ff_velocity.x;
		ff_variable[VAR_MOMENTUM+1] = ff_variable[VAR_DENSITY] * ff_velocity.y;
		ff_variable[VAR_MOMENTUM+2] = ff_variable[VAR_DENSITY] * ff_velocity.z;

		ff_variable[VAR_DENSITY_ENERGY] = ff_variable[VAR_DENSITY]*(float(0.5f)*(ff_speed*ff_speed)) + (ff_pressure / float(GAMMA-1.0f));

		float3 ff_momentum;
		ff_momentum.x = *(ff_variable+VAR_MOMENTUM+0);
		ff_momentum.y = *(ff_variable+VAR_MOMENTUM+1);
		ff_momentum.z = *(ff_variable+VAR_MOMENTUM+2);
		compute_flux_contribution(ff_variable[VAR_DENSITY], ff_momentum, ff_variable[VAR_DENSITY_ENERGY], ff_pressure, ff_velocity, ff_fc_momentum_x, ff_fc_momentum_y, ff_fc_momentum_z, ff_fc_density_energy);
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

		areas = new float[nelr];
		elements_surrounding_elements = new int[nelr*NNB];
		normals = new float[NDIM*NNB*nelr];

		// read in data
		for(int i = 0; i < nel; i++)
		{
			file >> areas[i];
			for(int j = 0; j < NNB; j++)
			{
				file >> elements_surrounding_elements[i*NNB + j];
				if(elements_surrounding_elements[i*NNB+j] < 0) elements_surrounding_elements[i*NNB+j] = -1;
				elements_surrounding_elements[i*NNB + j]--; //it's coming in with Fortran numbering

				for(int k = 0; k < NDIM; k++)
				{
					file >>  normals[(i*NNB + j)*NDIM + k];
					normals[(i*NNB + j)*NDIM + k] = -normals[(i*NNB + j)*NDIM + k];
				}
			}
		}

		// fill in remaining data
		int last = nel-1;
		for(int i = nel; i < nelr; i++)
		{
			areas[i] = areas[last];
			for(int j = 0; j < NNB; j++)
			{
				// duplicate the last element
				elements_surrounding_elements[i*NNB + j] = elements_surrounding_elements[last*NNB + j];
				for(int k = 0; k < NDIM; k++) normals[(i*NNB + j)*NDIM + k] = normals[(last*NNB + j)*NDIM + k];
			}
		}
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

	// these need to be computed the first time in order to compute time step
	std::cout << "Starting..." << std::endl;
	double start = omp_get_wtime();

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

	double end = omp_get_wtime();
	std::cout  << (end-start)  / iterations << " seconds per iteration" << std::endl;



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
