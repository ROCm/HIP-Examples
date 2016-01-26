#include "track_ellipse.h"


void ellipsetrack(avi_t *video, double *xc0, double *yc0, int Nc, int R, int Np, int Nf) {
	/*
	% ELLIPSETRACK tracks cells in the movie specified by 'video', at
	%  locations 'xc0'/'yc0' with radii R using an ellipse with Np discrete
	%  points, starting at frame number one and stopping at frame number 'Nf'.
	%
	% INPUTS:
	%   video.......pointer to avi video object
	%   xc0,yc0.....initial center location (Nc entries)
	%   Nc..........number of cells
	%   R...........initial radius
	%   Np..........nbr of snaxels points per snake
	%   Nf..........nbr of frames in which to track
	%
	% Matlab code written by: DREW GILLIAM (based on code by GANG DONG /
	%                                                        NILANJAN RAY)
	% Ported to C by: MICHAEL BOYER
	*/
	
	int i, j;
	
	// Compute angle parameter
	double *t = (double *) malloc(sizeof(double) * Np);
	double increment = (2.0 * PI) / (double) Np;
	for (i = 0; i < Np; i++) {
		t[i] =  increment * (double) i ;
	}

	// Allocate space for a snake for each cell in each frame
	double **xc = alloc_2d_double(Nc, Nf + 1);
	double **yc = alloc_2d_double(Nc, Nf + 1);
	double ***r = alloc_3d_double(Nc, Np, Nf + 1);
	double ***x = alloc_3d_double(Nc, Np, Nf + 1);
	double ***y = alloc_3d_double(Nc, Np, Nf + 1);
	
	// Save the first snake for each cell
	for (i = 0; i < Nc; i++) {
		xc[i][0] = xc0[i];
		yc[i][0] = yc0[i];
		for (j = 0; j < Np; j++) {
			r[i][j][0] = (double) R;
		}
	}
	
	// Generate ellipse points for each cell
	for (i = 0; i < Nc; i++) {
		for (j = 0; j < Np; j++) {
			x[i][j][0] = xc[i][0] + (r[i][j][0] * cos(t[j]));
			y[i][j][0] = yc[i][0] + (r[i][j][0] * sin(t[j]));
		}
	}
	
	// Keep track of the total time spent on computing
	//  the MGVF matrix and evolving the snakes
	long long  MGVF_time = 0;
	long long snake_time = 0;
	
	
	// Process each frame
	int frame_num, cell_num;
	for (frame_num = 1; frame_num <= Nf; frame_num++) {	 
		printf("\rProcessing frame %d / %d", frame_num, Nf);
		fflush(stdout);
		
		// Get the current video frame and its dimensions
		MAT *I = get_frame(video, frame_num, 0, 1);
		int Ih = I->m;
		int Iw = I->n;
	    
	    // Set the current positions equal to the previous positions		
		for (i = 0; i < Nc; i++) {
			xc[i][frame_num] = xc[i][frame_num - 1];
			yc[i][frame_num] = yc[i][frame_num - 1];
			for (j = 0; j < Np; j++) {
				r[i][j][frame_num] = r[i][j][frame_num - 1];
			}
		}
		
		// Split the work among multiple threads, if OPEN is defined
		#ifdef OPEN
		#pragma omp parallel for num_threads(omp_num_threads) private(i, j)
		#endif
		// Track each cell
		for (cell_num = 0; cell_num < Nc; cell_num++) {
			// Make copies of the current cell's location
			double xci = xc[cell_num][frame_num];
			double yci = yc[cell_num][frame_num];
			double *ri = (double *) malloc(sizeof(double) * Np);
			for (j = 0; j < Np; j++) {
				ri[j] = r[cell_num][j][frame_num];
			}
			
			// Add up the last ten y-values for this cell
			//  (or fewer if there are not yet ten previous frames)
			double ycavg = 0.0;
			for (i = (frame_num > 10 ? frame_num - 10 : 0); i < frame_num; i++) {
				ycavg += yc[cell_num][i];
			}
			// Compute the average of the last ten y-values
			//  (this represents the expected y-location of the cell)
			ycavg = ycavg / (double) (frame_num > 10 ? 10 : frame_num);
			
			// Determine the range of the subimage surrounding the current position
			int u1 = max(xci - 4.0 * R + 0.5, 0 );
			int u2 = min(xci + 4.0 * R + 0.5, Iw - 1);
			int v1 = max(yci - 2.0 * R + 1.5, 0 );    
			int v2 = min(yci + 2.0 * R + 1.5, Ih - 1);
			
			// Extract the subimage
			MAT *Isub = m_get(v2 - v1 + 1, u2 - u1 + 1);
			for (i = v1; i <= v2; i++) {
				for (j = u1; j <= u2; j++) {
					m_set_val(Isub, i - v1, j - u1, m_get_val(I, i, j));
				}
			}
			
	        // Compute the subimage gradient magnitude			
			MAT *Ix = gradient_x(Isub);
			MAT *Iy = gradient_y(Isub);
			MAT *IE = m_get(Isub->m, Isub->n);
			for (i = 0; i < Isub->m; i++) {
				for (j = 0; j < Isub->n; j++) {
					double temp_x = m_get_val(Ix, i, j);
					double temp_y = m_get_val(Iy, i, j);
					m_set_val(IE, i, j, sqrt((temp_x * temp_x) + (temp_y * temp_y)));
				}
			}
			
			// Compute the motion gradient vector flow (MGVF) edgemaps
			long long MGVF_start_time = get_time();
			MAT *IMGVF = MGVF(IE, 1, 1);
			MGVF_time += get_time() - MGVF_start_time;
			
			// Determine the position of the cell in the subimage			
			xci = xci - (double) u1;
			yci = yci - (double) (v1 - 1);
			ycavg = ycavg - (double) (v1 - 1);
			
			// Evolve the snake
			long long snake_start_time = get_time();
			ellipseevolve(IMGVF, &xci, &yci, ri, t, Np, (double) R, ycavg);
			snake_time += get_time() - snake_start_time;
			
			// Compute the cell's new position in the full image
			xci = xci + u1;
			yci = yci + (v1 - 1);
			
			// Store the new location of the cell and the snake
			xc[cell_num][frame_num] = xci;
			yc[cell_num][frame_num] = yci;
			for (j = 0; j < Np; j++) {
				r[cell_num][j][frame_num] = ri[j];
				x[cell_num][j][frame_num] = xc[cell_num][frame_num] + (ri[j] * cos(t[j]));
				y[cell_num][j][frame_num] = yc[cell_num][frame_num] + (ri[j] * sin(t[j]));
			}
			
			// Output the updated center of each cell
			//printf("%d,%f,%f\n", cell_num, xci[cell_num], yci[cell_num]);
			
			// Free temporary memory
			m_free(IMGVF);
			free(ri);
	    }

#ifdef OUTPUT
		if (frame_num == Nf)
		  {
		    FILE * pFile;
		    pFile = fopen ("result.txt","w+");
	
		    for (cell_num = 0; cell_num < Nc; cell_num++)		
		      fprintf(pFile,"\n%d,%f,%f", cell_num, xc[cell_num][Nf], yc[cell_num][Nf]);

		    fclose (pFile);
		  }
		
#endif
	
	
		// Output a new line to visually distinguish the output from different frames
		//printf("\n");
	}
	
	// Free temporary memory
	free(t);
	free_2d_double(xc);
	free_2d_double(yc);
	free_3d_double(r);
	free_3d_double(x);
	free_3d_double(y);
	
	// Report average processing time per frame
	printf("\n\nTracking runtime (average per frame):\n");
	printf("------------------------------------\n");
	printf("MGVF computation: %.5f seconds\n", ((float) (MGVF_time)) / (float) (1000*1000*Nf));
	printf(" Snake evolution: %.5f seconds\n", ((float) (snake_time)) / (float) (1000*1000*Nf));
}


MAT *MGVF(MAT *I, double vx, double vy) {
	/*
	% MGVF calculate the motion gradient vector flow (MGVF) 
	%  for the image 'I'
	%
	% Based on the algorithm in:
	%  Motion gradient vector flow: an external force for tracking rolling 
	%   leukocytes with shape and size constrained active contours
	%  Ray, N. and Acton, S.T.
	%  IEEE Transactions on Medical Imaging
	%  Volume: 23, Issue: 12, December 2004 
	%  Pages: 1466 - 1478
	%
	% INPUTS
	%   I...........image
	%   vx,vy.......velocity vector
	%   
	% OUTPUT
	%   IMGVF.......MGVF vector field as image
	%
	% Matlab code written by: DREW GILLIAM (based on work by GANG DONG /
	%                                                        NILANJAN RAY)
	% Ported to C by: MICHAEL BOYER
	*/

	// Constants
	double converge = 0.00001;
	double mu = 0.5;
	double epsilon = 0.0000000001;
	double lambda = 8.0 * mu + 1.0;
	// Smallest positive value expressable in double-precision
	double eps = pow(2.0, -52.0);
	// Maximum number of iterations to compute the MGVF matrix
	int iterations = 500;
	
	// Find the maximum and minimum values in I
	int m = I->m, n = I->n, i, j;
	double Imax = m_get_val(I, 0, 0);
	double Imin = m_get_val(I, 0, 0);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			double temp = m_get_val(I, i, j);
			if (temp > Imax) Imax = temp;
			else if (temp < Imin) Imin = temp;
		}
	}
	
	// Normalize the image I
	double scale = 1.0 / (Imax - Imin + eps);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			double old_val = m_get_val(I, i, j);
			m_set_val(I, i, j, (old_val - Imin) * scale);
		}
	}

	// Initialize the output matrix IMGVF with values from I
	MAT *IMGVF = m_get(m, n);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			m_set_val(IMGVF, i, j, m_get_val(I, i, j));
		}
	}
	
	// Precompute row and column indices for the
	//  neighbor difference computation below
	int *rowU = (int *) malloc(sizeof(int) * m);
	int *rowD = (int *) malloc(sizeof(int) * m);
	int *colL = (int *) malloc(sizeof(int) * n);
	int *colR = (int *) malloc(sizeof(int) * n);
	rowU[0] = 0;
	rowD[m - 1] = m - 1;
	for (i = 1; i < m; i++) {
		rowU[i] = i - 1;
		rowD[i - 1] = i;
	}
	colL[0] = 0;
	colR[n - 1] = n - 1;
	for (j = 1; j < n; j++) {
		colL[j] = j - 1;
		colR[j - 1] = j;
	}
	
	// Allocate matrices used in the while loop below
	MAT *U    = m_get(m, n), *D    = m_get(m, n), *L    = m_get(m, n), *R    = m_get(m, n);
	MAT *UR   = m_get(m, n), *DR   = m_get(m, n), *UL   = m_get(m, n), *DL   = m_get(m, n);
	MAT *UHe  = m_get(m, n), *DHe  = m_get(m, n), *LHe  = m_get(m, n), *RHe  = m_get(m, n);
	MAT *URHe = m_get(m, n), *DRHe = m_get(m, n), *ULHe = m_get(m, n), *DLHe = m_get(m, n);

	
	// Precompute constants to avoid division in the for loops below
	double mu_over_lambda = mu / lambda;
	double one_over_lambda = 1.0 / lambda;
	
	// Compute the MGVF
	int iter = 0;
	double mean_diff = 1.0;
	while ((iter < iterations) && (mean_diff > converge)) { 
	    
	    // Compute the difference between each pixel and its eight neighbors
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				double subtrahend = m_get_val(IMGVF, i, j);
				m_set_val(U, i, j, m_get_val(IMGVF, rowU[i], j) - subtrahend);
				m_set_val(D, i, j, m_get_val(IMGVF, rowD[i], j) - subtrahend);
				m_set_val(L, i, j, m_get_val(IMGVF, i, colL[j]) - subtrahend);
				m_set_val(R, i, j, m_get_val(IMGVF, i, colR[j]) - subtrahend);
				m_set_val(UR, i, j, m_get_val(IMGVF, rowU[i], colR[j]) - subtrahend);
				m_set_val(DR, i, j, m_get_val(IMGVF, rowD[i], colR[j]) - subtrahend);
				m_set_val(UL, i, j, m_get_val(IMGVF, rowU[i], colL[j]) - subtrahend);
				m_set_val(DL, i, j, m_get_val(IMGVF, rowD[i], colL[j]) - subtrahend);
			}
		}
		
	    // Compute the regularized heaviside version of the matrices above
		heaviside( UHe,  U, -vy,      epsilon);
		heaviside( DHe,  D,  vy,      epsilon);
		heaviside( LHe,  L, -vx,      epsilon);
		heaviside( RHe,  R,  vx,      epsilon);
		heaviside(URHe, UR,  vx - vy, epsilon);
		heaviside(DRHe, DR,  vx + vy, epsilon);
		heaviside(ULHe, UL, -vx - vy, epsilon);
		heaviside(DLHe, DL,  vy - vx, epsilon);
		
		// Update the IMGVF matrix
		double total_diff = 0.0;
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				// Store the old value so we can compute the difference later
				double old_val = m_get_val(IMGVF, i, j);
				
				// Compute IMGVF += (mu / lambda)(UHe .*U  + DHe .*D  + LHe .*L  + RHe .*R +
				//                                URHe.*UR + DRHe.*DR + ULHe.*UL + DLHe.*DL);
				double vU  = m_get_val(UHe,  i, j) * m_get_val(U,  i, j);
				double vD  = m_get_val(DHe,  i, j) * m_get_val(D,  i, j);
				double vL  = m_get_val(LHe,  i, j) * m_get_val(L,  i, j);
				double vR  = m_get_val(RHe,  i, j) * m_get_val(R,  i, j);
				double vUR = m_get_val(URHe, i, j) * m_get_val(UR, i, j);
				double vDR = m_get_val(DRHe, i, j) * m_get_val(DR, i, j);
				double vUL = m_get_val(ULHe, i, j) * m_get_val(UL, i, j);
				double vDL = m_get_val(DLHe, i, j) * m_get_val(DL, i, j);				
				double vHe = old_val + mu_over_lambda * (vU + vD + vL + vR + vUR + vDR + vUL + vDL);
				
				// Compute IMGVF -= (1 / lambda)(I .* (IMGVF - I))
				double vI = m_get_val(I, i, j);
				double new_val = vHe - (one_over_lambda * vI * (vHe - vI));
				m_set_val(IMGVF, i, j, new_val);
				
				// Keep track of the absolute value of the differences
				//  between this iteration and the previous one
				total_diff += fabs(new_val - old_val);
			}
		}
		
		// Compute the mean absolute difference between this iteration
		//  and the previous one to check for convergence
		mean_diff = total_diff / (double) (m * n);
	    
		iter++;
	}
	
	// Free memory
	free(rowU); free(rowD); free(colL); free(colR);
	m_free(U);    m_free(D);    m_free(L);    m_free(R);
	m_free(UR);   m_free(DR);   m_free(UL);   m_free(DL);
	m_free(UHe);  m_free(DHe);  m_free(LHe);  m_free(RHe);
	m_free(URHe); m_free(DRHe); m_free(ULHe); m_free(DLHe);

	return IMGVF;
}


// Regularized version of the Heaviside step function,
//  parameterized by a small positive number 'e'
void heaviside(MAT *H, MAT *z, double v, double e) {
	int m = z->m, n = z->n, i, j;
	
	// Precompute constants to avoid division in the for loops below
	double one_over_pi = 1.0 / PI;
	double one_over_e = 1.0 / e;
	
	// Compute H = (1 / pi) * atan((z * v) / e) + 0.5
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			double z_val = m_get_val(z, i, j) * v;
			double H_val = one_over_pi * atan(z_val * one_over_e) + 0.5;
			m_set_val(H, i, j, H_val);
		}
	}
	
	// A simpler, faster approximation of the Heaviside function
	/* for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			double z_val = m_get_val(z, i, j) * v;
			double H_val = 0.5;
			if (z_val < -0.0001) H_val = 0.0;
			else if (z_val > 0.0001) H_val = 1.0;
			m_set_val(H, i, j, H_val);
		}
	} */
}


void ellipseevolve(MAT *f, double *xc0, double *yc0, double *r0, double *t, int Np, double Er, double Ey) {
	/*
	% ELLIPSEEVOLVE evolves a parametric snake according
	%  to some energy constraints.
	%
	% INPUTS:
	%   f............potential surface
	%   xc0,yc0......initial center position
	%   r0,t.........initial radii & angle vectors (with Np elements each)
	%   Np...........number of snaxel points per snake
	%   Er...........expected radius
	%   Ey...........expected y position
	%
	% OUTPUTS
	%   xc0,yc0.......final center position
	%   r0...........final radii
	%
	% Matlab code written by: DREW GILLIAM (based on work by GANG DONG /
	%                                                        NILANJAN RAY)
	% Ported to C by: MICHAEL BOYER
	*/
	
	
	// Constants
	double deltax = 0.2;
	double deltay = 0.2;
	double deltar = 0.2; 
	double converge = 0.1;
	double lambdaedge = 1;
	double lambdasize = 0.2;
	double lambdapath = 0.05;
	int iterations = 1000;      // maximum number of iterations

	int i, j;

	// Initialize variables
	double xc = *xc0;
	double yc = *yc0;
	double *r = (double *) malloc(sizeof(double) * Np);
	for (i = 0; i < Np; i++) r[i] = r0[i];
	
	// Compute the x- and y-gradients of the MGVF matrix
	MAT *fx = gradient_x(f);
	MAT *fy = gradient_y(f);
	
	// Normalize the gradients
	int fh = f->m, fw = f->n;
	for (i = 0; i < fh; i++) {
		for (j = 0; j < fw; j++) {
			double temp_x = m_get_val(fx, i, j);
			double temp_y = m_get_val(fy, i, j);
			double fmag = sqrt((temp_x * temp_x) + (temp_y * temp_y));
			m_set_val(fx, i, j, temp_x / fmag);
			m_set_val(fy, i, j, temp_y / fmag);
		}
	}
	
	double *r_old = (double *) malloc(sizeof(double) * Np);
	VEC *x = v_get(Np);
	VEC *y = v_get(Np);
	
	
	// Evolve the snake
	int iter = 0;
	double snakediff = 1.0;
	while (iter < iterations && snakediff > converge) {
		
		// Save the values from the previous iteration
		double xc_old = xc, yc_old = yc;
		for (i = 0; i < Np; i++) {
			r_old[i] = r[i];
		}
		
		// Compute the locations of the snaxels
		for (i = 0; i < Np; i++) {
			v_set_val(x, i, xc + r[i] * cos(t[i]));
			v_set_val(y, i, yc + r[i] * sin(t[i]));
		}
		
		// See if any of the points in the snake are off the edge of the image
		double min_x = v_get_val(x, 0), max_x = v_get_val(x, 0);
		double min_y = v_get_val(y, 0), max_y = v_get_val(y, 0);
		for (i = 1; i < Np; i++) {
			double x_i = v_get_val(x, i);
			if (x_i < min_x) min_x = x_i;
			else if (x_i > max_x) max_x = x_i;
			double y_i = v_get_val(y, i);
			if (y_i < min_y) min_y = y_i;
			else if (y_i > max_y) max_y = y_i;
		}
		if (min_x < 0.0 || max_x > (double) fw - 1.0 || min_y < 0 || max_y > (double) fh - 1.0) break;
		
		
		// Compute the length of the snake		
		double L = 0.0;
		for (i = 0; i < Np - 1; i++) {
			double diff_x = v_get_val(x, i + 1) - v_get_val(x, i);
			double diff_y = v_get_val(y, i + 1) - v_get_val(y, i);
			L += sqrt((diff_x * diff_x) + (diff_y * diff_y));
		}
		double diff_x = v_get_val(x, 0) - v_get_val(x, Np - 1);
		double diff_y = v_get_val(y, 0) - v_get_val(y, Np - 1);
		L += sqrt((diff_x * diff_x) + (diff_y * diff_y));
		
		// Compute the potential surface at each snaxel
		MAT *vf  = linear_interp2(f,  x, y);
		MAT *vfx = linear_interp2(fx, x, y);
		MAT *vfy = linear_interp2(fy, x, y);
		
		// Compute the average potential surface around the snake
		double vfmean  = sum_m(vf ) / L;
		double vfxmean = sum_m(vfx) / L;
		double vfymean = sum_m(vfy) / L;
		
		// Compute the radial potential surface		
		int m = vf->m, n = vf->n;
		MAT *vfr = m_get(m, n);
		for (i = 0; i < n; i++) {
			double vf_val  = m_get_val(vf,  0, i);
			double vfx_val = m_get_val(vfx, 0, i);
			double vfy_val = m_get_val(vfy, 0, i);
			double x_val = v_get_val(x, i);
			double y_val = v_get_val(y, i);
			double new_val = (vf_val + vfx_val * (x_val - xc) + vfy_val * (y_val - yc) - vfmean) / L;
			m_set_val(vfr, 0, i, new_val);
		}		
		
		// Update the snake center and snaxels
		xc =  xc + (deltax * lambdaedge * vfxmean);
		yc = (yc + (deltay * lambdaedge * vfymean) + (deltay * lambdapath * Ey)) / (1.0 + deltay * lambdapath);
		double r_diff = 0.0;
		for (i = 0; i < Np; i++) {
			r[i] = (r[i] + (deltar * lambdaedge * m_get_val(vfr, 0, i)) + (deltar * lambdasize * Er)) /
			       (1.0 + deltar * lambdasize);
			r_diff += fabs(r[i] - r_old[i]);
		}
		
		// Test for convergence
		snakediff = fabs(xc - xc_old) + fabs(yc - yc_old) + r_diff;
		
		// Free temporary matrices
		m_free(vf);
		m_free(vfx);
		m_free(vfy);
		m_free(vfr);
	    
		iter++;
	}
	
	// Set the return values
	*xc0 = xc;
	*yc0 = yc;
	for (i = 0; i < Np; i++)
		r0[i] = r[i];
	
	// Free memory
	free(r); free(r_old);
	v_free( x); v_free( y);
	m_free(fx); m_free(fy);
}
	

// Returns the sum of all of the elements in the specified matrix
double sum_m(MAT *matrix) {
	if (matrix == NULL) return 0.0;	
	
	int i, j;
	double sum = 0.0;
	for (i = 0; i < matrix->m; i++)
		for (j = 0; j < matrix->n; j++)
			sum += m_get_val(matrix, i, j);
	
	return sum;
}


// Returns the sum of all of the elements in the specified vector
double sum_v(VEC *vector) {
	if (vector == NULL) return 0.0;	
	
	int i;
	double sum = 0.0;
	for (i = 0; i < vector->dim; i++)
		sum += v_get_val(vector, i);
	
	return sum;
}


// Creates a zeroed x-by-y matrix of doubles
double **alloc_2d_double(int x, int y) {
	if (x < 1 || y < 1) return NULL;
	
	// Allocate the data and the pointers to the data
	double *data = (double *) calloc(x * y, sizeof(double));
	double **pointers = (double **) malloc(sizeof(double *) * x);
	
	// Make the pointers point to the data
	int i;
	for (i = 0; i < x; i++) {
		pointers[i] = data + (i * y);
	}
	
	return pointers;
}


// Creates a zeroed x-by-y-by-z matrix of doubles
double ***alloc_3d_double(int x, int y, int z) {
	if (x < 1 || y < 1 || z < 1) return NULL;
	
	// Allocate the data and the two levels of pointers
	double *data = (double *) calloc(x * y * z, sizeof(double));
	double **pointers_to_data = (double **) malloc(sizeof(double *) * x * y);
	double ***pointers_to_pointers = (double ***) malloc(sizeof(double **) * x);
	
	// Make the pointers point to the data
	int i;
	for (i = 0; i < x * y; i++) pointers_to_data[i] = data + (i * z);
	for (i = 0; i < x; i++) pointers_to_pointers[i] = pointers_to_data + (i * y);
	
	return pointers_to_pointers;
}


// Frees a 2d matrix generated by the alloc_2d_double function
void free_2d_double(double **p) {
	if (p != NULL) {
		if (p[0] != NULL) free(p[0]);
		free(p);
	}
}


// Frees a 3d matrix generated by the alloc_3d_double function
void free_3d_double(double ***p) {
	if (p != NULL) {
		if (p[0] != NULL) {
			if (p[0][0] != NULL) free(p[0][0]);
			free(p[0]);
		}
		free(p);
	}
}
