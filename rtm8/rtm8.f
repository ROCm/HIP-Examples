	program rtm8
	implicit none
	integer	n, nt, nx, ny, nz
c	parameter( nt=100, nx=400, ny=400, nz=300 )
c	parameter( nt=100, nx=400, ny=100, nz=300 )
	parameter( nt=30, nx=680, ny=134, nz=450 )
	real	next_s(nx,ny,nz), current_s(nx,ny,nz)
	real	next_r(nx,ny,nz), current_r(nx,ny,nz)
	real	vsq(nx,ny,nz), image(nx,ny,nz)
	real	a(5)
	external	mysecond
	real*8		mysecond
c
	integer	t, x, y, z
	real*8	pts, t0, t1, dt, flops, pt_rate, flop_rate, speedup, memory
	real	div
c
	memory = nx*ny*nz*4*6
	pts = nt
	pts = pts*(nx-8)*(ny-8)*(nz-8)
	flops = 67.*pts
	print *, 'memory (MB) = ', memory/1e6
	print *, 'pts (billions) = ', pts/1e9
	print *, 'Tflops = ', flops/1e12
c
	a(1) = -1./560.
	a(2) = 8./315
	a(3) = -0.2
	a(4) = 1.6
	a(5) = -1435./504.
c
!$omp parallel
!$omp do
	do z = 1, nz
	do y = 1, ny
	do x = 1, nx
		vsq(x,y,z) = 1
		next_s(x,y,z) = 0
		current_s(x,y,z) = 0
		next_r(x,y,z) = 0
		current_r(x,y,z) = 0
		image(x,y,z) = 0
	enddo
	enddo
	enddo
!$omp enddo
!$omp end parallel
c
	t0 = mysecond()
	do t = 1, nt
	do z = 5, nz-4
	do y = 5, ny-4
	do x = 5, nx-4
     		div =
     &		a(1)*	current_s(x,y,z) +
     &		a(2)*(	current_s(x+1,y,z) + current_s(x-1,y,z) +
     &			current_s(x,y+1,z) + current_s(x,y-1,z) +
     &			current_s(x,y,z+1) + current_s(x,y,z-1) ) +
     &		a(3)*(	current_s(x+2,y,z) + current_s(x-2,y,z) +
     &			current_s(x,y+2,z) + current_s(x,y-2,z) +
     &			current_s(x,y,z+2) + current_s(x,y,z-2) ) +
     &		a(4)*(	current_s(x+3,y,z) + current_s(x-3,y,z) +
     &			current_s(x,y+3,z) + current_s(x,y-3,z) +
     &			current_s(x,y,z+3) + current_s(x,y,z-3) ) +
     &		a(5)*(	current_s(x+4,y,z) + current_s(x-4,y,z) +
     &			current_s(x,y+4,z) + current_s(x,y-4,z) +
     &			current_s(x,y,z+4) + current_s(x,y,z-4) )
     		next_s(x,y,z) = 2.*current_s(x,y,z)
     &				- next_s(x,y,z) + vsq(x,y,z)* div
     		div =
     &		a(1)*	current_r(x,y,z) +
     &		a(2)*(	current_r(x+1,y,z) + current_r(x-1,y,z) +
     &			current_r(x,y+1,z) + current_r(x,y-1,z) +
     &			current_r(x,y,z+1) + current_r(x,y,z-1) ) +
     &		a(3)*(	current_r(x+2,y,z) + current_r(x-2,y,z) +
     &			current_r(x,y+2,z) + current_r(x,y-2,z) +
     &			current_r(x,y,z+2) + current_r(x,y,z-2) ) +
     &		a(4)*(	current_r(x+3,y,z) + current_r(x-3,y,z) +
     &			current_r(x,y+3,z) + current_r(x,y-3,z) +
     &			current_r(x,y,z+3) + current_r(x,y,z-3) ) +
     &		a(5)*(	current_r(x+4,y,z) + current_r(x-4,y,z) +
     &			current_r(x,y+4,z) + current_r(x,y-4,z) +
     &			current_r(x,y,z+4) + current_r(x,y,z-4) )
     		next_r(x,y,z) = 2.*current_r(x,y,z)
     &				- next_r(x,y,z) + vsq(x,y,z)* div
		image(x,y,z) = next_s(x,y,z) * next_r(x,y,z)
	enddo
	enddo
	enddo
	enddo
	t1 = mysecond()
c
	dt = t1 - t0
	pt_rate = pts/dt
	flop_rate = flops/dt
	speedup = 2.*10**9/3./pt_rate
	print *, 'dt  = ', dt
	print *, 'pt_rate (millions/sec) = ', pt_rate/1e6
	print *, 'flop_rate (Gflops) = ', flop_rate/1e9
	print *, 'speedup = ', speedup
c
	stop
	end
