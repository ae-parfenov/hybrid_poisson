gpu_solver:
	nvcc -O3 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpi_ibm solver_gpu.cu -o solver.gpu

hyb_solver:
	mpixlC -O5 -qsmp=omp solver_hyb.cpp -o solver.hyb
