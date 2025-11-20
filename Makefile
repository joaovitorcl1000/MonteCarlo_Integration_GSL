GSLLIBS = -lgsl -lgslcblas

all:  
	g++ -fopenmp -O3 -o mcintegration mcintegration.cpp -lm $(GSLLIBS)

run: all
	./GBWintegration
