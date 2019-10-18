CC=g++
CFLAGS=-I/CGAL-4.14.1/include
LIBS=-lCGAL -lgmp

edge_aware: edge_aware_upsampling.o
        $(CC) -o edge_aware edge_aware_upsampling.o $(CFLAGS) $(LIBS)
