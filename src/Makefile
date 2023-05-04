CC = g++
NVCC = nvcc

SYSNAME = LINUX
DFLAG =
CFLAGS = $(DFLAG) -O2 -fopenmp
LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart

NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc

TARGET = gt

OBJDIR=objs

HDRS =  cmatrix.h gnmgame.h nfgame.h ipa.h gnm.h enum.h
SRCS =  cmatrix.cc gnmgame.cc nfgame.cc makegame.cc ipa.cc gnm.cc gt.cc enum.cc
OBJS = $(SRCS:.cc=.o) enummix2.o
PROGS = gt

default : $(TARGET) 

gt : $(OBJS)
	$(CC) -D$(SYSNAME) $(OBJS) $(CFLAGS) $(LDFLAGS) -o gt

cmatrix.o : cmatrix.h cmatrix.cc
	$(CC) -D$(SYSNAME) $(CFLAGS) -c cmatrix.cc

gnmgame.o : cmatrix.o gnmgame.h gnmgame.cc
	$(CC) -D$(SYSNAME) $(CFLAGS) -c gnmgame.cc

nfgame.o : gnmgame.o nfgame.h nfgame.cc
	$(CC) -D$(SYSNAME) $(CFLAGS) -c nfgame.cc

ipa.o : nfgame.o ipa.cc ipa.h
	$(CC) -D$(SYSNAME) $(CFLAGS) -c ipa.cc

gnm.o : nfgame.o gnm.cc gnm.h
	$(CC) -D$(SYSNAME) $(CFLAGS) -c gnm.cc

makegame.o : nfgame.o gnmgame.o makegame.cc makegame.h
	$(CC) -D$(SYSNAME) $(CFLAGS) -c makegame.cc

gt.o : gt.cc gnm.o ipa.o makegame.o
	$(CC) -D$(SYSNAME) $(CFLAGS) -c gt.cc

enum.o : enum.cc cmatrix.o gnm.o gnmgame.o
	$(CC) -D$(SYSNAME) $(CFLAGS) -c enum.cc

%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@

clean :
	@echo "Removing object files..."
	/bin/rm -f *.o a.out core $(PROGS)
