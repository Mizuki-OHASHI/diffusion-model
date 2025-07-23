CC = gcc
CFLAGS = -O3

default: sweepbeta sampling

all: sweepbeta sampling

xymetoropolis.o: xymetoropolis.c xymetoropolis.h
	$(CC) $(CFLAGS) -c xymetoropolis.c

sweepbeta: sweepbeta.o xymetoropolis.o
	$(CC) $(CFLAGS) -o sweepbeta sweepbeta.o xymetoropolis.o -lm

sweepbeta.o: sweepbeta.c xymetoropolis.h
	$(CC) $(CFLAGS) -c sweepbeta.c

sampling: sampling.o xymetoropolis.o
	$(CC) $(CFLAGS) -o sampling sampling.o xymetoropolis.o -lm

sampling.o: sampling.c xymetoropolis.h
	$(CC) $(CFLAGS) -c sampling.c

clean:
	rm -f xymetoropolis.o sweepbeta.o sweepbeta sampling.o sampling
