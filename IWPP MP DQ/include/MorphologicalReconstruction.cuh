#ifndef MORPHOLOGICAL_RECONSTRUCTION_HPP
#define MORPHOLOGICAL_RECONSTRUCTION_HPP

#include<stdio.h>
#include <chrono>
#include "globalQueue.cuh"

#define MEGAPIXEL_SIZE 32
#define BUFFER_SIZE 256

// Alternative Raster Anti-Raster pass
#define XX_THREADS		4
#define XY_THREADS		64
#define MAX_THREADS		256
#define WARP_SIZE 32

#define QUEUE_READ 0
#define MP_FETCH 1
#define MP_PROCESSING 2
#define MP_DUMP 3
#define QUEUE_WRITE 4

// Função atômica para interação com unsigned char
// Retorna a palavra que contém o byte, que estava na memória quando a tentativa de escrita ocorreu
unsigned int __device__ atomicCASUChar(unsigned char* address, unsigned int index, unsigned char assumed, unsigned char val);

// Retorna se algum pixel foi atualizado
bool __device__ XForwardPropagation(unsigned char mp_marker[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2], 
                                unsigned char mp_mask[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2]);

bool __device__ XBackwardPropagation(unsigned char mp_marker[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2], 
                                unsigned char mp_mask[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2]);

bool __device__ YUpwardPropagation(unsigned char mp_marker[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2], 
                                unsigned char mp_mask[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2]);

bool __device__ YDownwardPropagation(unsigned char mp_marker[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2], 
                                unsigned char mp_mask[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2]);

void __global__ MorphologicalReconstructionKernel(GlobalQueue* gq, volatile unsigned char* marker, unsigned char* mask, 
            size_t pitchMarker, size_t pitchMask, int size, int* memLeak);

// Primeira varredura raster/antiraster
void __global__ XForwardPropagationEntireImage(unsigned char* marker, unsigned char* mask, size_t pitchMarker, size_t pitchMask, int size);

void __global__ XBackwardPropagationEntireImage(unsigned char* marker, unsigned char* mask, size_t pitchMarker, size_t pitchMask, int size);

void __global__ YUpwardPropagationEntireImage(unsigned char* marker, unsigned char* mask, size_t pitchMarker, size_t pitchMask, int size);

void __global__ YDownwardPropagationEntireImage(unsigned char* marker, unsigned char* mask, size_t pitchMarker, size_t pitchMask, int size);

// Procedimento síncrono
void MorphologicalReconstruction(unsigned char* marker, unsigned char* mask, 
            size_t pitchMarker, size_t pitchMask, int size);

// Kernel usado para garantir que todo elemento do marker é menor ou igual que o elemento correspondente do mask
void __global__ clipImage(unsigned char* marker, unsigned char* mask, size_t pitchMarker, size_t pitchMask);

// Função usada para computar o tempo de execução do raster anti-raster alternativo
void one_alternative_raster_pass(unsigned char* marker, unsigned char* mask, int size);

// Função usada para computar o tempo de execução do raster anti-raster original
void one_current_pass(unsigned char* marker, unsigned char* mask, size_t pitchMarker, size_t pitchMask, int size);

#endif
