#ifndef EUCLIDEAN_DISTANCE_TRANSFORM_CUH
#define EUCLIDEAN_DISTANCE_TRANSFORM_CUH

#include<stdio.h>
#include "globalQueue.cuh"
#include "testCaseGenerator.cuh"

#define MAX_UNSIGNED_INT 4294967295
#define MAX_UNSIGNED_SHORT 65535

#define MEGAPIXEL_SIZE 32
#define BLOCK_SIZE 32

// Pode dar overflow para imagens de lado maior ou igual do que 16k
int __device__ euclideanDistance(int x1, int y1, int x2, int y2);

// Retorna se algum pixel foi atualizado
bool __device__ XForwardPropagation(ushort2 mp_voronoi[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2], int baseRow, int baseColumn);

bool __device__ XBackwardPropagation(ushort2 mp_voronoi[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2], int baseRow, int baseColumn);

bool __device__ YUpwardPropagation(ushort2 mp_voronoi[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2], int baseRow, int baseColumn);

bool __device__ YDownwardPropagation(ushort2 mp_voronoi[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2], int baseRow, int baseColumn);

void __global__ EuclideanDistanceTransformKernel(GlobalQueue* gq, ushort2* voronoi, size_t pitchVoronoi, int size, int* memLeak);

// Procedimento síncrono
void EuclideanDistanceTransform(ushort2* voronoi, size_t pitchVoronoi, int size);

// Kernel de computação das distâncias
void __global__ computeDistMap(float* distMap, size_t pitchDistMap, ushort2* voronoi, size_t pitchVoronoi);

// Kernel usado para inicializar o diagrama de voronoi a partir de uma imagem
void __global__ extractBaseVoronoi(unsigned char* image, int pitchImage, ushort2* voronoi, size_t pitchVoronoi);

#endif