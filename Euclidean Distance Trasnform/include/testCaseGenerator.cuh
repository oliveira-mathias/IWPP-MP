#ifndef TEST_CASE_GENERATOR_CUH
#define TEST_CASE_GENERATOR_CUH

#include<stdio.h>
#include<vector>
#include<utility>
#include<curand_kernel.h>


#define MAX_UNSIGNED_INT 4294967295
#define MAX_UNSIGNED_SHORT 65535


// Kernels para geração de números aleatórios(Setup)
void __global__ randomNumberGeneratorSetupKernel(curandState *state, int seed);

// Kernels para geração de números aleatórios(Generate)
// Note que a classe do pixel (Background/Forground) está embutida no seu valor associado
// BackgroundProb é a probabilidade de um pixel ser do tipo background
void __global__ randomNumberGeneratorGenerateKernel(curandState *state, ushort2* voronoi, size_t pitchVoronoi, float backgroundProb, int rowOffset, int colOffset);

// Preenche a imagem em blocos de tamamho sizexsize
// Ofssets controla os locais de escrita dos números aleatórios
// Assume que size é uma potência de 2 maior que 16
// Função síncrona
void InitRandomTestCase(ushort2* voronoi, size_t pitchVoronoi, int randomGeneratorGridSize, float backgroundProb, std::vector<std::pair<int, int>> offsets);

// Kernel usado para extrair a imagem correspondente ao voronoi gerado aleatóriamente
void __global__ extractImage(unsigned char* image, int pitchImage, ushort2* voronoi, size_t pitchVoronoi);

// Kernel usado para inicializar de forma customizada instâncias de teste
void __global__ initCustomTestCase(ushort2* voronoi, size_t pitchVoronoi);

#endif
