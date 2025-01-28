#ifndef TEST_CASE_GENERATOR_CUH
#define TEST_CASE_GENERATOR_CUH

#include<stdio.h>
#include <vector>
#include <utility>
#include <curand_kernel.h>

// Kernels para geração de números aleatórios(Setup)
void __global__ randomNumberGeneratorSetupKernel(curandState *state, int seed);

// Kernels para geração de números aleatórios(Generate)
void __global__ randomNumberGeneratorGenerateKernel(curandState *state, unsigned char* marker, unsigned char* mask, 
        size_t pitchMarker, size_t pitchMask, int rowOffset, int colOfsset);

// Preenche a imagem em blocos de tamamho sizexsize
// Ofssets controla os locais de escrita dos números aleatórios
// Assume que size é uma potência de 2 maior que 16
// Função síncrona
void InitRandomTestCase(unsigned char* marker, unsigned char* mask, 
            size_t pitchMarker, size_t pitchMask, int randomGeneratorGridSize, std::vector<std::pair<int, int>> offsets);

#endif