#include <stdio.h>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>

#include "globalQueue.cuh"
#include "MorphologicalReconstruction.cuh"
#include "testCaseGenerator.cuh"

int main(int argv, char** argc) {
    // |------------------------------------- 
    // | Teste com leitura de imagens
    // |-------------------------------------

    if(argv < 4) {
      printf("Número insuficiente de argumentos\n");
      return 1;
    }

    const char* marker_filename = argc[1];
    const char* mask_filename = argc[2];
    const char* output_filename = argc[3];

    cv::Mat marker_img, mask_img;
	  marker_img = cv::imread(marker_filename, cv::IMREAD_GRAYSCALE);
	  mask_img = cv::imread(mask_filename, cv::IMREAD_GRAYSCALE);

    // Verificando a leitura
    if(marker_img.empty() or mask_img.empty()) {
        printf("Falha ao ler as imagens\n");
        exit(1);
    }

    // Configurando o heap
    configureHeapSize();

    int size = marker_img.size().width;

    // Containers das imagens
    unsigned char* d_marker;
    size_t pitchMarker;
    unsigned char* d_mask;
    size_t pitchMask;

    // Alocamos as imagens
    cudaMallocPitch(&d_marker, &pitchMarker, size*sizeof(unsigned char), size);
    cudaMallocPitch(&d_mask, &pitchMask, size*sizeof(unsigned char), size);

    // Enviamos a imagem para a GPU
    cudaMemcpy2D(d_marker, pitchMarker, marker_img.data, marker_img.step, size*sizeof(unsigned char), size, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_mask, pitchMask, mask_img.data, mask_img.step, size*sizeof(unsigned char), size, cudaMemcpyHostToDevice);


    clipImage<<<dim3(size/FILL_BLOCK_SIZE,size/FILL_BLOCK_SIZE), dim3(FILL_BLOCK_SIZE,FILL_BLOCK_SIZE)>>>(d_marker, d_mask, pitchMarker, pitchMask);
    cudaDeviceSynchronize();
    
    auto tStart = std::chrono::high_resolution_clock::now();
    MorphologicalReconstruction(d_marker, d_mask, pitchMarker, pitchMask, size);
    auto tEnd = std::chrono::high_resolution_clock::now();

    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart);
    printf("%ld\n", ms_int.count());
    //printf("Tempo de execução: %ld (ms)\n", ms_int.count());

    // Enviamos o resultado para o host
    cudaMemcpy2D(marker_img.data, marker_img.step, d_marker, pitchMarker, size*sizeof(unsigned char), size, cudaMemcpyDeviceToHost);

    cv::imwrite(output_filename, marker_img, {259, 1});

    cudaFree(d_marker);
    cudaFree(d_mask);


    // |------------------------------------------
    // | Geração de imagens aleatórias
    // |------------------------------------------

    // int size = 8192;

    // // Configurando o heap
    // configureHeapSize();

    // cv::Mat marker_img(cv::Size(size, size), CV_8UC1); 
    // cv::Mat mask_img(cv::Size(size, size), CV_8UC1);

    // // Alocando as imagens
    // unsigned char *d_marker, *d_mask;
    // size_t pitchMarker, pitchMask;
    
    // cudaMallocPitch(&d_marker, &pitchMarker, size*sizeof(unsigned char), size);
    // cudaMallocPitch(&d_mask, &pitchMask, size*sizeof(unsigned char), size);

    // // Geramos a instância aleatória
    // InitRandomTestCase(d_marker, d_mask, pitchMarker, pitchMask, 4096, {{0,0}, {0,4096}, {4096,0}, {4096,4096}});
    // cudaDeviceSynchronize(); 

    // // Salvamos as imagens
    // cudaMemcpy2D(marker_img.data, marker_img.step, d_marker, pitchMarker, size*sizeof(unsigned char), size, cudaMemcpyDeviceToHost);
    // cudaMemcpy2D(mask_img.data, mask_img.step, d_mask, pitchMask, size*sizeof(unsigned char), size, cudaMemcpyDeviceToHost);


    // cv::imwrite("./Imagens_de_Teste/R-marker.tiff", marker_img, {259, 1});
    // cv::imwrite("./Imagens_de_Teste/R-mask.tiff", mask_img, {259, 1});

    // cudaFree(d_marker);
    // cudaFree(d_mask);


    return 0;
}
