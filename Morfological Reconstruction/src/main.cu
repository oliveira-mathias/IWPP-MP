#include <stdio.h>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/imgcodecs.hpp>

#include "globalQueue.cuh"
#include "MorphologicalReconstruction.cuh"
#include "testCaseGenerator.cuh"

int main() {
    // |------------------------------------- 
    // | Teste com leitura de imagens
    // |-------------------------------------
    cv::Mat marker_img, mask_img;
	marker_img = cv::imread("./Imagens_de_Teste/75-percent-marker.jpg", cv::IMREAD_GRAYSCALE);
	mask_img = cv::imread("./Imagens_de_Teste/75-percent-mask.jpg", cv::IMREAD_GRAYSCALE);

    // Verificando a leitura
    if(marker_img.empty() or mask_img.empty()) {
        printf("Falha ao ler as imagens\n");
        exit(1);
    }

    // Configurando o heap
    configureHeapSize();

    int size = 4096;

    // Containers das imagens
    cv::cuda::GpuMat d_marker;
	cv::cuda::GpuMat d_mask;

    // Enviamos a imagem para a GPU
    d_marker.upload(marker_img);
	d_mask.upload(mask_img);


    clipImage<<<dim3(size/FILL_BLOCK_SIZE,size/FILL_BLOCK_SIZE), dim3(FILL_BLOCK_SIZE,FILL_BLOCK_SIZE)>>>((unsigned char*)d_marker.data, (unsigned char*)d_mask.data, d_marker.step, d_mask.step);
    cudaDeviceSynchronize();
    
    auto tStart = std::chrono::high_resolution_clock::now();
    MorphologicalReconstruction((unsigned char*)d_marker.data, (unsigned char*)d_mask.data, d_marker.step, d_mask.step, size);
    auto tEnd = std::chrono::high_resolution_clock::now();

    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart);
    printf("Tempo de execução: %ld (ms)\n", ms_int.count());

    d_marker.download(marker_img);

    cv::imwrite("./Imagens_de_Teste/res-cuda-75.tiff", marker_img, {259, 1});


    // |------------------------------------------
    // | Geração de imagens aleatórias
    // |------------------------------------------

    // int size = 8192;

    // // Configurando o heap
    // configureHeapSize();

    // cv::Mat marker_img(cv::Size(size, size), CV_8UC1); 
    // cv::Mat mask_img(cv::Size(size, size), CV_8UC1);

    // cv::cuda::GpuMat d_marker(cv::Size(size, size), CV_8UC1);
	// cv::cuda::GpuMat d_mask(cv::Size(size, size), CV_8UC1);

    // // Geramos a instância aleatória
    // InitRandomTestCase(d_marker.data, d_mask.data, d_marker.step, d_mask.step, 4096, {{0,0}, {0,4096}, {4096,0}, {4096,4096}});
    // cudaDeviceSynchronize(); 

    // // Salvamos as imagens
    // d_marker.download(marker_img);
    // d_mask.download(mask_img);

    // cv::imwrite("./Imagens_de_Teste/R-marker.tiff", marker_img, {259, 1});
    // cv::imwrite("./Imagens_de_Teste/R-mask.tiff", mask_img, {259, 1});


    return 0;
}
