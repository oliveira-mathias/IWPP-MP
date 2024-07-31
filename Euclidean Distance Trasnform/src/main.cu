#include <stdio.h>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/imgcodecs.hpp>

#include "globalQueue.cuh"
#include "EuclideanDistanceTransform.cuh"
#include "testCaseGenerator.cuh"


void distMatrixToCsv(float* distMap, int width, int height, const char* filename) {
    FILE* arquivo;

    arquivo = fopen(filename, "w");

    int i, j;

    for(i=0; i<height; i++) {
        fprintf(arquivo, "%.5f", distMap[i*width]);
        for(j=1; j<width; j++) {
            fprintf(arquivo, ",%.5f", distMap[i*width+ j]);
        }
        fprintf(arquivo, "\n");
    }

    fclose(arquivo);
}

int main() {
    
    // |-------------------------
    // | Geração de caso de teste
    // |--------------------------
    // int size = 8192;

    // ushort2* dVoronoi;
    // size_t pitchVoronoi;

    // cudaError_t err;

    // // Aumenta o tamnho do Heap do device para a execução ser bem sucedida
    // configureHeapSize();

    // // Alocamos a matriz do diagrama de Voronoi
    // err = cudaMallocPitch(&dVoronoi, &pitchVoronoi, size*sizeof(ushort2), size);
    // printf("%s\n", cudaGetErrorString(err));

    // // InitRandomTestCase(dVoronoi, pitchVoronoi, 4096, 0.05, {{0, 0}, {0, 4096}, {4096, 0}, {4096,4096}});
    // initCustomTestCase<<<dim3(size/FILL_BLOCK_SIZE, size/FILL_BLOCK_SIZE), dim3(FILL_BLOCK_SIZE,FILL_BLOCK_SIZE)>>>(dVoronoi, pitchVoronoi);
    // cudaDeviceSynchronize();
    // printf("Imagem gerada\n");

    // // Container da imagem gerada
    // cv::cuda::GpuMat d_imagem(cv::Size(size, size), CV_8UC1);

    // extractImage<<<dim3(size/FILL_BLOCK_SIZE,size/FILL_BLOCK_SIZE), dim3(FILL_BLOCK_SIZE,FILL_BLOCK_SIZE)>>>((unsigned char*) d_imagem.data, d_imagem.step, dVoronoi, pitchVoronoi);
    // cudaDeviceSynchronize();

    // // Puxamos a imagem pro host
    // cv::Mat h_image(cv::Size(size, size), CV_8UC1);
    // d_imagem.download(h_image);

    // // Salvamos a imagem
    // cv::imwrite("./ImagensTeste/casoTeste.tiff", h_image, {259, 1});

    // // Desalocamos a matriz
    // cudaFree(dVoronoi);

    // |-------------------------------------------------------------------------------
    // | Execução da transformada de distância euclidiana a partir da leitura de imagem
    // |-------------------------------------------------------------------------------

    int size=4096;

    // Configurando o heap
    configureHeapSize();

    cv::Mat mask_img;
	mask_img = cv::imread("./ImagensTeste/corners.tiff", cv::IMREAD_GRAYSCALE);

    // Verificando a leitura
    if(mask_img.empty()) {
        printf("Falha ao ler a imagem\n");
        exit(1);
    }

    // Lemos a imagem
	cv::cuda::GpuMat d_mask;

    // Enviamos a imagem para a GPU
	d_mask.upload(mask_img);

    // Variáveis do algoritmo
    ushort2* d_voronoi;
    size_t pitch_voronoi;

    float* d_dist_map;
    size_t pitch_dist_map;

    cudaMallocPitch(&d_voronoi, &pitch_voronoi, size*sizeof(ushort2), size);
    cudaMallocPitch(&d_dist_map, &pitch_dist_map, size*sizeof(float), size);

    // Extraimos o input para o algoritmo de transformada de distância euclidiana
    extractBaseVoronoi<<<dim3(size/FILL_BLOCK_SIZE, size/FILL_BLOCK_SIZE), dim3(FILL_BLOCK_SIZE, FILL_BLOCK_SIZE)>>>(d_mask.data, d_mask.step, d_voronoi, pitch_voronoi);
    cudaDeviceSynchronize();

    // Realizamos a computação
    auto tStart = std::chrono::high_resolution_clock::now();
    EuclideanDistanceTransform(d_voronoi, pitch_voronoi, size);
    computeDistMap<<<dim3(size/FILL_BLOCK_SIZE,size/FILL_BLOCK_SIZE), dim3(FILL_BLOCK_SIZE,FILL_BLOCK_SIZE)>>>(d_dist_map, pitch_dist_map, d_voronoi, pitch_voronoi);
    cudaDeviceSynchronize();
    auto tEnd = std::chrono::high_resolution_clock::now();

    // Imprimimos o tempo de execução
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart);
    printf("Tempo de execução: %ld (ms)\n", ms_int.count());

    // Salvamos o resultado da computação
    float* h_dist_map = (float*)malloc(size*size*sizeof(float));

    cudaMemcpy2D(h_dist_map, size*sizeof(float), d_dist_map, pitch_dist_map, size*sizeof(float), size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("Salvando resultados\n");
    distMatrixToCsv(h_dist_map, size, size, "./ImagensTeste/corners_sol.csv");

    // Liberamos os recursos alocados
    cudaFree(d_voronoi);
    cudaFree(d_dist_map);
    free(h_dist_map);

    return 0;
}
