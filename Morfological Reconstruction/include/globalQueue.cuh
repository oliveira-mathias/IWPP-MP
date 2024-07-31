#ifndef GLOBAL_QUEUE_CUH
#define GLOBAL_QUEUE_CUH

#include <stdio.h>

// Cabe 100 grids de megapixels (32x32) de tamanho 4096*4096
#define DEVICE_MALLOC_LIMIT 1500*1024*1024
#define MP_GLOBAL_QUEUE_SIZE 91*1024*1024

#define LOCKED_MUTEX 1
#define FREE_MUTEX 0

#define GQ_SUCCESS 0
#define GQ_FAILED 1

#define FILL_BLOCK_SIZE 16

struct GlobalQueue {
    // Ponteiro para memória global
    int2* globalQueue;
    // Tamanho da fila
    int* globalSize;
    // Mutex de acesso
    int* mutex;
    int activeBlocksCount;


    // O tamanho da fila é a primeira posição disponível no array
};

// Aumenta o tamanho do heap do device
void configureHeapSize();

// Espera-se gq seja ponteiro para memória no device
void initGlobalQueue(GlobalQueue* gq);

// Retorna se a operação de escrita ocorreu
__device__ bool atomicCASInt2(int2* address, int2 compare, int2 val);

// n é o número de registros a serem inseridos na fila
// Assume que vai ser executado por um kernel de forma (1, k)
// A escrita só falha se não tiver espaço suficiente
__device__ int insertIntoGlobalQueue(GlobalQueue* gq, int2* data, int n, bool atomic);

// Inicializa a fila com todos os megapixels
__global__ void initFillGlobalQueue(GlobalQueue* gq);

// Assume que readBuffer está na memória compartilhada
__device__ int readFromGlobalQueue(GlobalQueue* gq, int2* readBuffer, int n, bool& currentActive);

// Kernel interno do procedimento initGlobalQueue
__global__ void allocGlobalQueue(GlobalQueue* gq);

// Kernel para desalocar a GlobalQueue no device
__global__ void freeGlobalQueue(GlobalQueue* gq);

__global__ void testGlobalQueue(GlobalQueue* gq);

#endif