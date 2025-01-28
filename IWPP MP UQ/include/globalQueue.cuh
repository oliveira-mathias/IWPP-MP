#ifndef GLOBAL_QUEUE_CUH
#define GLOBAL_QUEUE_CUH

#include <stdio.h>

#define DEVICE_MALLOC_LIMIT 1024*1024*1024
// Suficiente para acomodar uma fila de uma imgem de 102.400x102.400
#define GLOBAL_QUEUE_SIZE (5*10250000+10)

#define GQ_SUCCESS 0
#define GQ_FAILED 1

#define FILL_BLOCK_SIZE 16

struct GlobalQueue {
    // Ponteiros para a fila
    ushort2* readQueue;
    ushort2* writeQueue;

    int readQueueSize;
    int writeQueueSize;

    // O tamanho da fila é a primeira posição disponível no array
};

// Aumenta o tamanho do heap do device
void configureHeapSize();

// Espera-se gq seja ponteiro para memória no device
// Procedimento asssíncrono
void initGlobalQueue(GlobalQueue* gq);

// amount é o numero de registros a serem inseridos na fila
// Assume que vai ser executado por um kernel de forma
// A escrita so falha se nao tiver espaco suficiente
// Retornamos o numero de registros que foram escritos na fila
__device__ int insertIntoGlobalQueue(ushort2* queue, int* queueSize, int maxQueueSize, ushort2* buffer, int amount);

// Inicializa a fila com todos os megapixels
__global__ void initFillGlobalQueue(GlobalQueue* gq);

// Retorna a quantidade de registros lidos da fila
// O procedimento atualiza o offset de leitura (espera-se que esse esteja armazenado na memoria compartilhada) 
// amount e o numero de registros a serem lidos da fila
__device__ int readFromGlobalQueue(ushort2* queue, int& offset, int maxQueueSize, ushort2* buffer, int amount);

// Kernel interno do procedimento initGlobalQueue
__global__ void allocGlobalQueue(GlobalQueue* gq);

// Kernel para desalocar a GlobalQueue no device
__global__ void freeGlobalQueue(GlobalQueue* gq);

// Kernel para trocar de filas e apagar a antiga fila de leitura
__global__ void swapQueues(GlobalQueue* gq);

// Kernel para resetar a fila de leitura. Pré-processamento do Kernel extract uniques
__global__ void resetReadQueue(GlobalQueue* gq);

void testGlobalQueue();

#endif
