#include "globalQueue.cuh"

void configureHeapSize() {
    // Aumentamos o heap para acomodar a fila
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, DEVICE_MALLOC_LIMIT);
}

void initGlobalQueue(GlobalQueue* gq) {
    // Alocamos a fila
    allocGlobalQueue<<<1,1>>>(gq);
    cudaDeviceSynchronize();
}

__device__ bool atomicCASInt2(int2* address, int2 compare, int2 val) {
    unsigned long long* address_as_ull = (unsigned long long*) address;
    unsigned long long* compare_address_as_ull = (unsigned long long*) &compare;
    unsigned long long* val_address_as_ull = (unsigned long long*) &val;
    unsigned long long result;

    result = atomicCAS(address_as_ull, *compare_address_as_ull, *val_address_as_ull);

    return (result == *compare_address_as_ull);
}

__device__ int insertIntoGlobalQueue(GlobalQueue* gq, int2* data, int n, bool atomic) {
    __shared__ int writeOffset;
    int tid = threadIdx.x * blockDim.y + threadIdx.y;
    int assumedQueueSize_cached;

    if(tid==0) {
        writeOffset = -1;

        // Obtemos a trava da mutex
        // Busy waiting
        while(atomicExch(gq->mutex, LOCKED_MUTEX)) {}

        assumedQueueSize_cached = *(gq->globalSize);
        if(assumedQueueSize_cached + n <= MP_GLOBAL_QUEUE_SIZE) {
            //writeOffset = atomicAdd(gq->globalSize, n);
            writeOffset = assumedQueueSize_cached;
            *(gq->globalSize) += n;
        }
    }

    __syncthreads();

    if(writeOffset < 0) {
        if(tid == 0) {
            // Liberamos a mutex
            atomicExch(gq->mutex, FREE_MUTEX);
        }
        __syncthreads();
        return GQ_FAILED;
    }

    // Escrevemos os n elementos na fila
    if(tid < n) {
        //if(!atomic) {
        //    gq->globalQueue[writeOffset + tid] = data[tid];
        //}
        //else {
        //    // Estamos dentro de uma seção crítica
        //    atomicCASInt2(&(gq->globalQueue[writeOffset + tid]), gq->globalQueue[writeOffset + tid], data[tid]);
        //    // atomicExch(&(gq->globalQueue[writeOffset + tid].x), data[tid].x);
        //    // atomicExch(&(gq->globalQueue[writeOffset + tid].y), data[tid].y);
        //}
        gq->globalQueue[writeOffset + tid].x = data[tid].x;
        gq->globalQueue[writeOffset + tid].y = data[tid].y;
    }

    // Só podemos liberar a mutex quando todas as threads tiverem terminado de escrever
    __syncthreads();

    if(tid == 0) {
        // Liberamos a mutex
        atomicExch(gq->mutex, FREE_MUTEX);
    }


    return GQ_SUCCESS;
}

__global__ void initFillGlobalQueue(GlobalQueue* gq) {
    int xCoord = blockDim.x * blockIdx.x + threadIdx.x;
    int yCoord = blockDim.y * blockIdx.y + threadIdx.y;

    int tid = threadIdx.x * blockDim.y + threadIdx.y;

    __shared__ int2 writeData[FILL_BLOCK_SIZE*FILL_BLOCK_SIZE];

    writeData[tid] = make_int2(xCoord, yCoord);

    __syncthreads();

    // Após o encerramento desse kernel a fila já vai estar consistente na memória global
    insertIntoGlobalQueue(gq, writeData, FILL_BLOCK_SIZE*FILL_BLOCK_SIZE, false);
}

// Dá pra fazer um leitores escritores!!!
__device__ int readFromGlobalQueue(GlobalQueue* gq, int2* readBuffer, int n, bool& currentActive) {
    __shared__ int readOffset;
    __shared__ int readAmount;

    int currGlobalSize;

    int tid = threadIdx.x * blockDim.y + threadIdx.y;


    if(tid==0) {
        readAmount = n;

        // Obtemos a trava da mutex
        // Busy waiting
        while(atomicExch(gq->mutex, LOCKED_MUTEX)) {}

        currGlobalSize = *(gq->globalSize);
        if(readAmount > currGlobalSize) {
            readAmount = currGlobalSize;
        }

        //readOffset = atomicSub(gq->globalSize, readAmount);
        readOffset = currGlobalSize;
        *(gq->globalSize) -= readAmount;
    }

    __syncthreads();

    if(tid < readAmount) {
        // Deve dar para otimizar
        readBuffer[tid].x = gq->globalQueue[readOffset - tid -1].x;
        readBuffer[tid].y = gq->globalQueue[readOffset - tid -1].y;
    }

    if(tid==0) {
        // Acertamos o contador de blocos ativos
        if(readAmount > 0 and !currentActive) {
            //atomicAdd(&(gq->activeBlocksCount), 1);
            gq->activeBlocksCount += 1;
            currentActive = true;
        }
        else if(readAmount == 0 and currentActive) {
            //atomicSub(&(gq->activeBlocksCount), 1);
            gq->activeBlocksCount -= 1;
            currentActive = false;
        }

        // Liberamos a mutex
        atomicExch(gq->mutex, FREE_MUTEX);
    }

    __syncthreads();

    return readAmount;
}

// Deve ter só uma thread e 1 bloco executando
__global__ void allocGlobalQueue(GlobalQueue* gq) {
    int2* dataPtr;
    int* sizePtr;
    int* mutexPtr;

    // Pode ser necessário aumentar o tamanho do heap
    cudaMalloc(&dataPtr, MP_GLOBAL_QUEUE_SIZE*sizeof(int2));
    cudaMalloc(&sizePtr, sizeof(int));
    cudaMalloc(&mutexPtr, sizeof(int));

    *sizePtr = 0;
    *mutexPtr = FREE_MUTEX;

    // printf("StartPos: %p\n", dataPtr);

    gq->globalQueue = dataPtr;
    gq->globalSize = sizePtr;
    gq->mutex = mutexPtr;
    gq->activeBlocksCount = 0;
}

// Deve ter só uma thread e 1 bloco executando
__global__ void freeGlobalQueue(GlobalQueue* gq) {
    cudaFree((int2*) gq->globalQueue);
    cudaFree((int*) gq->globalSize);
    cudaFree(gq->mutex);
}

__global__ void testGlobalQueue(GlobalQueue* gq) {
    int tid = threadIdx.x;
    // int bid = blockIdx.x;

    // __shared__ GlobalQueue gq;
    __shared__ int2 readBuffer[32];
    int totalReaded = 0;

    // if(tid==0) {
        // gq.globalQueue = globalQueuePtr;
        // gq.globalSize = globalQueueSizePtr;

        // printf("%d\n", *(gq.globalSize));
        // printf("%d\n", tid);
    // }

    int readAmount = 1;

    bool state = true;
    for(int i=0; i<10; i++) {
        readAmount = readFromGlobalQueue(gq, readBuffer, 5, state);

        // if(tid < readAmount) {
        //     printf("(%d, %d),\n", readBuffer[tid].x, readBuffer[tid].y);
        // }

        totalReaded += readAmount;

        __syncthreads();
    }
    // for(int i=0; i<*(gq->globalSize); i++) {
    //     printf("(%d, %d),\n", gq->globalQueue[i].x, gq->globalQueue[i].y);
    // }

    if(tid==0) {
        printf("%d ,", totalReaded);
    }

    // if(tid==0) {
    //     printf("size: %d\n", *(gq->globalSize));
    // }
    // __syncthreads();

}
