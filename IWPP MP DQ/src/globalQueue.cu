#include "globalQueue.cuh"

void configureHeapSize() {
    // Aumentamos o heap para acomodar a fila
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, DEVICE_MALLOC_LIMIT);
}

void initGlobalQueue(GlobalQueue* gq) {
    // Alocamos a fila
    allocGlobalQueue<<<1,1>>>(gq);
    //cudaDeviceSynchronize();
}

__device__ int insertIntoGlobalQueue(ushort2* queue, int* queueSize, int maxQueueSize, ushort2* buffer, int amount) {
    __shared__ int offset;
    __shared__ int realWrittenAmount;
    int tid = threadIdx.x * blockDim.y + threadIdx.y;
    int blockSize = blockDim.x*blockDim.y;

    int i;

    // Para evitar que ocorra uma escrita antes da escrita anterior terminar
    __syncthreads();

    if(tid==0) {
        //Obtemos o offset de escrita
        offset = atomicAdd(queueSize, amount);

        // Computamos quantos registros de fato serão escritos
        realWrittenAmount = amount;
        // Aqui ocorreria um buffer overflow
        if(offset + amount -1 >= maxQueueSize) {
          realWrittenAmount = max(maxQueueSize - offset, 0);
        }

    }

    // Sincronia para garantia de consistencia de memoria compartilhada
    __syncthreads();

    // Sera que as escritas sao agregadas?????
    for(i=0; blockSize*i + tid < realWrittenAmount; i++){
      queue[offset + blockSize*i + tid] = buffer[blockSize*i + tid];
    }

    return realWrittenAmount;
}

__global__ void initFillGlobalQueue(GlobalQueue* gq) {
    int xCoord = blockDim.x * blockIdx.x + threadIdx.x;
    int yCoord = blockDim.y * blockIdx.y + threadIdx.y;

    int tid = threadIdx.x * blockDim.y + threadIdx.y;

    __shared__ ushort2 writeData[FILL_BLOCK_SIZE*FILL_BLOCK_SIZE];

    writeData[tid] = make_ushort2(xCoord, yCoord);

    // Após o encerramento desse kernel a fila já vai estar consistente na memória global
    // Note que, temos que escrever na fila de leitura, para evitar uma troca de filas denecessaria!!!
    insertIntoGlobalQueue(gq->readQueue, &(gq->readQueueSize), GLOBAL_QUEUE_SIZE, writeData, FILL_BLOCK_SIZE*FILL_BLOCK_SIZE);
}

__device__ int readFromGlobalQueue(ushort2* queue, int& offset, int maxQueueSize, ushort2* buffer, int amount) {
    __shared__ int realReadAmount;

    int tid = threadIdx.x * blockDim.y + threadIdx.y;
    int blockSize = blockDim.x*blockDim.y;
    int i;

    // Para evitar que ocorra uma leitura antes da leitura anterior terminar
    __syncthreads();

    if(tid==0) {
        realReadAmount = amount;

        // Verificamos se de fato os registros desejados existem na fila
        if(offset + realReadAmount -1 >= maxQueueSize) {
          realReadAmount = maxQueueSize - offset;
        }

    }

    //Sincronizamos para garantia de consistencia de memoria compartilhada
    __syncthreads();

    // Sera que as leituras sao agregadas?????
    for(i=0; blockSize*i + tid < realReadAmount; i++) {
      buffer[blockSize*i + tid] = queue[offset + blockSize*i + tid];
    }

    // Todo mundo deve terminar de escrever para que o offset possa ser atualizado
    __syncthreads();

    if(tid==0) {
        // Acertamos offset
        offset += realReadAmount;
    }

    return realReadAmount;
}

// Deve ter só uma thread e 1 bloco executando
__global__ void allocGlobalQueue(GlobalQueue* gq) {
    cudaError_t err;

    // Pode ser necessário aumentar o tamanho do heap
    // Alocamos as filas
    err = cudaMalloc(&(gq->readQueue), GLOBAL_QUEUE_SIZE*sizeof(ushort2));
    if(err) {
      printf("%s\n", cudaGetErrorString(err));
    }
    
    err = cudaMalloc(&(gq->writeQueue), GLOBAL_QUEUE_SIZE*sizeof(ushort2));
    if(err) {
      printf("%s\n", cudaGetErrorString(err));
    }

    //Inicializando o tamanho das filas
    gq->readQueueSize = 0;
    gq->writeQueueSize = 0;

}

// Deve ter só uma thread e 1 bloco executando
__global__ void freeGlobalQueue(GlobalQueue* gq) {
    cudaFree(gq->readQueue);
    cudaFree(gq->writeQueue);
}

__global__ void debugQueue(GlobalQueue* gq) {
  printf("Endereco fila de leitura: %p\n", gq->readQueue);
  printf("Tamanho da fila de leitura: %d\n", gq->readQueueSize);
  printf("Endereco fila de escrita: %p\n", gq->writeQueue);
  printf("Tamanho da fila de escrita: %d\n", gq->writeQueueSize);
  printf("-------------------------------------\n");
}

__global__ void swapQueues(GlobalQueue* gq) {
  ushort2* ptrAux;
        
  // Primeiro estabelecemos a integridade da fila de escrita
  // Nesse caso ocorreu um overflow durante a escrita
  if(gq->writeQueueSize > GLOBAL_QUEUE_SIZE) {
     gq->writeQueueSize = GLOBAL_QUEUE_SIZE;
  }

  // Trocamos o tamanho das filas e "apagamos" a nova fila de escrita
  gq->readQueueSize = gq->writeQueueSize;
  gq->writeQueueSize = 0;

  // Trocamos as filas
  ptrAux = gq->readQueue;
  gq->readQueue = gq->writeQueue;
  gq->writeQueue = ptrAux;

}


// Assumimos que os registros da fila sao de um grid quadrado
__global__ void verifyQueue(GlobalQueue* gq, int* queueCounter, int lateralSize) {
  const int tid = threadIdx.x*blockDim.y + threadIdx.y;
  const int blockId = blockIdx.x*gridDim.y + blockIdx.y;
  const int numBlocks = gridDim.x*gridDim.y;
  const int blockSize = blockDim.x*blockDim.y;
        
  // Divisao logica da fila de leitura
  const int quotient = (gq->readQueueSize)/numBlocks;
  const int remainder = (gq->readQueueSize)%numBlocks;

  // Calculando o tamanho da fila
  int readQueueSize = quotient;
  if(blockId < remainder) {
    readQueueSize++;
  }

  // Calculando o offset
  int offset = min(remainder, blockId)*(quotient + 1);
  if(blockId > remainder) {
    offset += (blockId - remainder)*quotient;
  }

  // Ponteiro para a fila
  ushort2* readGlobalQueue = &(gq->readQueue[offset]);

  // Variaveis de leitura
  __shared__ ushort2 readBuffer[32];
  __shared__ int readIndex;
  int readAmount;

  if(tid==0) {
    readIndex = 0;
  }

  __syncthreads();

  while(readIndex < readQueueSize) {
    readAmount = readFromGlobalQueue(readGlobalQueue, readIndex, readQueueSize, readBuffer, blockSize);
    
    // Processamos os registros lidos
    if(tid < readAmount) {
      atomicAdd(&(queueCounter[readBuffer[tid].x*lateralSize + readBuffer[tid].y]), 1);
    }
  }

  
}

void __global__ compareKernel(int* queueCounter, int lateralSide, int toCompare) {
  int myRow = blockIdx.x*blockDim.x + threadIdx.x;
  int myCol = blockIdx.y*blockDim.y + threadIdx.y;

  if(queueCounter[myRow*lateralSide + myCol] != toCompare) {
    printf("Verificacao de Queue: (%d, %d)\n", myRow, myCol);
  }
}

void testGlobalQueue() {
  GlobalQueue* gq;

  // Alocamos um verificador
  int *queueCounter;

  cudaMalloc(&queueCounter, 128*128*sizeof(int));
  cudaMemset(queueCounter, 0, 128*128*sizeof(int));

  // Alocamos a fila
  cudaMalloc(&gq, sizeof(GlobalQueue));

  // Inicializamos a fila
  initGlobalQueue(gq);

  // Vamos inserir alguns registros na fila
  initFillGlobalQueue<<<dim3(8, 8), dim3(16, 16)>>>(gq);
  initFillGlobalQueue<<<dim3(8, 8), dim3(16, 16)>>>(gq);
  initFillGlobalQueue<<<dim3(8, 8), dim3(16, 16)>>>(gq);
  initFillGlobalQueue<<<dim3(8, 8), dim3(16, 16)>>>(gq);
  //debugQueue<<<1,1>>>(gq);
  swapQueues<<<1,1>>>(gq);
  //debugQueue<<<1,1>>>(gq);
  initFillGlobalQueue<<<dim3(8, 8), dim3(16, 16)>>>(gq);
  //debugQueue<<<1,1>>>(gq);
  swapQueues<<<1,1>>>(gq);
  initFillGlobalQueue<<<dim3(8, 8), dim3(16, 16)>>>(gq);
  initFillGlobalQueue<<<dim3(8, 8), dim3(16, 16)>>>(gq);
  //debugQueue<<<1,1>>>(gq);
  verifyQueue<<<200,32>>>(gq, queueCounter, 128);
  compareKernel<<<dim3(8,8), dim3(16,16)>>>(queueCounter, 128, 2);
  cudaDeviceSynchronize();

  cudaFree(gq);
  cudaFree(queueCounter);
}
