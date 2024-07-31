#include "EuclideanDistanceTransform.cuh"

int __device__ euclideanDistance(int x1, int y1, int x2, int y2) {
    return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
}

bool __device__ XForwardPropagation(ushort2 mp_voronoi[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2], int baseRow, int baseColumn) {
    int tid = threadIdx.x;
    int elementsUpdated = 0;
    int j;
    const int myBonusRow = (MEGAPIXEL_SIZE+1)*tid;


    for(j=0; j<MEGAPIXEL_SIZE+1; j++){
        if(mp_voronoi[tid+1][j].x != MAX_UNSIGNED_SHORT) {
            if((mp_voronoi[tid+1][j+1].x == MAX_UNSIGNED_SHORT) or (mp_voronoi[tid+1][j+1].x != MAX_UNSIGNED_SHORT and 
                (euclideanDistance(baseRow + tid, baseColumn + j, mp_voronoi[tid+1][j+1].x, mp_voronoi[tid+1][j+1].y)
                > euclideanDistance(baseRow + tid, baseColumn + j, mp_voronoi[tid+1][j].x, mp_voronoi[tid+1][j].y)) )) 
            {
                mp_voronoi[tid+1][j+1] = mp_voronoi[tid+1][j];
                elementsUpdated = 1;
            }
        }
    }

    // Laterais superior e inferior
    if(!(threadIdx.x >> 1)) {
        for(j=0; j<MEGAPIXEL_SIZE+1; j++){
            if(mp_voronoi[myBonusRow][j].x != MAX_UNSIGNED_SHORT) {
                if((mp_voronoi[myBonusRow][j+1].x == MAX_UNSIGNED_SHORT) or (mp_voronoi[myBonusRow][j+1].x != MAX_UNSIGNED_SHORT and 
                    (euclideanDistance(baseRow + myBonusRow -1, baseColumn + j, mp_voronoi[myBonusRow][j+1].x, mp_voronoi[myBonusRow][j+1].y)
                    > euclideanDistance(baseRow + myBonusRow -1, baseColumn + j, mp_voronoi[myBonusRow][j].x, mp_voronoi[myBonusRow][j].y)))) 
                {
                    mp_voronoi[myBonusRow][j+1] = mp_voronoi[myBonusRow][j];
                    elementsUpdated = 1;
                }
            }

        }
    }

    // Verificamos se algum elemento foi atualizado
    elementsUpdated = __syncthreads_or(elementsUpdated);

    return (elementsUpdated != 0);
}

bool __device__ XBackwardPropagation(ushort2 mp_voronoi[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2], int baseRow, int baseColumn) {
    int tid = threadIdx.x;
    int elementsUpdated = 0;
    int j;
    const int myBonusRow = (MEGAPIXEL_SIZE+1)*tid;


    for(j=MEGAPIXEL_SIZE+1; j>0; j--){
        if(mp_voronoi[tid+1][j].x != MAX_UNSIGNED_SHORT) {
            if((mp_voronoi[tid+1][j-1].x == MAX_UNSIGNED_SHORT) or (mp_voronoi[tid+1][j-1].x != MAX_UNSIGNED_SHORT and 
                (euclideanDistance(baseRow + tid, baseColumn + j - 2, mp_voronoi[tid+1][j-1].x, mp_voronoi[tid+1][j-1].y)
                > euclideanDistance(baseRow + tid, baseColumn + j - 2, mp_voronoi[tid+1][j].x, mp_voronoi[tid+1][j].y)))) 
            {
                mp_voronoi[tid+1][j-1] = mp_voronoi[tid+1][j];
                elementsUpdated = 1;
            }
        }

    }

    // Laterais superior e inferior
    if(!(threadIdx.x >> 1)) {
        for(j=MEGAPIXEL_SIZE+1; j>0; j--){
            if(mp_voronoi[myBonusRow][j].x != MAX_UNSIGNED_SHORT) {
                if((mp_voronoi[myBonusRow][j-1].x == MAX_UNSIGNED_SHORT) or (mp_voronoi[myBonusRow][j-1].x != MAX_UNSIGNED_SHORT and 
                    (euclideanDistance(baseRow + myBonusRow -1, baseColumn + j - 2, mp_voronoi[myBonusRow][j-1].x, mp_voronoi[myBonusRow][j-1].y)
                    > euclideanDistance(baseRow + myBonusRow -1, baseColumn + j - 2, mp_voronoi[myBonusRow][j].x, mp_voronoi[myBonusRow][j].y)))) 
                {
                    mp_voronoi[myBonusRow][j-1] = mp_voronoi[myBonusRow][j];
                    elementsUpdated = 1;
                }
            }
        }

    }

    // Verificamos se algum elemento foi atualizado
    elementsUpdated = __syncthreads_or(elementsUpdated);

    return (elementsUpdated != 0);
}

bool __device__ YUpwardPropagation(ushort2 mp_voronoi[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2], int baseRow, int baseColumn) {
    int tid = threadIdx.x;
    int elementsUpdated = 0;
    int i;
    const int myBonusCol = (MEGAPIXEL_SIZE+1)*tid;


    for(i=MEGAPIXEL_SIZE+1; i>0; i--){
        if(mp_voronoi[i][tid+1].x != MAX_UNSIGNED_SHORT) {
            if((mp_voronoi[i-1][tid+1].x == MAX_UNSIGNED_SHORT) or (mp_voronoi[i-1][tid+1].x != MAX_UNSIGNED_SHORT and 
                (euclideanDistance(baseRow + i -2, baseColumn + tid, mp_voronoi[i-1][tid+1].x, mp_voronoi[i-1][tid+1].y)
                > euclideanDistance(baseRow + i -2, baseColumn + tid, mp_voronoi[i][tid+1].x, mp_voronoi[i][tid+1].y)))) 
            {
                mp_voronoi[i-1][tid+1] = mp_voronoi[i][tid+1];
                elementsUpdated = 1;
            }
        }

    }

    // Laterais superior e inferior
    if(!(threadIdx.x >> 1)) {
        for(i=MEGAPIXEL_SIZE+1; i>0; i--){
            if(mp_voronoi[i][myBonusCol].x != MAX_UNSIGNED_SHORT) {
                if((mp_voronoi[i-1][myBonusCol].x == MAX_UNSIGNED_SHORT) or (mp_voronoi[i-1][myBonusCol].x != MAX_UNSIGNED_SHORT and 
                    (euclideanDistance(baseRow + i - 2, baseColumn + myBonusCol - 1, mp_voronoi[i-1][myBonusCol].x, mp_voronoi[i-1][myBonusCol].y)
                    > euclideanDistance(baseRow + i - 2, baseColumn + myBonusCol - 1, mp_voronoi[i][myBonusCol].x, mp_voronoi[i][myBonusCol].y))))
                {
                    mp_voronoi[i-1][myBonusCol] = mp_voronoi[i][myBonusCol];
                    elementsUpdated = 1;
                }
            }
            
        }
    }

    // Verificamos se algum elemento foi atualizado
    elementsUpdated = __syncthreads_or(elementsUpdated);

    return (elementsUpdated != 0);
}

bool __device__ YDownwardPropagation(ushort2 mp_voronoi[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2], int baseRow, int baseColumn) {
    unsigned int tid = threadIdx.x;
    int elementsUpdated = 0;
    int i;
    const int myBonusCol = (MEGAPIXEL_SIZE+1)*tid;


    for(i=0; i<MEGAPIXEL_SIZE+1; i++){
        if(mp_voronoi[i][tid+1].x != MAX_UNSIGNED_SHORT) {
            if((mp_voronoi[i+1][tid+1].x == MAX_UNSIGNED_SHORT) or (mp_voronoi[i+1][tid+1].x != MAX_UNSIGNED_SHORT and 
                (euclideanDistance(baseRow + i, baseColumn + tid, mp_voronoi[i+1][tid+1].x, mp_voronoi[i+1][tid+1].y)
                > euclideanDistance(baseRow + i, baseColumn + tid, mp_voronoi[i][tid+1].x, mp_voronoi[i][tid+1].y))))
            {
                mp_voronoi[i+1][tid+1] = mp_voronoi[i][tid+1];
                elementsUpdated = 1;
            }    
        }

    }

    // Laterais superior e inferior
    if(!(threadIdx.x >> 1)) {
        for(i=0; i<MEGAPIXEL_SIZE+1; i++){
            if(mp_voronoi[i][myBonusCol].x != MAX_UNSIGNED_SHORT) {
                if((mp_voronoi[i+1][myBonusCol].x == MAX_UNSIGNED_SHORT) or (mp_voronoi[i+1][myBonusCol].x != MAX_UNSIGNED_SHORT and 
                    (euclideanDistance(baseRow + i, baseColumn + myBonusCol - 1, mp_voronoi[i+1][myBonusCol].x, mp_voronoi[i+1][myBonusCol].y)
                    > euclideanDistance(baseRow + i, baseColumn + myBonusCol - 1, mp_voronoi[i][myBonusCol].x, mp_voronoi[i][myBonusCol].y))))
                {
                    mp_voronoi[i+1][myBonusCol] = mp_voronoi[i][myBonusCol];
                    elementsUpdated = 1;
                }
            }

        }
    }

    // Verificamos se algum elemento foi atualizado
    elementsUpdated = __syncthreads_or(elementsUpdated);

    return (elementsUpdated != 0);
}

void __global__ EuclideanDistanceTransformKernel(GlobalQueue* gq, ushort2* voronoi, size_t pitchVoronoi, int size, int* memLeak) {
    // Assume que o bloco tem a forma (1,32)
    const int tid = threadIdx.x;

    // Variáveis do MegaPixel
    ushort2* voronoiRow;
    __shared__ ushort2 mp_voronoi[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2];

    // Variáveis fila global
    __shared__ ushort2 globalQueueReadBuffer;
    __shared__ ushort2 globalQueueWriteBuffer[5];
    __shared__ unsigned char toWrite;
    bool elementsChanged;
    int elementsChangedCount;
    bool currentActive = true;
    int readCount;

    // Variáveis da propagação dentro do megapixel
    int baseRow, baseColumn;
    int readedCount = 0;

    // Variáveis para a escrita de volta na memória global
    ushort2 val;
    unsigned int assumed;
    unsigned int* val_as_uint = (unsigned int*) &val;


    // Variáveis auxiliares
    int i;
    unsigned long long iters = 0;


    // Incializando a primitiva de sincronização
    if(tid==0) {
        // Incrementa o contador de blocos ativos
        // Assume que a fila global já está inicializada
        atomicAdd(&(gq->activeBlocksCount), 1);
    }


    while((readCount = readFromGlobalQueue(gq, &globalQueueReadBuffer, 1, currentActive)) or gq->activeBlocksCount) {
        // Se nada foi lido o bloco simplesmente continua tentando ler um megapixel válido
        if(readCount == 0) {
            continue;
        }
        readedCount++;

        baseRow = MEGAPIXEL_SIZE*globalQueueReadBuffer.x;
        baseColumn = MEGAPIXEL_SIZE*globalQueueReadBuffer.y;

        // Realizando o fetch da memória global para o megapixel
        // Leitura do bloco principal
        #pragma unroll
        for(i=0; i<MEGAPIXEL_SIZE; i++) {
            voronoiRow = (ushort2*)((char*)voronoi + (baseRow + i) * pitchVoronoi);

            mp_voronoi[i+1][tid+1] = voronoiRow[baseColumn + tid]; 
        }

        // Leitura da lateral superior
        if(globalQueueReadBuffer.x > 0) {
            // Ponteiro para o megapixel acima
            voronoiRow = (ushort2*)((char*)voronoi + (baseRow - 1) * pitchVoronoi);

            mp_voronoi[0][tid+1] = voronoiRow[baseColumn + tid];
        }
        else {
            // Flag de pixel inf
            mp_voronoi[0][tid+1] = make_ushort2(MAX_UNSIGNED_SHORT, MAX_UNSIGNED_SHORT);
        }

        // Leitura da lateral inferior
        if(globalQueueReadBuffer.x < size/MEGAPIXEL_SIZE-1) {
            // Ponteiro para o megapixel abaixo
            voronoiRow = (ushort2*)((char*)voronoi + (baseRow+MEGAPIXEL_SIZE) * pitchVoronoi);

            mp_voronoi[MEGAPIXEL_SIZE+1][tid+1] = voronoiRow[baseColumn + tid];
        }
        else {
            mp_voronoi[MEGAPIXEL_SIZE+1][tid+1] = make_ushort2(MAX_UNSIGNED_SHORT, MAX_UNSIGNED_SHORT);
        }

        // Leitura da lateral direita
        if(globalQueueReadBuffer.y < size/MEGAPIXEL_SIZE-1) {
            voronoiRow = (ushort2*)((char*)voronoi + (baseRow + tid) * pitchVoronoi);

            mp_voronoi[tid+1][MEGAPIXEL_SIZE+1] = voronoiRow[baseColumn + MEGAPIXEL_SIZE];
        }
        else {
            mp_voronoi[tid+1][MEGAPIXEL_SIZE+1] = make_ushort2(MAX_UNSIGNED_SHORT, MAX_UNSIGNED_SHORT);
        }

        // // Leitura da lateral esquerda
        if(globalQueueReadBuffer.y > 0) {
            voronoiRow = (ushort2*)((char*)voronoi + (baseRow + tid) * pitchVoronoi);

            mp_voronoi[tid+1][0] = voronoiRow[baseColumn - 1];
        }
        else {
            mp_voronoi[tid+1][0] = make_ushort2(MAX_UNSIGNED_SHORT, MAX_UNSIGNED_SHORT);
        }

        // Inicializamos as quinas pra evitar problemas durante a propagação
        // Pode ser otimizado para explorar o paralelismo
        if(tid==0) {
            mp_voronoi[0][0] = make_ushort2(MAX_UNSIGNED_SHORT, MAX_UNSIGNED_SHORT);
            mp_voronoi[0][MEGAPIXEL_SIZE+1] = make_ushort2(MAX_UNSIGNED_SHORT, MAX_UNSIGNED_SHORT);
            mp_voronoi[MEGAPIXEL_SIZE+1][0] = make_ushort2(MAX_UNSIGNED_SHORT, MAX_UNSIGNED_SHORT);
            mp_voronoi[MEGAPIXEL_SIZE+1][MEGAPIXEL_SIZE+1] = make_ushort2(MAX_UNSIGNED_SHORT, MAX_UNSIGNED_SHORT);
        }

        // Aqui podemos realizar a computação na memória compartilhada
        // Synchronização para garantia de consistência de memória
        __syncthreads();

        do {
            elementsChanged = XForwardPropagation(mp_voronoi, baseRow, baseColumn);
            elementsChanged = XBackwardPropagation(mp_voronoi, baseRow, baseColumn) or elementsChanged;
            elementsChanged = YUpwardPropagation(mp_voronoi, baseRow, baseColumn) or elementsChanged;
            elementsChanged = YDownwardPropagation(mp_voronoi, baseRow, baseColumn) or elementsChanged;
            iters++;
        } while(elementsChanged);

        // Note que a memória compartilhada já está consistente, pois os métodos anteriores realizam uma sincronização
        // antes de encerrarem
        if(tid==0) {
            toWrite = 0;
        }
        elementsChangedCount = 0;

        // Escrevemos o resultado na memória global
        // Bloco principal
        for(i=0; i<MEGAPIXEL_SIZE; i++) {
            voronoiRow = (ushort2*)((char*)voronoi + (baseRow + i) * pitchVoronoi);

            val = voronoiRow[baseColumn + tid];
            elementsChanged = false;
            
            // Realizamos a escrita
            if(mp_voronoi[i+1][tid+1].x != MAX_UNSIGNED_SHORT) {
                do {
                    assumed = *val_as_uint;
                    elementsChanged = false;

                    if( (val.x == MAX_UNSIGNED_SHORT) or
                        (val.x != MAX_UNSIGNED_SHORT and euclideanDistance(baseRow + i, baseColumn + tid, val.x, val.y)
                        > euclideanDistance(baseRow + i, baseColumn + tid, mp_voronoi[i+1][tid+1].x, mp_voronoi[i+1][tid+1].y))
                    ) {
                        val = mp_voronoi[i+1][tid+1];
                        elementsChanged = true;
                    }
                    // Caso contrário não escrevemos o resultado
                    else {
                        break;
                    }

                    *(val_as_uint) = atomicCAS((unsigned int*)(&voronoiRow[baseColumn + tid]), assumed, *val_as_uint);

                }while(*val_as_uint != assumed);
            }

            elementsChangedCount += elementsChanged;
        }


        elementsChangedCount = __syncthreads_or(elementsChangedCount);
        if(tid==0 and elementsChangedCount) {
            globalQueueWriteBuffer[toWrite++] = globalQueueReadBuffer;
        }

        // Lateral superior
        if(globalQueueReadBuffer.x > 0) {
            voronoiRow = (ushort2*)((char*)voronoi + (baseRow - 1) * pitchVoronoi);

            val = voronoiRow[baseColumn + tid];
            elementsChangedCount = 0;
            
            // Realizamos a escrita
            if(mp_voronoi[0][tid+1].x != MAX_UNSIGNED_SHORT) {
                do {
                    assumed = *val_as_uint;
                    elementsChangedCount = 0;

                    if( (val.x == MAX_UNSIGNED_SHORT) or
                        (val.x != MAX_UNSIGNED_SHORT and euclideanDistance(baseRow - 1, baseColumn + tid, val.x, val.y)
                        > euclideanDistance(baseRow - 1, baseColumn + tid, mp_voronoi[0][tid+1].x, mp_voronoi[0][tid+1].y))
                    ) {
                        val = mp_voronoi[0][tid+1];
                        elementsChangedCount = 1;
                    }
                    // Caso contrário não escrevemos o resultado
                    else {
                        break;
                    }

                    *(val_as_uint) = atomicCAS((unsigned int*)(&voronoiRow[baseColumn + tid]), assumed, *val_as_uint);

                }while(*val_as_uint != assumed);
            }

            
            elementsChangedCount = __syncthreads_or(elementsChangedCount);
            if(tid==0 and elementsChangedCount) {
                globalQueueWriteBuffer[toWrite++] = make_ushort2(globalQueueReadBuffer.x -1, globalQueueReadBuffer.y);
            }
        }

        // Lateral inferior
        if(globalQueueReadBuffer.x < size/MEGAPIXEL_SIZE-1) {
            voronoiRow = (ushort2*)((char*)voronoi + (baseRow + MEGAPIXEL_SIZE) * pitchVoronoi);

            val = voronoiRow[baseColumn + tid];
            elementsChangedCount = 0;
            
            // Realizamos a escrita
            if(mp_voronoi[MEGAPIXEL_SIZE+1][tid+1].x != MAX_UNSIGNED_SHORT) {
                do {
                    assumed = *val_as_uint;
                    elementsChangedCount = 0;

                    if((val.x==MAX_UNSIGNED_SHORT) or
                        (val.x!=MAX_UNSIGNED_SHORT and euclideanDistance(baseRow + MEGAPIXEL_SIZE, baseColumn + tid, val.x, val.y)
                        > euclideanDistance(baseRow + MEGAPIXEL_SIZE, baseColumn + tid, mp_voronoi[MEGAPIXEL_SIZE+1][tid+1].x, mp_voronoi[MEGAPIXEL_SIZE+1][tid+1].y))
                    ) {
                        val = mp_voronoi[MEGAPIXEL_SIZE+1][tid+1];
                        elementsChangedCount = 1;
                    }
                    // Caso contrário não escrevemos o resultado
                    else {
                        break;
                    }

                    *(val_as_uint) = atomicCAS((unsigned int*)(&voronoiRow[baseColumn + tid]), assumed, *val_as_uint);

                }while(*val_as_uint != assumed);
            }

            elementsChangedCount = __syncthreads_or(elementsChangedCount);
            if(tid==0 and elementsChangedCount) {
                globalQueueWriteBuffer[toWrite++] = make_ushort2(globalQueueReadBuffer.x + 1, globalQueueReadBuffer.y);
            }
        }
        
        // Lateral direita
        if(globalQueueReadBuffer.y < size/MEGAPIXEL_SIZE-1) {
            voronoiRow = (ushort2*)((char*)voronoi + (baseRow + tid) * pitchVoronoi);

            val = voronoiRow[baseColumn + MEGAPIXEL_SIZE];
            elementsChangedCount = 0;
            
            // Realizamos a escrita
            if(mp_voronoi[tid+1][MEGAPIXEL_SIZE+1].x != MAX_UNSIGNED_SHORT) {
                do {
                    assumed = *val_as_uint;
                    elementsChangedCount = 0;

                    if((val.x==MAX_UNSIGNED_SHORT) or 
                        (val.x!=MAX_UNSIGNED_SHORT and euclideanDistance(baseRow + tid, baseColumn + MEGAPIXEL_SIZE, val.x, val.y)
                        > euclideanDistance(baseRow + tid, baseColumn + MEGAPIXEL_SIZE, mp_voronoi[tid+1][MEGAPIXEL_SIZE+1].x, mp_voronoi[tid+1][MEGAPIXEL_SIZE+1].y))
                    ) {
                        val = mp_voronoi[tid+1][MEGAPIXEL_SIZE+1];
                        elementsChangedCount = 1;
                    }
                    // Caso contrário não escrevemos o resultado
                    else {
                        break;
                    }

                    *(val_as_uint) = atomicCAS((unsigned int*)(&voronoiRow[baseColumn + MEGAPIXEL_SIZE]), assumed, *val_as_uint);

                }while(*val_as_uint != assumed);
            }

            elementsChangedCount = __syncthreads_or(elementsChangedCount);
            if(tid==0 and elementsChangedCount) {
                globalQueueWriteBuffer[toWrite++] = make_ushort2(globalQueueReadBuffer.x, globalQueueReadBuffer.y + 1);
            }
        }

        // Lateral esquerda
        if(globalQueueReadBuffer.y > 0) {
            voronoiRow = (ushort2*)((char*)voronoi + (baseRow + tid) * pitchVoronoi);

            val = voronoiRow[baseColumn - 1];
            elementsChangedCount = 0;
            
            // Realizamos a escrita
            if(mp_voronoi[tid+1][0].x != MAX_UNSIGNED_SHORT) {
                do {
                    assumed = *val_as_uint;
                    elementsChangedCount = 0;

                    if((val.x == MAX_UNSIGNED_SHORT) or 
                        (val.x != MAX_UNSIGNED_SHORT and euclideanDistance(baseRow + tid, baseColumn - 1, val.x, val.y)
                        > euclideanDistance(baseRow + tid, baseColumn - 1, mp_voronoi[tid+1][0].x, mp_voronoi[tid+1][0].y))
                    ) {
                        val = mp_voronoi[tid+1][0];
                        elementsChangedCount = 1;
                    }
                    // Caso contrário não escrevemos o resultado
                    else {
                        break;
                    }

                    *(val_as_uint) = atomicCAS((unsigned int*)(&voronoiRow[baseColumn - 1]), assumed, *val_as_uint);

                }while(*val_as_uint != assumed);
            }

            elementsChangedCount = __syncthreads_or(elementsChangedCount);
            if(tid==0 and elementsChangedCount) {
                globalQueueWriteBuffer[toWrite++] = make_ushort2(globalQueueReadBuffer.x, globalQueueReadBuffer.y - 1);
            }
        }

        // Sincronizamos para garantia de consistência de memória compartilhada
        __syncthreads();

        // Escrevemos na fila de megapixels os megapixels que devem ser atualizados
        if(toWrite > 0) {
            elementsChangedCount = insertIntoGlobalQueue(gq, globalQueueWriteBuffer, toWrite, true);
            if(elementsChangedCount==GQ_FAILED and tid==0) {
                *memLeak = 1;
            }
        }

    }

}



// Procedimento síncrono
void EuclideanDistanceTransform(ushort2* voronoi, size_t pitchVoronoi, int size) {
    int blockSize = BLOCK_SIZE;
    int gridSize = 96;

    GlobalQueue *gq;

    int megapixelGridSize = size/MEGAPIXEL_SIZE;

    int hMemLeak;
    int* dMemLeak;

    // inicializa o mecanismo de controle de overflow
    cudaMalloc(&dMemLeak, sizeof(int)); 

    // Inicializa a struct da fila global
    cudaMalloc(&gq, sizeof(GlobalQueue));

    // Inicializa os atributos da struct 
    initGlobalQueue(gq);


    do{
        // Inicializa a fila de megapixels e a flag de overflow de memória
        cudaMemset(dMemLeak, 0, sizeof(int));
        initFillGlobalQueue<<<dim3(megapixelGridSize/FILL_BLOCK_SIZE,megapixelGridSize/FILL_BLOCK_SIZE), 
                    dim3(FILL_BLOCK_SIZE, FILL_BLOCK_SIZE)>>>(gq);

        // Dispara o kernel de computação da transformada de distância euclidiana
        EuclideanDistanceTransformKernel<<<gridSize, blockSize>>>(gq, voronoi, pitchVoronoi, size, dMemLeak);
        cudaDeviceSynchronize();

        // Obtem a flag de overflow
        cudaMemcpy(&hMemLeak, dMemLeak, sizeof(int), cudaMemcpyDeviceToHost);

    } while(hMemLeak);

    // Libera a memória alocada
    freeGlobalQueue<<<1,1>>>(gq);
    cudaDeviceSynchronize();


    cudaFree(gq);
}

void __global__ computeDistMap(float* distMap, size_t pitchDistMap, ushort2* voronoi, size_t pitchVoronoi) {
    int myRow = blockIdx.x * blockDim.x + threadIdx.x;
    int myCol = blockIdx.y * blockDim.y + threadIdx.y;

    float* distMapRow = (float*)((char*)distMap + (myRow) * pitchDistMap);
    ushort2* voronoiRow = (ushort2*)((char*)voronoi + (myRow)*pitchVoronoi);

    ushort2 closest = voronoiRow[myCol];

    distMapRow[myCol] = sqrtf((myRow-closest.x)*(myRow-closest.x) + (myCol-closest.y)*(myCol-closest.y));
}

void __global__ extractBaseVoronoi(unsigned char* image, int pitchImage, ushort2* voronoi, size_t pitchVoronoi) {
    int myRow = blockIdx.x * blockDim.x + threadIdx.x;
    int myCol = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned char* imageRow = (unsigned char*)((char*)image + (myRow) * pitchImage);;
    ushort2* voronoiRow = (ushort2*)((char*)voronoi + (myRow)*pitchVoronoi);

    // Aqui o pixel é foreground
    if(imageRow[myCol] > 0) {
        voronoiRow[myCol] = make_ushort2(MAX_UNSIGNED_SHORT, MAX_UNSIGNED_SHORT);
    }
    // Aqui o pixel é background
    else {
        voronoiRow[myCol] = make_ushort2(myRow, myCol);
    }
}