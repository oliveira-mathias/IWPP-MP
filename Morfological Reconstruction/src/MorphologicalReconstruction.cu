#include "MorphologicalReconstruction.cuh"

unsigned int __device__ atomicCASUChar(unsigned char* address, unsigned int index, unsigned char assumed, unsigned char val) {
    unsigned int assumed_word;
    unsigned int val_word;
    unsigned char* val_word_as_uchar;
    unsigned char* assumed_word_as_uchar;
    unsigned int* address_as_uint;

    unsigned int word_index, byte_offset;

    // Endereçamento
    word_index = index/sizeof(unsigned int);
    byte_offset = index - word_index*sizeof(unsigned int);

    address_as_uint = (unsigned int*)(&address[word_index*sizeof(unsigned int)]);
    assumed_word_as_uchar = (unsigned char*) &assumed_word;
    val_word_as_uchar = (unsigned char*) &val_word;
    
    // Leitura das palavras da memória
    assumed_word = *address_as_uint;
    val_word = assumed_word;

    // Atualizando os valores
    assumed_word_as_uchar[byte_offset] = assumed;
    val_word_as_uchar[byte_offset] = val;

    val_word = atomicCAS(address_as_uint, assumed_word, val_word);

    return val_word;
}


bool __device__ XForwardPropagation(unsigned char mp_marker[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2], 
                                unsigned char mp_mask[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2]) 
{
    int tid = threadIdx.x;
    int elementsUpdated = 0;
    int j;
    int myBonusRow;

    for(j=0; j<MEGAPIXEL_SIZE+1; j++){
        if((mp_marker[tid+1][j] > mp_marker[tid+1][j+1]) and (mp_marker[tid+1][j+1] < mp_mask[tid+1][j+1])) {
            mp_marker[tid+1][j+1] = min(mp_marker[tid+1][j], mp_mask[tid+1][j+1]);
            elementsUpdated = 1;
        }
    }

    // Laterais superior e inferior
    myBonusRow = (MEGAPIXEL_SIZE+1)*tid;
    if(tid==0 or tid==1) {
        for(j=0; j<MEGAPIXEL_SIZE+1; j++){
            if(mp_marker[myBonusRow][j] > mp_marker[myBonusRow][j+1] and mp_marker[myBonusRow][j+1] < mp_mask[myBonusRow][j+1]) {
                mp_marker[myBonusRow][j+1] = min(mp_marker[myBonusRow][j], mp_mask[myBonusRow][j+1]);
                elementsUpdated = 1;
            }
        }
    }

    // Verificamos se algum elemento foi atualizado
    elementsUpdated = __syncthreads_or(elementsUpdated);

    return (elementsUpdated != 0);
}

bool __device__ XBackwardPropagation(unsigned char mp_marker[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2], 
                                unsigned char mp_mask[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2]) 
{
    int tid = threadIdx.x;
    int elementsUpdated = 0;
    int j;
    int myBonusRow;

    for(j=MEGAPIXEL_SIZE+1; j>0; j--){
        if(mp_marker[tid+1][j] > mp_marker[tid+1][j-1] and mp_marker[tid+1][j-1] < mp_mask[tid+1][j-1]) {
            mp_marker[tid+1][j-1] = min(mp_marker[tid+1][j], mp_mask[tid+1][j-1]);
            elementsUpdated = 1;
        }
    }

    // Laterais superior e inferior
    myBonusRow = (MEGAPIXEL_SIZE+1)*tid;
    if(tid==0 or tid==1) {
        for(j=MEGAPIXEL_SIZE+1; j>0; j--){
            if(mp_marker[myBonusRow][j] > mp_marker[myBonusRow][j-1] and mp_marker[myBonusRow][j-1] < mp_mask[myBonusRow][j-1]) {
                mp_marker[myBonusRow][j-1] = min(mp_marker[myBonusRow][j], mp_mask[myBonusRow][j-1]);
                elementsUpdated = 1;
            }
        }
    }

    // Verificamos se algum elemento foi atualizado
    elementsUpdated = __syncthreads_or(elementsUpdated);

    return (elementsUpdated != 0);
}

bool __device__ YUpwardPropagation(unsigned char mp_marker[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2], 
                                unsigned char mp_mask[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2])
{
    int tid = threadIdx.x;
    int elementsUpdated = 0;
    int i;
    int myBonusCol;

    for(i=MEGAPIXEL_SIZE+1; i>0; i--){
        if(mp_marker[i][tid+1] > mp_marker[i-1][tid+1] and mp_marker[i-1][tid+1] < mp_mask[i-1][tid+1]) {
            mp_marker[i-1][tid+1] = min(mp_marker[i][tid+1], mp_mask[i-1][tid+1]);
            elementsUpdated = 1;
        }
    }

    // Laterais superior e inferior
    myBonusCol = (MEGAPIXEL_SIZE+1)*tid;
    if(tid==0 or tid==1) {
        for(i=MEGAPIXEL_SIZE+1; i>0; i--){
            if(mp_marker[i][myBonusCol] > mp_marker[i-1][myBonusCol] and mp_marker[i-1][myBonusCol] < mp_mask[i-1][myBonusCol]) {
                mp_marker[i-1][myBonusCol] = min(mp_marker[i][myBonusCol], mp_mask[i-1][myBonusCol]);
                elementsUpdated = 1;
            }
        }
    }

    // Verificamos se algum elemento foi atualizado
    elementsUpdated = __syncthreads_or(elementsUpdated);

    return (elementsUpdated != 0);   
}

// Note que a segunda condição do if pode ser removida
bool __device__ YDownwardPropagation(unsigned char mp_marker[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2], 
                                unsigned char mp_mask[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2])
{
    int tid = threadIdx.x;
    int elementsUpdated = 0;
    int i;
    int myBonusCol;

    for(i=0; i<MEGAPIXEL_SIZE+1; i++){
        if(mp_marker[i][tid+1] > mp_marker[i+1][tid+1] and mp_marker[i+1][tid+1] < mp_mask[i+1][tid+1]) {
            mp_marker[i+1][tid+1] = min(mp_marker[i][tid+1], mp_mask[i+1][tid+1]);
            elementsUpdated = 1;
        }
    }

    // Laterais superior e inferior
    myBonusCol = (MEGAPIXEL_SIZE+1)*tid;
    if(tid==0 or tid==1) {
        for(i=0; i<MEGAPIXEL_SIZE+1; i++){
            if(mp_marker[i][myBonusCol] > mp_marker[i+1][myBonusCol] and mp_marker[i+1][myBonusCol] < mp_mask[i+1][myBonusCol]) {
                mp_marker[i+1][myBonusCol] = min(mp_marker[i][myBonusCol], mp_mask[i+1][myBonusCol]);
                elementsUpdated = 1;
            }
        }
    }

    // Verificamos se algum elemento foi atualizado
    elementsUpdated = __syncthreads_or(elementsUpdated);

    return (elementsUpdated != 0);
}

void __global__ MorphologicalReconstructionKernel(GlobalQueue* gq, unsigned char* marker, unsigned char* mask, 
            size_t pitchMarker, size_t pitchMask, int size, int* memLeak)
{
    // Assume que o bloco tem a forma (1,32)
    int tid = threadIdx.x;

    // Variáveis do MegaPixel
    unsigned char* markerRow;
    unsigned char* maskRow;
    __shared__ unsigned char mp_marker[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2];
    __shared__ unsigned char mp_mask[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2];

    // Variáveis fila global
    __shared__ int2 globalQueueReadBuffer;
    __shared__ int2 globalQueueWriteBuffer[5];
    bool currentActive = true;
    int readCount;

    // Variáveis auxiliares de leitura e escrita
    __shared__ int totalToWrite;

     // Variáveis auxiliares
    int i, j;

    // Variáveis de propagação
    int elementsUpdated;

    // Write back to memory variables
    int rowOffset = tid/8; 
    int wordOffset = (tid%8)*4;
    unsigned int val, assumed;
    unsigned char* valAsChar = (unsigned char*) &val;
    int writeResult;

    int countElementsUpdated;


    // Incializando a fila do bloco e primitiva de sincronização
    if(tid==0) {
        // Incrementa o contador de blocos ativos
        // Assume que a fila global já está inicializada
        atomicAdd(&(gq->activeBlocksCount), 1);
    }

    __syncthreads();

    // Processamento da fila global
    while((readCount = readFromGlobalQueue(gq, &globalQueueReadBuffer, 1, currentActive)) or gq->activeBlocksCount) {
        // Se nada foi lido o bloco simplesmente continua tentando ler um megapixel válido
        if(readCount == 0) {
            continue;
        }

        // Realizando o fetch da memória global para o megapixel
        // Leitura do bloco principal
        #pragma unroll
        for(i=0; i<MEGAPIXEL_SIZE; i++) {
            markerRow = (unsigned char*)((char*)marker + (MEGAPIXEL_SIZE*globalQueueReadBuffer.x + i) * pitchMarker);
            maskRow = (unsigned char*)((char*)mask + (MEGAPIXEL_SIZE*globalQueueReadBuffer.x + i) * pitchMask);

            mp_marker[i+1][tid+1] = markerRow[MEGAPIXEL_SIZE*globalQueueReadBuffer.y + tid];
            mp_mask[i+1][tid+1] = maskRow[MEGAPIXEL_SIZE*globalQueueReadBuffer.y + tid];

        }

        // Leitura da lateral superior
        if(globalQueueReadBuffer.x > 0) {
            // Ponteiro para o megapixel acima
            markerRow = (unsigned char*)((char*)marker + (MEGAPIXEL_SIZE*globalQueueReadBuffer.x - 1) * pitchMarker);
            maskRow = (unsigned char*)((char*)mask + (MEGAPIXEL_SIZE*globalQueueReadBuffer.x - 1) * pitchMask);

            mp_marker[0][tid+1] = markerRow[MEGAPIXEL_SIZE*globalQueueReadBuffer.y + tid];
            mp_mask[0][tid+1] = maskRow[MEGAPIXEL_SIZE*globalQueueReadBuffer.y + tid];
        }
        else {
            // Acho que (0,0) funciona para marcar um pixel que não deve interagir com os demais
            mp_marker[0][tid+1] = 0;
            mp_mask[0][tid+1] = 0;
        }

        // Leitura da lateral inferior
        if(globalQueueReadBuffer.x < (size/MEGAPIXEL_SIZE)-1) {
            // Ponteiro para o megapixel abaixo
            markerRow = (unsigned char*)((char*)marker + (MEGAPIXEL_SIZE*(globalQueueReadBuffer.x + 1)) * pitchMarker);
            maskRow = (unsigned char*)((char*)mask + (MEGAPIXEL_SIZE*(globalQueueReadBuffer.x + 1)) * pitchMask);

            mp_marker[MEGAPIXEL_SIZE+1][tid+1] = markerRow[MEGAPIXEL_SIZE*globalQueueReadBuffer.y + tid];
            mp_mask[MEGAPIXEL_SIZE+1][tid+1] = maskRow[MEGAPIXEL_SIZE*globalQueueReadBuffer.y + tid];
        }
        else {
            mp_marker[MEGAPIXEL_SIZE+1][tid+1] = 0;
            mp_mask[MEGAPIXEL_SIZE+1][tid+1] = 0;
        }

        // Leitura da lateral direita
        if(globalQueueReadBuffer.y < (size/MEGAPIXEL_SIZE)-1) {
            markerRow = (unsigned char*)((char*)marker + (MEGAPIXEL_SIZE*globalQueueReadBuffer.x + tid) * pitchMarker);
            maskRow = (unsigned char*)((char*)mask + (MEGAPIXEL_SIZE*globalQueueReadBuffer.x + tid) * pitchMask);

            mp_marker[tid+1][MEGAPIXEL_SIZE+1] = markerRow[MEGAPIXEL_SIZE*(globalQueueReadBuffer.y+1)];
            mp_mask[tid+1][MEGAPIXEL_SIZE+1] = maskRow[MEGAPIXEL_SIZE*(globalQueueReadBuffer.y+1)];
        }
        else {
            mp_marker[tid+1][MEGAPIXEL_SIZE+1] = 0;
            mp_mask[tid+1][MEGAPIXEL_SIZE+1] = 0;
        }

        // Leitura da lateral esquerda
        if(globalQueueReadBuffer.y > 0) {
            markerRow = (unsigned char*)((char*)marker + (MEGAPIXEL_SIZE*globalQueueReadBuffer.x + tid) * pitchMarker);
            maskRow = (unsigned char*)((char*)mask + (MEGAPIXEL_SIZE*globalQueueReadBuffer.x + tid) * pitchMask);

            mp_marker[tid+1][0] = markerRow[MEGAPIXEL_SIZE*globalQueueReadBuffer.y - 1];
            mp_mask[tid+1][0] = maskRow[MEGAPIXEL_SIZE*globalQueueReadBuffer.y - 1];
        }
        else {
            mp_marker[tid+1][0] = 0;
            mp_mask[tid+1][0] = 0;
        }

        // Inicializamos as quinas pra evitar problemas durante a propagação
        // Pode ser otimizado para explorar o paralelismo
        if(tid==0) {
            mp_marker[0][0] = 0;
            mp_marker[0][MEGAPIXEL_SIZE+1] = 0;
            mp_marker[MEGAPIXEL_SIZE+1][0] = 0;
            mp_marker[MEGAPIXEL_SIZE+1][MEGAPIXEL_SIZE+1] = 0;
        }
        if(tid==1) {
            mp_mask[0][0] = 0;
            mp_mask[0][MEGAPIXEL_SIZE+1] = 0;
            mp_mask[MEGAPIXEL_SIZE+1][0] = 0;
            mp_mask[MEGAPIXEL_SIZE+1][MEGAPIXEL_SIZE+1] = 0;
        }

        // Aqui podemos realizar a computação na memória compartilhada
        // Synchronização para garantia de consistência
        __syncthreads();

        // Análogo a propagação SR
        do{
            elementsUpdated = XForwardPropagation(mp_marker, mp_mask);
            elementsUpdated = YDownwardPropagation(mp_marker, mp_mask) or elementsUpdated;
            elementsUpdated = XBackwardPropagation(mp_marker, mp_mask) or elementsUpdated;
            elementsUpdated = YUpwardPropagation(mp_marker, mp_mask) or elementsUpdated;
        }while(elementsUpdated);


        // Escrevemos o resultado de volta na memória global
        if(tid==0) {
            totalToWrite = 0;
        }

        // Bloco principal
        countElementsUpdated = 0;
        #pragma unroll
        for(i=0; i<MEGAPIXEL_SIZE; i+=4) {
            markerRow = (unsigned char*)((char*)marker + (MEGAPIXEL_SIZE*globalQueueReadBuffer.x + i + rowOffset) * pitchMarker);

            // Carregamos o conteudo atual da memória global
            val = *((unsigned int*)(&(markerRow[globalQueueReadBuffer.y*MEGAPIXEL_SIZE + wordOffset])));
            do{
                elementsUpdated = 0;
                assumed = val;
                // Verificamos se alguma alteração foi feita
                // Note que um int comporta 4 unsigned char
                for(j=0; j<4; j++) {
                    if(mp_marker[i+rowOffset+1][wordOffset +j+1] > valAsChar[j]) {
                        valAsChar[j] = mp_marker[i+rowOffset+1][wordOffset +j+1];
                        elementsUpdated = 1;
                    }
                }

                // Saimos do loop caso nenhum elemento dessa palavra precise ser atualizado 
                if(!elementsUpdated) {
                    break;
                }

                // Realizamos a escrita na memória global
                val = atomicCAS((unsigned int*)(&(markerRow[globalQueueReadBuffer.y*MEGAPIXEL_SIZE + wordOffset])), assumed, val);
            } while(val != assumed);
            countElementsUpdated += elementsUpdated;
        }

        // Verificamos se esse megapixel deve ser reinserido na fila de megapixels
        countElementsUpdated = __syncthreads_or(countElementsUpdated);
        if(tid==0 and countElementsUpdated) {
            globalQueueWriteBuffer[totalToWrite++] = globalQueueReadBuffer;
        }

        // Lateral Superior
        elementsUpdated = 0;
        if(tid<8 and globalQueueReadBuffer.x > 0) {
            // Olhamos para a última linha do bloco de cima
            markerRow = (unsigned char*)((char*)marker + (MEGAPIXEL_SIZE*globalQueueReadBuffer.x - 1) * pitchMarker);

            // Carregamos o conteudo atual da memória global
            val = *((unsigned int*)(&(markerRow[globalQueueReadBuffer.y*MEGAPIXEL_SIZE + wordOffset])));
            do{
                elementsUpdated = 0;
                assumed = val;
                // Verificamos se alguma alteração foi feita
                // Note que um int comporta 4 unsigned char
                for(j=0; j<4; j++) {
                    if(mp_marker[0][wordOffset +j+1] > valAsChar[j]) {
                        valAsChar[j] = mp_marker[0][wordOffset +j+1];
                        elementsUpdated = 1;
                    }
                }

                // Saimos do loop caso nenhum elemento dessa palavra precise ser atualizado 
                if(!elementsUpdated) {
                    break;
                }

                // Realizamos a escrita na memória global
                val = atomicCAS((unsigned int*)(&(markerRow[globalQueueReadBuffer.y*MEGAPIXEL_SIZE + wordOffset])), assumed, val);
            } while(val != assumed);

        }

        // ------------collective verification
        countElementsUpdated = __syncthreads_or(elementsUpdated);

        if(tid==0 and countElementsUpdated) {
            globalQueueWriteBuffer[totalToWrite++] = make_int2(globalQueueReadBuffer.x-1, globalQueueReadBuffer.y);
        }

        // Lateral inferior
        elementsUpdated = 0;
        if(tid<8 and globalQueueReadBuffer.x < (size/MEGAPIXEL_SIZE)-1) {
            // Olhamos para a primeira linha do bloco de baixo
            markerRow = (unsigned char*)((char*)marker + (MEGAPIXEL_SIZE*(globalQueueReadBuffer.x+1)) * pitchMarker);

            // Carregamos o conteudo atual da memória global
            val = *((unsigned int*)(&(markerRow[globalQueueReadBuffer.y*MEGAPIXEL_SIZE + wordOffset])));
            do{
                elementsUpdated = 0;
                assumed = val;
                // Verificamos se alguma alteração foi feita
                // Note que um int comporta 4 unsigned char
                for(j=0; j<4; j++) {
                    if(mp_marker[MEGAPIXEL_SIZE+1][wordOffset +j+1] > valAsChar[j]) {
                        valAsChar[j] = mp_marker[MEGAPIXEL_SIZE+1][wordOffset +j+1];
                        elementsUpdated = 1;
                    }
                }

                // Saimos do loop caso nenhum elemento dessa palavra precise ser atualizado 
                if(!elementsUpdated) {
                    break;
                }

                // Realizamos a escrita na memória global
                val = atomicCAS((unsigned int*)(&(markerRow[globalQueueReadBuffer.y*MEGAPIXEL_SIZE + wordOffset])), assumed, val);
            } while(val != assumed);

        }
        // ------------collective verification
        countElementsUpdated = __syncthreads_or(elementsUpdated);

        if(tid==0 and countElementsUpdated) {
            globalQueueWriteBuffer[totalToWrite++] = make_int2(globalQueueReadBuffer.x+1, globalQueueReadBuffer.y);
        }

        // // Laterais esquerda e direita
        // Lateral esquerda
        if(globalQueueReadBuffer.y > 0) {
            // Cada thread olha para a sua linha no megapixel
            markerRow = (unsigned char*)((char*)marker + (MEGAPIXEL_SIZE*globalQueueReadBuffer.x + tid) * pitchMarker);
            
            // Olhamos para o megapixel à esquera
            // Novamente trabalhamos com 4 bytes
            val = *((unsigned int*)(&(markerRow[globalQueueReadBuffer.y*MEGAPIXEL_SIZE - 4])));
            do {
                elementsUpdated = 0;
                assumed = val;
                // Verificamos se alguma alteração foi feita
                // Note que um int comporta 4 unsigned char
                if(mp_marker[tid+1][0] > valAsChar[3]) {
                    valAsChar[3] = mp_marker[tid+1][0];
                    elementsUpdated = 1;
                }
                else {
                    break;
                }

                // Realizamos a escrita na memória global
                val = atomicCAS((unsigned int*)(&(markerRow[globalQueueReadBuffer.y*MEGAPIXEL_SIZE - 4])), assumed, val);
            } while(assumed != val);

            // ------------collective verification
            countElementsUpdated = __syncthreads_or(elementsUpdated);

            if(tid==0 and countElementsUpdated) {
                globalQueueWriteBuffer[totalToWrite++] = make_int2(globalQueueReadBuffer.x, globalQueueReadBuffer.y-1);
            }
        }

        // Lateral direita
        if(globalQueueReadBuffer.y < (size/MEGAPIXEL_SIZE)-1) {
            // Cada thread olha para a sua linha no megapixel
            markerRow = (unsigned char*)((char*)marker + (MEGAPIXEL_SIZE*globalQueueReadBuffer.x + tid) * pitchMarker);
            
            // Olhamos para o megapixel à esquera
            // Novamente trabalhamos com 4 bytes
            val = *((unsigned int*)(&(markerRow[(globalQueueReadBuffer.y+1)*MEGAPIXEL_SIZE])));
            do {
                elementsUpdated = 0;
                assumed = val;
                // Verificamos se alguma alteração foi feita
                // Note que um int comporta 4 unsigned char
                if(mp_marker[tid+1][MEGAPIXEL_SIZE+1] > valAsChar[0]) {
                    valAsChar[0] = mp_marker[tid+1][MEGAPIXEL_SIZE+1];
                    elementsUpdated = 1;
                }
                else {
                    break;
                }

                // Realizamos a escrita na memória global
                val = atomicCAS((unsigned int*)(&(markerRow[(globalQueueReadBuffer.y+1)*MEGAPIXEL_SIZE])), assumed, val);
            } while(assumed != val);

            // ------------collective verification
            countElementsUpdated = __syncthreads_or(elementsUpdated);

            if(tid==0 and countElementsUpdated) {
                globalQueueWriteBuffer[totalToWrite++] = make_int2(globalQueueReadBuffer.x, globalQueueReadBuffer.y+1);
            }
        }


        // Escrevendo os megapixels de volta na fila de megapixels
        // Para garantir a consistência da memória compartilhada
        __syncthreads();


        if(totalToWrite > 0) {
            writeResult = insertIntoGlobalQueue(gq, globalQueueWriteBuffer, totalToWrite, true);
            if(writeResult==GQ_FAILED and tid==0) {
                *memLeak = 1;
            }
        }
    }

}

// Propagação da imagem inteira
// Assume que a imagem é quadrada e que o tamanho da imagem é múltiplo de 32
// Assume que os blocos tem 32 threads
void __global__ XForwardPropagationEntireImage(unsigned char* marker, unsigned char* mask, size_t pitchMarker, size_t pitchMask, int size) {
    const int baseRow = blockIdx.x * MEGAPIXEL_SIZE;
    unsigned char* markerRow;
    unsigned char* maskRow;

    int tid = threadIdx.x;

    // Mega pixel
    __shared__ unsigned char mp_marker[MEGAPIXEL_SIZE][MEGAPIXEL_SIZE+1];
    __shared__ unsigned char mp_mask[MEGAPIXEL_SIZE][MEGAPIXEL_SIZE+1];

    int col, i;

    // inicializamos a coluna de carry do megapixel
    mp_marker[tid][0] = 0;

    for(col=0; col<size; col+=MEGAPIXEL_SIZE) {
        // Realizamos a leitura do megapixel
        // Linha por linha
        for(i=0; i<MEGAPIXEL_SIZE; i++) {
            markerRow = (unsigned char*)((char*)marker + (baseRow + i)*pitchMarker);
            maskRow = (unsigned char*)((char*)mask + (baseRow + i)*pitchMask);
            mp_marker[i][tid+1] = markerRow[col + tid];
            mp_mask[i][tid+1] = maskRow[col + tid];
        }

        // Sincronizamos para garantia de cosnsitência da memória compartilhada
        __syncthreads();

        // Realizamos a propagação, cada thread pega uma linha
        // Note que como essa rotina vai ser executada apenas uma vez, podemos trocar uma comparação por uma escrita
        for(i=0; i<MEGAPIXEL_SIZE; i++) {
            if(mp_marker[tid][i] > mp_marker[tid][i+1]) {
                mp_marker[tid][i+1] = min(mp_marker[tid][i], mp_mask[tid][i+1]);
            }
        }

        // Sincronizamos para garantia de cosnsitência da memória compartilhada
        __syncthreads();

        // Escrevemos o resultado de volta na memória global
        for(i=0; i<MEGAPIXEL_SIZE; i++) {
            markerRow = (unsigned char*)((char*)marker + (baseRow + i)*pitchMarker);
            markerRow[col + tid] = mp_marker[i][tid+1];
        }

        // Copiamos a coluna de carry para continuar a propagação
        mp_marker[tid][0] = mp_marker[tid][MEGAPIXEL_SIZE];
    }
}

// Assume que a imagem é quadrada e que o tamanho da imagem é múltiplo de 32
// Assume que os blocos tem 32 threads
void __global__ XBackwardPropagationEntireImage(unsigned char* marker, unsigned char* mask, size_t pitchMarker, size_t pitchMask, int size) {
    const int baseRow = blockIdx.x * MEGAPIXEL_SIZE;
    unsigned char* markerRow;
    unsigned char* maskRow;

    int tid = threadIdx.x;

    // Mega pixel
    __shared__ unsigned char mp_marker[MEGAPIXEL_SIZE][MEGAPIXEL_SIZE+1];
    __shared__ unsigned char mp_mask[MEGAPIXEL_SIZE][MEGAPIXEL_SIZE+1];

    int col, i;

    // inicializamos a coluna de carry do megapixel
    mp_marker[tid][MEGAPIXEL_SIZE] = 0;

    for(col=size-MEGAPIXEL_SIZE; col>=0; col-=MEGAPIXEL_SIZE) {
        // Realizamos a leitura do megapixel
        // Linha por linha
        for(i=0; i<MEGAPIXEL_SIZE; i++) {
            markerRow = (unsigned char*)((char*)marker + (baseRow + i)*pitchMarker);
            maskRow = (unsigned char*)((char*)mask + (baseRow + i)*pitchMask);
            mp_marker[i][tid] = markerRow[col + tid];
            mp_mask[i][tid] = maskRow[col + tid];
        }

        // Sincronizamos para garantia de cosnsitência da memória compartilhada
        __syncthreads();

        // Realizamos a propagação, cada thread pega uma linha
        // Note que como essa rotina vai ser executada apenas uma vez, podemos trocar uma comparação por uma escrita
        for(i=MEGAPIXEL_SIZE; i>0; i--) {
            if(mp_marker[tid][i] > mp_marker[tid][i-1]) {
                mp_marker[tid][i-1] = min(mp_marker[tid][i], mp_mask[tid][i-1]);
            }
        }

        // Sincronizamos para garantia de cosnsitência da memória compartilhada
        __syncthreads();

        // Escrevemos o resultado de volta na memória global
        for(i=0; i<MEGAPIXEL_SIZE; i++) {
            markerRow = (unsigned char*)((char*)marker + (baseRow + i)*pitchMarker);
            markerRow[col + tid] = mp_marker[i][tid];
        }

        // Copiamos a coluna de carry para continuar a propagação
        mp_marker[tid][MEGAPIXEL_SIZE] = mp_marker[tid][0];
    }
}

// Assume que a imagem é quadrada e que o tamanho da imagem é múltiplo de 32
// Assume que os blocos tem 32 threads
void __global__ YUpwardPropagationEntireImage(unsigned char* marker, unsigned char* mask, size_t pitchMarker, size_t pitchMask, int size) {
    const int baseCol = blockIdx.y * MEGAPIXEL_SIZE;
    unsigned char* markerRow;
    unsigned char* maskRow;

    int tid = threadIdx.x;

    // Mega pixel
    __shared__ unsigned char mp_marker[MEGAPIXEL_SIZE+1][MEGAPIXEL_SIZE];
    __shared__ unsigned char mp_mask[MEGAPIXEL_SIZE+1][MEGAPIXEL_SIZE];

    int row, i;

    // inicializamos a coluna de carry do megapixel
    mp_marker[MEGAPIXEL_SIZE][tid] = 0;

    for(row=size-MEGAPIXEL_SIZE; row>=0; row-=MEGAPIXEL_SIZE) {
        // Realizamos a leitura do megapixel
        // Linha por linha
        for(i=0; i<MEGAPIXEL_SIZE; i++) {
            markerRow = (unsigned char*)((char*)marker + (row + i)*pitchMarker);
            maskRow = (unsigned char*)((char*)mask + (row + i)*pitchMask);
            mp_marker[i][tid] = markerRow[baseCol + tid];
            mp_mask[i][tid] = maskRow[baseCol + tid];
        }

        // Sincronizamos para garantia de cosnsitência da memória compartilhada
        __syncthreads();

        // Realizamos a propagação, cada thread pega uma linha
        // Note que como essa rotina vai ser executada apenas uma vez, podemos trocar uma comparação por uma escrita
        for(i=MEGAPIXEL_SIZE; i>0; i--) {
            if(mp_marker[i][tid] > mp_marker[i-1][tid]) {
                mp_marker[i-1][tid] = min(mp_marker[i][tid], mp_mask[i-1][tid]);
            }
        }

        // Sincronizamos para garantia de cosnsitência da memória compartilhada
        __syncthreads();

        // Escrevemos o resultado de volta na memória global
        for(i=0; i<MEGAPIXEL_SIZE; i++) {
            markerRow = (unsigned char*)((char*)marker + (row + i)*pitchMarker);
            markerRow[baseCol + tid] = mp_marker[i][tid];
        }

        // Copiamos a coluna de carry para continuar a propagação
        mp_marker[MEGAPIXEL_SIZE][tid] = mp_marker[0][tid];
    }
}

void __global__ YDownwardPropagationEntireImage(unsigned char* marker, unsigned char* mask, size_t pitchMarker, size_t pitchMask, int size) {
    const int baseCol = blockIdx.y * MEGAPIXEL_SIZE;
    unsigned char* markerRow;
    unsigned char* maskRow;

    int tid = threadIdx.x;

    // Mega pixel
    __shared__ unsigned char mp_marker[MEGAPIXEL_SIZE+1][MEGAPIXEL_SIZE];
    __shared__ unsigned char mp_mask[MEGAPIXEL_SIZE+1][MEGAPIXEL_SIZE];

    int row, i;

    // inicializamos a coluna de carry do megapixel
    mp_marker[0][tid] = 0;

    for(row=0; row<size; row+=MEGAPIXEL_SIZE) {
        // Realizamos a leitura do megapixel
        // Linha por linha
        for(i=0; i<MEGAPIXEL_SIZE; i++) {
            markerRow = (unsigned char*)((char*)marker + (row + i)*pitchMarker);
            maskRow = (unsigned char*)((char*)mask + (row + i)*pitchMask);
            mp_marker[i+1][tid] = markerRow[baseCol + tid];
            mp_mask[i+1][tid] = maskRow[baseCol + tid];
        }

        // Sincronizamos para garantia de cosnsitência da memória compartilhada
        __syncthreads();

        // Realizamos a propagação, cada thread pega uma linha
        // Note que como essa rotina vai ser executada apenas uma vez, podemos trocar uma comparação por uma escrita
        for(i=0; i<MEGAPIXEL_SIZE; i++) {
            if(mp_marker[i][tid] > mp_marker[i+1][tid]) {
                mp_marker[i+1][tid] = min(mp_marker[i][tid], mp_mask[i+1][tid]);
            }
        }

        // Sincronizamos para garantia de cosnsitência da memória compartilhada
        __syncthreads();

        // Escrevemos o resultado de volta na memória global
        for(i=0; i<MEGAPIXEL_SIZE; i++) {
            markerRow = (unsigned char*)((char*)marker + (row + i)*pitchMarker);
            markerRow[baseCol + tid] = mp_marker[i+1][tid];
        }

        // Copiamos a coluna de carry para continuar a propagação
        mp_marker[0][tid] = mp_marker[MEGAPIXEL_SIZE][tid];
    }
}


void MorphologicalReconstruction(unsigned char* marker, unsigned char* mask, 
            size_t pitchMarker, size_t pitchMask, int size)
{
    int blockSize = 32;
    int gridSize = 97;

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

        // Varreduras raster e antiraster
        XForwardPropagationEntireImage<<<dim3(size/MEGAPIXEL_SIZE, 1), 32>>>(marker, mask, pitchMarker, pitchMask, size);
        YDownwardPropagationEntireImage<<<dim3(1, size/MEGAPIXEL_SIZE), 32>>>(marker, mask, pitchMarker, pitchMask, size);
        XBackwardPropagationEntireImage<<<dim3(size/MEGAPIXEL_SIZE, 1), 32>>>(marker, mask, pitchMarker, pitchMask, size);
        YUpwardPropagationEntireImage<<<dim3(1, size/MEGAPIXEL_SIZE), 32>>>(marker, mask, pitchMarker, pitchMask, size);

        // Dispara o kernel de computação da reconstrução morfológica
        MorphologicalReconstructionKernel<<<gridSize, blockSize>>>(gq, marker, mask, pitchMarker, pitchMask, size, dMemLeak);
        cudaDeviceSynchronize();

        // Obtem a flag de overflow
        cudaMemcpy(&hMemLeak, dMemLeak, sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        printf("MemLeak: %d\n", hMemLeak);
    } while(hMemLeak);

    // Libera a memória alocada
    freeGlobalQueue<<<1,1>>>(gq);
    cudaDeviceSynchronize();


    cudaFree(gq);
}


// Após a execução desse kernel, todo elemento do marker é menor ou igual que o elemento correspondente do mask
void __global__ clipImage(unsigned char* marker, unsigned char* mask, size_t pitchMarker, size_t pitchMask) {
    int myRow = blockIdx.x * blockDim.x + threadIdx.x;
    int myCol = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned char* markerRow =  (unsigned char*)((char*)marker + (myRow) * pitchMarker);
    unsigned char* maskRow =  (unsigned char*)((char*)mask + (myRow) * pitchMask);

    markerRow[myCol] = min(markerRow[myCol], maskRow[myCol]);

}