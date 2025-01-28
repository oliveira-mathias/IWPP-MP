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

// Pode apresentar overflow na fila de megapixels se a imagem possuir pelo menos 65536 megapixels de altura/largura
void __global__ MorphologicalReconstructionKernel(GlobalQueue* gq, volatile unsigned char* marker, unsigned char* mask, 
            size_t pitchMarker, size_t pitchMask, int size, int* memLeak)
{
    // Assume que o bloco tem a forma (1,32)
    int tid = threadIdx.x;
    int blockId = blockIdx.x;
    int numBlocks = gridDim.x;

    // Variáveis do MegaPixel
    volatile unsigned char* markerRow;
    unsigned char* maskRow;
    __shared__ unsigned char mp_marker[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2];
    __shared__ unsigned char mp_mask[MEGAPIXEL_SIZE+2][MEGAPIXEL_SIZE+2];

    // Variáveis fila global
    // 1KB de buffer de leitura e de escrita
    __shared__ ushort2 readBuffer[BUFFER_SIZE];
    __shared__ ushort2 writeBuffer[BUFFER_SIZE];

    // Variáveis auxiliares de leitura e escrita
    __shared__ int writeSize;
    __shared__ int readOffset;
    int readedAmount;
    int writtenAmount;
    int currentRegister;

    // Fila de leitura
    ushort2* readGlobalQueue;
    int readQueueSize;

     // Variáveis auxiliares
    int i, j;

    // Variáveis de propagação
    int elementsUpdated;


    // Write back to memory variables
    int rowOffset = tid/(MEGAPIXEL_SIZE/4); 
    int wordOffset = (tid%(MEGAPIXEL_SIZE/4))*4;
    unsigned int val, assumed;
    unsigned char* valAsChar = (unsigned char*) &val;

    int countElementsUpdated;

    // Incializando a fila do bloco e primitiva de sincronização
    if(tid==0) {
        writeSize = 0;
        readOffset = 0;
    }

    // Inicializando a fila de leitura
    const int quotient = (gq->readQueueSize)/numBlocks;
    const int remainder = (gq->readQueueSize)%numBlocks;
    
    // Tamanho da fila
    readQueueSize = quotient;
    if(blockId < remainder) {
        readQueueSize++;
    }

    // Offset
    int offset = min(remainder, blockId)*(quotient + 1);
    if(blockId > remainder) {
        offset += (blockId - remainder)*quotient;
    }

    // Ponteiro para a fila
    readGlobalQueue = &(gq->readQueue[offset]);

    // Sincronia para garantia de consistência de memória compartilhada
    __syncthreads();

    
    // Processamento da fila global
    while(readedAmount = readFromGlobalQueue(readGlobalQueue, readOffset, readQueueSize, readBuffer, BUFFER_SIZE)) {

        // Processamos os registros lidos da fila global
        for(currentRegister=0; currentRegister<readedAmount; currentRegister++) {


            // Realizando o fetch da memória global para o megapixel
            // Leitura do bloco principal
            #pragma unroll
            for(i=0; i<MEGAPIXEL_SIZE; i++) {
                markerRow = (volatile unsigned char*)((char*)marker + (MEGAPIXEL_SIZE*readBuffer[currentRegister].x + i) * pitchMarker);
                maskRow = (unsigned char*)((char*)mask + (MEGAPIXEL_SIZE*readBuffer[currentRegister].x + i) * pitchMask);

                mp_marker[i+1][tid+1] = markerRow[MEGAPIXEL_SIZE*readBuffer[currentRegister].y + tid];
                mp_mask[i+1][tid+1] = maskRow[MEGAPIXEL_SIZE*readBuffer[currentRegister].y + tid];

            }

            // Leitura da lateral superior
            if(readBuffer[currentRegister].x > 0) {
                // Ponteiro para o megapixel acima
                markerRow = (volatile unsigned char*)((char*)marker + (MEGAPIXEL_SIZE*readBuffer[currentRegister].x - 1) * pitchMarker);
                maskRow = (unsigned char*)((char*)mask + (MEGAPIXEL_SIZE*readBuffer[currentRegister].x - 1) * pitchMask);

                mp_marker[0][tid+1] = markerRow[MEGAPIXEL_SIZE*readBuffer[currentRegister].y + tid];
                mp_mask[0][tid+1] = maskRow[MEGAPIXEL_SIZE*readBuffer[currentRegister].y + tid];
            }
            else {
                // Acho que (0,0) funciona para marcar um pixel que não deve interagir com os demais
                mp_marker[0][tid+1] = 0;
                mp_mask[0][tid+1] = 0;
            }

            // Leitura da lateral inferior
            if(readBuffer[currentRegister].x < (size/MEGAPIXEL_SIZE)-1) {
                // Ponteiro para o megapixel abaixo
                markerRow = (volatile unsigned char*)((char*)marker + (MEGAPIXEL_SIZE*(readBuffer[currentRegister].x + 1)) * pitchMarker);
                maskRow = (unsigned char*)((char*)mask + (MEGAPIXEL_SIZE*(readBuffer[currentRegister].x + 1)) * pitchMask);

                mp_marker[MEGAPIXEL_SIZE+1][tid+1] = markerRow[MEGAPIXEL_SIZE*readBuffer[currentRegister].y + tid];
                mp_mask[MEGAPIXEL_SIZE+1][tid+1] = maskRow[MEGAPIXEL_SIZE*readBuffer[currentRegister].y + tid];
            }
            else {
                mp_marker[MEGAPIXEL_SIZE+1][tid+1] = 0;
                mp_mask[MEGAPIXEL_SIZE+1][tid+1] = 0;
            }

            // Leitura da lateral direita
            if(readBuffer[currentRegister].y < (size/MEGAPIXEL_SIZE)-1) {
                markerRow = (volatile unsigned char*)((char*)marker + (MEGAPIXEL_SIZE*readBuffer[currentRegister].x + tid) * pitchMarker);
                maskRow = (unsigned char*)((char*)mask + (MEGAPIXEL_SIZE*readBuffer[currentRegister].x + tid) * pitchMask);

                mp_marker[tid+1][MEGAPIXEL_SIZE+1] = markerRow[MEGAPIXEL_SIZE*(readBuffer[currentRegister].y+1)];
                mp_mask[tid+1][MEGAPIXEL_SIZE+1] = maskRow[MEGAPIXEL_SIZE*(readBuffer[currentRegister].y+1)];
            }
            else {
                mp_marker[tid+1][MEGAPIXEL_SIZE+1] = 0;
                mp_mask[tid+1][MEGAPIXEL_SIZE+1] = 0;
            }

            // Leitura da lateral esquerda
            if(readBuffer[currentRegister].y > 0) {
                markerRow = (volatile unsigned char*)((char*)marker + (MEGAPIXEL_SIZE*readBuffer[currentRegister].x + tid) * pitchMarker);
                maskRow = (unsigned char*)((char*)mask + (MEGAPIXEL_SIZE*readBuffer[currentRegister].x + tid) * pitchMask);

                mp_marker[tid+1][0] = markerRow[MEGAPIXEL_SIZE*readBuffer[currentRegister].y - 1];
                mp_mask[tid+1][0] = maskRow[MEGAPIXEL_SIZE*readBuffer[currentRegister].y - 1];
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
            // Bloco principal
            countElementsUpdated = 0;
            #pragma unroll
            for(i=0; i<MEGAPIXEL_SIZE; i+=4) {
                markerRow = (volatile unsigned char*)((char*)marker + (MEGAPIXEL_SIZE*readBuffer[currentRegister].x + i + rowOffset) * pitchMarker);

                // Carregamos o conteudo atual da memória global
                val = *((unsigned int*)(&(markerRow[readBuffer[currentRegister].y*MEGAPIXEL_SIZE + wordOffset])));
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
                    val = atomicCAS((unsigned int*)(&(markerRow[readBuffer[currentRegister].y*MEGAPIXEL_SIZE + wordOffset])), assumed, val);
                } while(val != assumed);
                countElementsUpdated += elementsUpdated;
            }


            // Verificamos se esse megapixel deve ser reinserido na fila de megapixels
            countElementsUpdated = __syncthreads_or(countElementsUpdated);
            if(tid==0 and countElementsUpdated) {
                writeBuffer[writeSize++] = readBuffer[currentRegister];
            }

            // Lateral Superior
            // Resta a flag de quem nao entra no loop
            elementsUpdated = 0;
            if(tid<(MEGAPIXEL_SIZE/4) and readBuffer[currentRegister].x > 0) {
                // Olhamos para a última linha do bloco de cima
                markerRow = (volatile unsigned char*)((char*)marker + (MEGAPIXEL_SIZE*readBuffer[currentRegister].x - 1) * pitchMarker);

                // Carregamos o conteudo atual da memória global
                val = *((unsigned int*)(&(markerRow[readBuffer[currentRegister].y*MEGAPIXEL_SIZE + wordOffset])));
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
                    val = atomicCAS((unsigned int*)(&(markerRow[readBuffer[currentRegister].y*MEGAPIXEL_SIZE + wordOffset])), assumed, val);
                } while(val != assumed);

            }


            // ------------collective verification
            countElementsUpdated = __syncthreads_or(elementsUpdated);
            if(tid==0 and countElementsUpdated) {
                writeBuffer[writeSize++] = readBuffer[currentRegister];
                writeBuffer[writeSize - 1].x -= 1;
                // globalQueueWriteBuffer[totalToWrite++] = make_int2(globalQueueReadBuffer.x-1, globalQueueReadBuffer.y);
            }

                  
            // Lateral inferior
            // Reseta a flag de quem nao entra no loop
            elementsUpdated = 0;
            if(tid<(MEGAPIXEL_SIZE/4) and readBuffer[currentRegister].x < (size/MEGAPIXEL_SIZE)-1) {
                // Olhamos para a primeira linha do bloco de baixo
                markerRow = (volatile unsigned char*)((char*)marker + (MEGAPIXEL_SIZE*(readBuffer[currentRegister].x+1)) * pitchMarker);

                // Carregamos o conteudo atual da memória global
                val = *((unsigned int*)(&(markerRow[readBuffer[currentRegister].y*MEGAPIXEL_SIZE + wordOffset])));
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
                    val = atomicCAS((unsigned int*)(&(markerRow[readBuffer[currentRegister].y*MEGAPIXEL_SIZE + wordOffset])), assumed, val);
                } while(val != assumed);

            }

            // ------------collective verification
            countElementsUpdated = __syncthreads_or(elementsUpdated);

            if(tid==0 and countElementsUpdated) {
                writeBuffer[writeSize++] = readBuffer[currentRegister];
                writeBuffer[writeSize - 1].x += 1;
                // globalQueueWriteBuffer[totalToWrite++] = make_int2(globalQueueReadBuffer.x+1, globalQueueReadBuffer.y);
            }

            // // Laterais esquerda e direita
            // Lateral esquerda
            if(readBuffer[currentRegister].y > 0) {
                // Cada thread olha para a sua linha no megapixel
                markerRow = (volatile unsigned char*)((char*)marker + (MEGAPIXEL_SIZE*readBuffer[currentRegister].x + tid) * pitchMarker);
                
                // Olhamos para o megapixel à esquera
                // Novamente trabalhamos com 4 bytes
                val = *((unsigned int*)(&(markerRow[readBuffer[currentRegister].y*MEGAPIXEL_SIZE - 4])));
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
                    val = atomicCAS((unsigned int*)(&(markerRow[readBuffer[currentRegister].y*MEGAPIXEL_SIZE - 4])), assumed, val);
                } while(assumed != val);

                // ------------collective verification
                countElementsUpdated = __syncthreads_or(elementsUpdated);

                if(tid==0 and countElementsUpdated) {
                    writeBuffer[writeSize++] = readBuffer[currentRegister];
                    writeBuffer[writeSize - 1].y -= 1;
                    // globalQueueWriteBuffer[totalToWrite++] = make_int2(globalQueueReadBuffer.x, globalQueueReadBuffer.y-1);
                }
            }

            // Lateral direita
            if(readBuffer[currentRegister].y < (size/MEGAPIXEL_SIZE)-1) {
                // Cada thread olha para a sua linha no megapixel
                markerRow = (volatile unsigned char*)((char*)marker + (MEGAPIXEL_SIZE*readBuffer[currentRegister].x + tid) * pitchMarker);
                
                // Olhamos para o megapixel à esquera
                // Novamente trabalhamos com 4 bytes
                val = *((unsigned int*)(&(markerRow[(readBuffer[currentRegister].y+1)*MEGAPIXEL_SIZE])));
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
                    val = atomicCAS((unsigned int*)(&(markerRow[(readBuffer[currentRegister].y+1)*MEGAPIXEL_SIZE])), assumed, val);
                } while(assumed != val);

                // ------------collective verification
                countElementsUpdated = __syncthreads_or(elementsUpdated);

                if(tid==0 and countElementsUpdated) {
                    writeBuffer[writeSize++] = readBuffer[currentRegister];
                    writeBuffer[writeSize - 1].y += 1;
                    // globalQueueWriteBuffer[totalToWrite++] = make_int2(globalQueueReadBuffer.x, globalQueueReadBuffer.y+1);
                }
            }

            // Para garantir a consistência da memória compartilhada
            __syncthreads();


            // Verificamos se devemos escrever o buffer de escrita na memória global
            if(BUFFER_SIZE - 5 < writeSize) {
                // Aqui escrevemos o buffer na memória global
                writtenAmount = insertIntoGlobalQueue(gq->writeQueue, &(gq->writeQueueSize), GLOBAL_QUEUE_SIZE, writeBuffer, writeSize);

                // Restaurando a fila
                if(tid==0) {
                    // Verificamos se a flag de overflow deve ser settada
                    if(writtenAmount < writeSize) {
                        *memLeak = true;
                    }
                    writeSize = 0;
                }
            }

            // Esperamos todo mundo encerrar o loop
            // ------------NÃO SEI SE PRECISA DESSA SINCRONIZAÇÃO
            __syncthreads();
        }

    }

    // Aqui escrevemos na memória global os resultados do buffer que ainda não foram escritos
    if(writeSize > 0) {
        writtenAmount = insertIntoGlobalQueue(gq->writeQueue, &(gq->writeQueueSize), GLOBAL_QUEUE_SIZE, writeBuffer, writeSize);
        
        // Verificamos se a flag de overflow deve ser settada
        if(tid==0 and writtenAmount < writeSize) {
            *memLeak = true;
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

// ---------------------------------------------------------------------
// RASTER ANTI-RASTER SCAN ALTERNATIVO
// ---------------------------------------------------------------------

template <typename T>
__global__ void
iRec1DForward_X_dilation ( T* marker, const T* mask, const unsigned int sx, const unsigned int sy, bool* change )
{
	const unsigned int x = (threadIdx.x + threadIdx.y * XX_THREADS) % WARP_SIZE;
	const unsigned int y = (threadIdx.x + threadIdx.y * XX_THREADS) / WARP_SIZE;
	const unsigned int ychunk = WARP_SIZE / XX_THREADS;
	const unsigned int xstop = sx - WARP_SIZE;
//	printf("(tx, ty) -> (x, y) : (%d, %d)->(%d,%d)\n", threadIdx.x, threadIdx.y, x, y);

	// XY_THREADS should be 32==warpSize, XX_THREADS should be 4 or 8.
	// init to 0...
	volatile __shared__ T s_marker[XY_THREADS][WARP_SIZE+1];
	volatile __shared__ T s_mask  [XY_THREADS][WARP_SIZE+1];
	volatile unsigned int s_change = 0;
	T s_old, s_new;
	unsigned int startx;
	unsigned int start;



	s_marker[threadIdx.y][WARP_SIZE] = 0;  // only need x=0 to be 0

	// the increment allows overlap by 1 between iterations to move the data to next block.
	for (startx = 0; startx < xstop; startx += WARP_SIZE) {
		start = (blockIdx.x * XY_THREADS + y * ychunk) * sx + startx + x;
//			printf("tx: %d, ty: %d, x: %d, y: %d, startx: %d, start: %d", threadIdx.x, threadIdx.y, x, y, startx, start);

		s_marker[threadIdx.y][0] = s_marker[threadIdx.y][WARP_SIZE];

		// copy part of marker and mask to shared memory.  works for 1 warp at a time...
//#pragma unroll
		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			s_marker[y * ychunk+i][x+1] = marker[start + i*sx];
			s_mask  [y * ychunk+i][x+1] = mask[start + i*sx];
		}

		// perform iteration   all X threads do the same operations, so there may be read/write hazards.  but the output is the same.
		// this is looping for BLOCK_SIZE times, and each iteration the final results are propagated 1 step closer to tx.
//			if (threadIdx.x == 0) {  // have all threads do the same work
    //#pragma unroll
    if (threadIdx.y + blockIdx.x * XY_THREADS < sy) {   //require dimension to be perfectly padded.
            for (unsigned int i = 1; i <= WARP_SIZE; ++i) {
                s_old = s_marker[threadIdx.y][i];
                s_new = min( max( s_marker[threadIdx.y][i-1], s_old ), s_mask[threadIdx.y][i] );
                s_change |= s_new ^ s_old;
                s_marker[threadIdx.y][i] = s_new;
            }
    }
		// output result back to global memory and set up for next x chunk
//#pragma unroll
		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			marker[start + i*sx] = s_marker[y * ychunk+i][x+1];
		}
//			printf("startx: %d, change = %d\n", startx, s_change);

	}

	if (startx < sx) {
		s_marker[threadIdx.y][0] = s_marker[threadIdx.y][sx-startx];  // getting ix-1st entry, which has been offsetted by 1 in s_marker
		// shared mem copy
		startx = sx - WARP_SIZE;
		start = (blockIdx.x * XY_THREADS + y * ychunk) * sx + startx + x;
//			printf("tx: %d, ty: %d, x: %d, y: %d, startx: %d, start: %d", threadIdx.x, threadIdx.y, x, y, startx, start);

		// copy part of marker and mask to shared memory.  works for 1 warp at a time...
    //#pragma unroll
		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			s_marker[y * ychunk+i][x+1] = marker[start + i*sx];
			s_mask  [y * ychunk+i][x+1] = mask[start + i*sx];
		}

		// perform iteration   all X threads do the same operations, so there may be read/write hazards.  but the output is the same.
		// this is looping for BLOCK_SIZE times, and each iteration the final results are propagated 1 step closer to tx.
    //#pragma unroll
    if (threadIdx.y + blockIdx.x * XY_THREADS < sy) {   //require dimension to be perfectly padded.
            for (unsigned int i = 1; i <= WARP_SIZE; ++i) {
                s_old = s_marker[threadIdx.y][i];
                s_new = min( max( s_marker[threadIdx.y][i-1], s_old ), s_mask[threadIdx.y][i] );
                s_change |= s_new ^ s_old;
                s_marker[threadIdx.y][i] = s_new;
            }
    }
		// output result back to global memory and set up for next x chunk
//#pragma unroll
    for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			marker[start + i*sx] = s_marker[y * ychunk+i][x+1];
		}
	}


//	__syncthreads();
	if (s_change > 0) *change = true;
//	__syncthreads();

}

template <typename T>
__global__ void
iRec1DForward_Y_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
{
	// parallelize along x.
	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;

	unsigned int  s_change = 0;
	T s_old, s_new, s_prev;
	
if ( (bx + tx) < sx ) {

		s_prev = 0;

		for (int iy = 0; iy < sy; ++iy) {
			// copy part of marker and mask to shared memory
			s_old = marker[iy * sx + bx + tx];

			// perform iteration
			s_new = min( max( s_prev, s_old ), mask[iy * sx + bx + tx] );
			s_change |= s_old ^ s_new;
			s_prev = s_new;

			// output result back to global memory
			marker[iy * sx + bx + tx] = s_new;

		}
}
		
		if (s_change != 0) *change = true;


}

template <typename T>
__global__ void
iRec1DBackward_X_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
{

	const unsigned int x = (threadIdx.x + threadIdx.y * XX_THREADS) % WARP_SIZE;
	const unsigned int y = (threadIdx.x + threadIdx.y * XX_THREADS) / WARP_SIZE;
	const unsigned int ychunk = WARP_SIZE / XX_THREADS;
	const unsigned int xstop = sx - WARP_SIZE;
	//	printf("(tx, ty) -> (x, y) : (%d, %d)->(%d,%d)\n", threadIdx.x, threadIdx.y, x, y);

	// XY_THREADS should be 32==warpSize, XX_THREADS should be 4 or 8.
	// init to 0...
	volatile __shared__ T s_marker[XY_THREADS][WARP_SIZE+1];
	volatile __shared__ T s_mask  [XY_THREADS][WARP_SIZE+1];
	volatile unsigned int s_change = 0;
	T s_old, s_new;
	int startx;
	unsigned int start;
	
	s_marker[threadIdx.y][0] = 0;  // only need x=WARPSIZE to be 0

	// the increment allows overlap by 1 between iterations to move the data to next block.
	for (startx = xstop; startx > 0; startx -= WARP_SIZE) {
		start = (blockIdx.x * XY_THREADS + y * ychunk) * sx + startx + x;
//			printf("tx: %d, ty: %d, x: %d, y: %d, startx: %d, start: %d", threadIdx.x, threadIdx.y, x, y, startx, start);

		s_marker[threadIdx.y][WARP_SIZE] = s_marker[threadIdx.y][0];

		// copy part of marker and mask to shared memory.  works for 1 warp at a time...
//#pragma unroll
		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			s_marker[y * ychunk+i][x] = marker[start + i*sx];
			s_mask  [y * ychunk+i][x] = mask[start + i*sx];
		}

		// perform iteration   all X threads do the same operations, so there may be read/write hazards.  but the output is the same.
		// this is looping for BLOCK_SIZE times, and each iteration the final results are propagated 1 step closer to tx.
//			if (threadIdx.x == 0) {  // have all threads do the same work
//#pragma unroll
        if (threadIdx.y + blockIdx.x * XY_THREADS < sy) {   //require dimension to be perfectly padded.
                for (int i = WARP_SIZE - 1; i >= 0; --i) {
                    s_old = s_marker[threadIdx.y][i];
                    s_new = min( max( s_marker[threadIdx.y][i+1], s_old ), s_mask[threadIdx.y][i] );
                    s_change |= s_new ^ s_old;
                    s_marker[threadIdx.y][i] = s_new;
                }
        }
		// output result back to global memory and set up for next x chunk
//#pragma unroll
		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			marker[start + i*sx] = s_marker[y * ychunk+i][x];
		}
//			printf("startx: %d, change = %d\n", startx, s_change);
	}

	if (startx <= 0) {
		s_marker[threadIdx.y][WARP_SIZE] = s_marker[threadIdx.y][-startx];  // getting ix-1st entry, which has been offsetted by 1 in s_marker
		// shared mem copy
		startx = 0;
		start = (blockIdx.x * XY_THREADS + y * ychunk) * sx + startx + x;
//			printf("tx: %d, ty: %d, x: %d, y: %d, startx: %d, start: %d", threadIdx.x, threadIdx.y, x, y, startx, start);

		// copy part of marker and mask to shared memory.  works for 1 warp at a time...
//#pragma unroll
		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			s_marker[y * ychunk+i][x] = marker[start + i*sx];
			s_mask  [y * ychunk+i][x] = mask[start + i*sx];
		}

		// perform iteration   all X threads do the same operations, so there may be read/write hazards.  but the output is the same.
		// this is looping for BLOCK_SIZE times, and each iteration the final results are propagated 1 step closer to tx.
//#pragma unroll
        if (threadIdx.y + blockIdx.x * XY_THREADS < sy) {   //require dimension to be perfectly padded.
                for (int i = WARP_SIZE - 1; i >= 0; --i) {
                    s_old = s_marker[threadIdx.y][i];
                    s_new = min( max( s_marker[threadIdx.y][i+1], s_old ), s_mask[threadIdx.y][i] );
                    s_change |= s_new ^ s_old;
                    s_marker[threadIdx.y][i] = s_new;
                }
        }
		// output result back to global memory and set up for next x chunk
//#pragma unroll
		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			marker[start + i*sx] = s_marker[y * ychunk+i][x];
		}
	}

	//	__syncthreads();
	if (s_change > 0) *change = true;
	//	__syncthreads();
}

template <typename T>
	__global__ void
iRec1DBackward_Y_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const unsigned int sx, const unsigned int sy, bool* __restrict__ change )
{

	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;

	unsigned int s_change=0;
	T s_old, s_new, s_prev;

	if ( (bx + tx) < sx ) {

		s_prev = 0;

		for (int iy = sy - 1; iy >= 0; --iy) {

			// copy part of marker and mask to shared memory
			s_old = marker[iy * sx + bx + tx];

			// perform iteration
			s_new = min( max( s_prev, s_old ), mask[iy * sx + bx + tx] );
			s_change |= s_old ^ s_new;
			s_prev = s_new;

			// output result back to global memory
			marker[iy * sx + bx + tx] = s_new;
		}
	}

	if (s_change != 0) *change = true;

}

// Assumimos que a imagem é quadrada
void one_alternative_raster_pass(unsigned char* marker, unsigned char* mask, int size) {
    // Dimensões de Launch
    dim3 threadsx( XX_THREADS, XY_THREADS );
	dim3 blocksx( (size + threadsx.y - 1) / threadsx.y );
    dim3 threadsy( MAX_THREADS );
	dim3 blocksy( (size + threadsy.x - 1) / threadsy.x );

    bool *d_change;

    int i;
    cudaEvent_t start[4], stop[4];
    float time[4];

    for(i=0; i<4; i++) {
        cudaEventCreate(&start[i]);
        cudaEventCreate(&stop[i]);
    }


    // Alocamos a variável
    cudaMalloc(&d_change, sizeof(bool));

    // dopredny pruchod pres osu X
    cudaEventRecord(start[0]);
    iRec1DForward_X_dilation <<< blocksx, threadsx, 0, 0 >>> ( marker, mask, size, size, d_change );
    cudaEventRecord(stop[0]);

    // dopredny pruchod pres osu Y
    cudaEventRecord(start[1]);
    iRec1DForward_Y_dilation <<< blocksy, threadsy, 0, 0 >>> ( marker, mask, size, size, d_change );
    cudaEventRecord(stop[1]);

    // zpetny pruchod pres osu X
    cudaEventRecord(start[2]);
    iRec1DBackward_X_dilation<<< blocksx, threadsx, 0, 0 >>> ( marker, mask, size, size, d_change );
    cudaEventRecord(stop[2]);

    // zpetny pruchod pres osu Y
    cudaEventRecord(start[3]);
    iRec1DBackward_Y_dilation<<< blocksy, threadsy, 0, 0 >>> ( marker, mask, size, size, d_change );
    cudaEventRecord(stop[3]);

    cudaDeviceSynchronize();

    for(i=0; i<4; i++) {
        cudaEventElapsedTime(&time[i], start[i], stop[i]);
        //printf("Passada %d: %f (ms)\n", i+1, time[i]);
        printf("%f\n", time[i]);
    }

    cudaFree(d_change);
    for(i=0; i<4; i++) {
        cudaEventDestroy(start[i]);
        cudaEventDestroy(stop[i]);
    }

}


void one_current_pass(unsigned char* marker, unsigned char* mask, size_t pitchMarker, size_t pitchMask, int size) {
    int i;
    cudaEvent_t start[4], stop[4];
    float time[4];

    for(i=0; i<4; i++) {
        cudaEventCreate(&start[i]);
        cudaEventCreate(&stop[i]);
    }

    // dopredny pruchod pres osu X
    cudaEventRecord(start[0]);
    XForwardPropagationEntireImage<<<dim3(size/MEGAPIXEL_SIZE, 1), 32>>>(marker, mask, pitchMarker, pitchMask, size);
    cudaEventRecord(stop[0]);

    // dopredny pruchod pres osu Y
    cudaEventRecord(start[1]);
    YDownwardPropagationEntireImage<<<dim3(1, size/MEGAPIXEL_SIZE), 32>>>(marker, mask, pitchMarker, pitchMask, size);
    cudaEventRecord(stop[1]);

    // zpetny pruchod pres osu X
    cudaEventRecord(start[2]);
    XBackwardPropagationEntireImage<<<dim3(size/MEGAPIXEL_SIZE, 1), 32>>>(marker, mask, pitchMarker, pitchMask, size);
    cudaEventRecord(stop[2]);

    // zpetny pruchod pres osu Y
    cudaEventRecord(start[3]);
    YUpwardPropagationEntireImage<<<dim3(1, size/MEGAPIXEL_SIZE), 32>>>(marker, mask, pitchMarker, pitchMask, size);
    cudaEventRecord(stop[3]);

    cudaDeviceSynchronize();

    for(i=0; i<4; i++) {
        cudaEventElapsedTime(&time[i], start[i], stop[i]);
        //printf("Passada %d: %f (ms)\n", i+1, time[i]);
        printf("%f\n", time[i]);
    }

    for(i=0; i<4; i++) {
        cudaEventDestroy(start[i]);
        cudaEventDestroy(stop[i]);
    }
}

// ---------------------------------------------------------------------

// Extract unique elements from write queue and insert them into readQueue
void __global__ extractUniques(GlobalQueue* gq, int* flags, size_t flags_pitch) {
  
  // Thread block identification
  int tid = threadIdx.x;
  int blockId = blockIdx.x;
  int numBlocks = gridDim.x;
  
  // Geting a slice of the queue
  const int quotient = (gq->writeQueueSize)/numBlocks;
  const int remainder = (gq->writeQueueSize)%numBlocks;
  
  // Fila de leitura
  ushort2* writeGlobalQueue;
  int writeQueueSize;
  int readedAmount;
  int currentRegister;
  __shared__ ushort2 readBuffer[1024];
  __shared__ ushort2 writeBuffer[1024];
  __shared__ int writePos;
  __shared__ int readOffset;
  int writeIndex;

  // Processamento de registro
  int* flags_row;

  // Tamanho da fila
  writeQueueSize = quotient;
  if(blockId < remainder) {
      writeQueueSize++;
  }

  // Offset
  int offset = min(remainder, blockId)*(quotient + 1);
  if(blockId > remainder) {
      offset += (blockId - remainder)*quotient;
  }

  // Ponteiro para a fila
  writeGlobalQueue = &(gq->writeQueue[offset]);

  // Vai ter condição de corrida, mas isso é irrelevante
  writePos = 0;
  readOffset = 0;


  while(readedAmount = readFromGlobalQueue(writeGlobalQueue, readOffset, writeQueueSize, readBuffer, 1024)) {
    for(currentRegister=0; currentRegister<readedAmount; currentRegister+=blockDim.x) {
      
      // Processamos um elemento
      if(currentRegister + tid < readedAmount){
        flags_row = (int*)((char*)flags + flags_pitch*readBuffer[currentRegister + tid].x);

        //Verificamos se o elemento já está na fila
        if(!atomicExch(&flags_row[readBuffer[currentRegister + tid].y], 1)) {
          writeIndex = atomicAdd(&writePos, 1);

          writeBuffer[writeIndex] = readBuffer[currentRegister + tid];
        }
      }

      //Sincronizamos para grantia de consistência de memória
      __syncthreads();

      // Verificamos se precisamos fazer o dump do buffer
      if(1024 - blockDim.x < writePos) {
        insertIntoGlobalQueue(gq->readQueue, &(gq->readQueueSize), GLOBAL_QUEUE_SIZE, writeBuffer, writePos);
        writePos = 0;
        // Falta a verificação de overflow
      }
    }

  }

  // Escrevemos o resto do vetor
  if(writePos) {
    insertIntoGlobalQueue(gq->readQueue, &(gq->readQueueSize), GLOBAL_QUEUE_SIZE, writeBuffer, writePos);
  }


}


void __global__ clearHashAndResetWriteQueue(GlobalQueue* gq, int* flags, size_t flags_pitch) {
  
  // Thread block identification
  int tid = threadIdx.x;
  int blockId = blockIdx.x;
  int numBlocks = gridDim.x;
  
  // Geting a slice of the queue
  const int quotient = (gq->readQueueSize)/numBlocks;
  const int remainder = (gq->readQueueSize)%numBlocks;
  
  // Fila de leitura
  ushort2* readGlobalQueue;
  int readQueueSize;
  int readedAmount;
  int currentRegister;
  __shared__ ushort2 readBuffer[1024];
  __shared__ int readOffset;

  // Processamento de registro
  int* flags_row;

  // Tamanho da fila
  readQueueSize = quotient;
  if(blockId < remainder) {
      readQueueSize++;
  }

  // Offset
  int offset = min(remainder, blockId)*(quotient + 1);
  if(blockId > remainder) {
      offset += (blockId - remainder)*quotient;
  }

  // Ponteiro para a fila
  readGlobalQueue = &(gq->readQueue[offset]);

  // Vai ter condição de corrida, mas isso é irrelevante
  readOffset = 0;


  while(readedAmount = readFromGlobalQueue(readGlobalQueue, readOffset, readQueueSize, readBuffer, 1024)) {
    for(currentRegister=0; currentRegister<readedAmount; currentRegister+=blockDim.x) {
      
      // Processamos um elemento
      if(currentRegister + tid < readedAmount){
        flags_row = (int*)((char*)flags + flags_pitch*readBuffer[currentRegister + tid].x);

        // Resetamos apenas as posições que foram alteradas
        flags_row[readBuffer[currentRegister + tid].y] = 0;
      }

    }

  }

  // Escritas redundantes, mas isso não é um problema
  gq->writeQueueSize = 0;

}

// Esse kerne devera ser executado por apenas uma thread
void __global__ printReadQueueu(GlobalQueue* gq) {
    int i;


    // Imprimimos o tamanho da fila
    printf("%d\n", gq->readQueueSize);

    // Imprimimos os elementos da fila
    for(i=0; i<gq->readQueueSize; i++) {
        printf("(%d, %d) ", gq->readQueue[i].x, gq->readQueue[i].y);
    }

    printf("\n");
}

// Retorna quantas entradas são diferentes da constante
void __global__ verifyConstantMatrix(int* matrix, size_t matrix_pitch, int constant, int* result) {
  int myRow = blockIdx.x*blockDim.x + threadIdx.x;
  int myCol = blockIdx.y*blockDim.y + threadIdx.y;

  int* matrixRow = (int*) ((char*)matrix + myRow*matrix_pitch);

  if(matrixRow[myCol] != constant) {
    atomicAdd(result, 1);
  }

}

void MorphologicalReconstruction(unsigned char* marker, unsigned char* mask, 
            size_t pitchMarker, size_t pitchMask, int size)
{
    int blockSize = MEGAPIXEL_SIZE;
    int maximumBlockOccupancy = 2048;
    int gridSize = maximumBlockOccupancy;
    


    // cudaEvent_t start, stop;

    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // We make a copy of the image 
    //unsigned char* marker_copy;
    //size_t marker_copy_pitch;

    //cudaMallocPitch(&marker_copy, &marker_copy_pitch, size*sizeof(unsigned char), size);

    //cudaMemcpy2D(marker_copy, marker_copy_pitch, marker, pitchMarker, size*sizeof(unsigned char), size, cudaMemcpyDeviceToDevice);

    // We allocate a table of flags
    int* flags;
    size_t flags_pitch;
    // A tabela de flags é para o grid de megapixels e não para o grid de pixels
    cudaMallocPitch(&flags, &flags_pitch, (size/MEGAPIXEL_SIZE)*sizeof(int), (size/MEGAPIXEL_SIZE));
    cudaMemset2DAsync(flags, flags_pitch, 0, (size/MEGAPIXEL_SIZE)*sizeof(int), (size/MEGAPIXEL_SIZE));

    int megapixelGridSize = size/MEGAPIXEL_SIZE;
    
    GlobalQueue *gq;
    // Assumimos que a imagem é quadrada
    int readQueueSize;
    //int uniqueQueueSize = megapixelGridSize*megapixelGridSize; 

    int hMemLeak;
    int* dMemLeak;

    //int count = 0;
    
    // float morphReconTime = 0;
    // float queueMaintainceTime = 0;
    // float uniqueExtractionTime = 0;
    // float cleaningHashTime = 0;

    unsigned long long megapixelsProcessed = 0;

    // Teste do clear hash kernel 
    // int* d_result;
    // int h_result;
    // cudaMalloc(&d_result, sizeof(int));

    //GlobalQueue *gq_copy;

     // inicializa o mecanismo de controle de overflow
    cudaMalloc(&dMemLeak, sizeof(int)); 

    // Inicializa a struct da fila global
    cudaMalloc(&gq, sizeof(GlobalQueue));
    //cudaMalloc(&gq_copy, sizeof(GlobalQueue));

    // Inicializa os atributos da struct 
    initGlobalQueue(gq);
    
    //initGlobalQueue(gq_copy);

    // auto tStart = std::chrono::high_resolution_clock::now();
    do{
        // Inicializa a fila de megapixels e a flag de overflow de memória
        cudaMemset(dMemLeak, 0, sizeof(int));
        initFillGlobalQueue<<<dim3(megapixelGridSize/FILL_BLOCK_SIZE,megapixelGridSize/FILL_BLOCK_SIZE), 
                    dim3(FILL_BLOCK_SIZE, FILL_BLOCK_SIZE)>>>(gq);
        readQueueSize = megapixelGridSize*megapixelGridSize; 
        //initFillGlobalQueue<<<dim3(megapixelGridSize/FILL_BLOCK_SIZE,megapixelGridSize/FILL_BLOCK_SIZE), 
        //            dim3(FILL_BLOCK_SIZE, FILL_BLOCK_SIZE)>>>(gq_copy);
        
        // Computamos a reconstrução morfológica
        while(readQueueSize > 0) {
            gridSize = min(readQueueSize, maximumBlockOccupancy);
            
            // cudaEventRecord(start);
            megapixelsProcessed += readQueueSize;
            MorphologicalReconstructionKernel<<<gridSize, blockSize>>>(gq, marker, mask, pitchMarker, pitchMask, size, dMemLeak);
            // cudaEventRecord(stop);
            // cudaEventSynchronize(stop);
            // cudaEventElapsedTime(&morphReconTime, start, stop);

            //cudaEventRecord(start);
            //if(readQueueSize >= 1344) {
            resetReadQueue<<<1,1>>>(gq);
            // cudaEventRecord(stop);
            // cudaEventSynchronize(stop);
            // cudaEventElapsedTime(&queueMaintainceTime, start, stop);

            // cudaEventRecord(start);
            extractUniques<<<2048, 32>>>(gq, flags, flags_pitch);
            // extractUniques<<<maximumBlockOccupancy, MEGAPIXEL_SIZE>>>(gq, flags, flags_pitch);
            // cudaEventRecord(stop);
            // cudaEventSynchronize(stop);
            // cudaEventElapsedTime(&uniqueExtractionTime, start, stop);

            // cudaEventRecord(start);
            clearHashAndResetWriteQueue<<<2048, 32>>>(gq, flags, flags_pitch);
            // cudaEventRecord(stop);
            // cudaEventSynchronize(stop);
            // cudaEventElapsedTime(&cleaningHashTime, start, stop);
            //count++;
            // }
            // else {
            //   swapQueues<<<1,1>>>(gq);
            // }

            // cudaMemset(d_result, 0, sizeof(int));
            // verifyConstantMatrix<<<dim3(8,8), dim3(16, 16)>>>(flags, flags_pitch, 0, d_result);
            // cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
            // cudaDeviceSynchronize();
            // Kernel que consome a fila de leitura
            //cudaEventRecord(start);
            //MorphologicalReconstructionKernel<<<gridSize, blockSize>>>(gq, marker_copy, mask, pitchMarker, pitchMask, size, dMemLeak);
            // cudaEventRecord(stop);
            // cudaEventSynchronize(stop);
            // cudaEventElapsedTime(&time, start, stop);

            //cudaEventRecord(start);
            //MorphologicalReconstructionKernel<<<gridSize, blockSize>>>(gq_copy, marker, mask, marker_copy_pitch, pitchMask, size, dMemLeak);
            //cudaEventRecord(stop);
            //cudaEventSynchronize(stop);
            //cudaEventElapsedTime(&uniqueTime, start, stop);
            
            //printf("%d\t%f\t%d\t%f\n", readQueueSize, time, uniqueQueueSize, uniqueTime);

            //swapQueues<<<1,1>>>(gq);
            //swapQueues<<<1,1>>>(gq_copy);
            //swapQueues<<<1,1>>>(gq_copy);

            // cudaMemset2D(flags, flags_pitch, 0, size*sizeof(int), size);
            // cudaMemset(&(gq->writeQueueSize), 0, sizeof(int));
            //extractUniques<<<1, 32>>>(gq, flags, flags_pitch);
             
            //printReadQueueu<<<1,1>>>(gq);
            //printReadQueueu<<<1,1>>>(gq_copy);

            // cudaDeviceSynchronize();

            // Fazer mais testes depois---------------------------------------------
            //cudaMemcpy(&uniqueQueueSize, &(gq_copy->readQueueSize), sizeof(int), cudaMemcpyDeviceToHost);
            // printf("%d\n", readQueueSize);
            //printf("Read Queue Size: %d\n", readQueueSize);
            //printf("Result: %d\n", h_result);
            
            // printf("%d\t%f\t%f\t%f\t%f\n", readQueueSize, morphReconTime, queueMaintainceTime, uniqueExtractionTime, cleaningHashTime);
            cudaMemcpy(&readQueueSize, &(gq->readQueueSize), sizeof(int), cudaMemcpyDeviceToHost);
            
            //printf("Original Queue: %d (%f ms)\nUnique Queue: %d (%f ms)\n", readQueueSize, time, uniqueQueueSize, uniqueTime);
            //printf("%d\t%f\t%d\t%f\n", readQueueSize, time, uniqueQueueSize, uniqueTime);

            //cudaDeviceSynchronize();
        }


        // Obtem a flag de overflow
        cudaMemcpy(&hMemLeak, dMemLeak, sizeof(int), cudaMemcpyDeviceToHost);
        // cudaDeviceSynchronize();
        printf("MemLeak: %d\n", hMemLeak);

    } while(hMemLeak);
    
    // auto tEnd = std::chrono::high_resolution_clock::now();
    // auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart);
    // printf("Tempo total do loop: %ld\n", ms_int.count());
    //cudaDeviceSynchronize();

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    // printf("Megapixels Processed: %llu\n", megapixelsProcessed);

    // Libera a memória alocada
    freeGlobalQueue<<<1,1>>>(gq);
    cudaDeviceSynchronize();

    cudaFree(gq);
    //cudaFree(gq_copy);
    //cudaFree(marker_copy);
}


// Após a execução desse kernel, todo elemento do marker é menor ou igual que o elemento correspondente do mask
void __global__ clipImage(unsigned char* marker, unsigned char* mask, size_t pitchMarker, size_t pitchMask) {
    int myRow = blockIdx.x * blockDim.x + threadIdx.x;
    int myCol = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned char* markerRow =  (unsigned char*)((char*)marker + (myRow) * pitchMarker);
    unsigned char* maskRow =  (unsigned char*)((char*)mask + (myRow) * pitchMask);

    markerRow[myCol] = min(markerRow[myCol], maskRow[myCol]);

}
