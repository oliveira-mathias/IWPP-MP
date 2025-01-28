#include "testCaseGenerator.cuh"

void __global__ randomNumberGeneratorSetupKernel(curandState *state, int seed) {
    int myRow = blockIdx.x*blockDim.x + threadIdx.x;
    int myCol = blockIdx.y*blockDim.y + threadIdx.y;

    int id = (myRow)*(gridDim.y*blockDim.y) + myCol;

    curand_init(seed + id, id%64 + 7, 0, &state[id]);   
}

void __global__ randomNumberGeneratorGenerateKernel(curandState *state, unsigned char* marker, unsigned char* mask, 
        size_t pitchMarker, size_t pitchMask, int rowOffset, int colOfsset) 
{
    int myRow = blockIdx.x*blockDim.x + threadIdx.x;
    int myCol = blockIdx.y*blockDim.y + threadIdx.y;

    int id = (myRow)*(gridDim.y*blockDim.y) + myCol;

    unsigned char* markerRow = (unsigned char*)((char*)marker + (myRow + rowOffset) * pitchMarker);
    unsigned char* maskRow = (unsigned char*)((char*)mask + (myRow + rowOffset) * pitchMask);

    int markerVal, maskVal;

    /* Copy state to local memory for efficiency */
    curandState localState = state[id];

    markerVal = curand(&localState)%256;
    maskVal = curand(&localState)%256;

    // A máscara tem que ser >= do que o marcador
    maskVal = maskVal%(255-markerVal+1);

    markerRow[myCol + colOfsset] = markerVal;
    maskRow[myCol + colOfsset] = markerVal + maskVal;

    /* Copy state back to global memory */
    state[id] = localState;
}

void InitRandomTestCase(unsigned char* marker, unsigned char* mask, 
            size_t pitchMarker, size_t pitchMask, int randomGeneratorGridSize, std::vector<std::pair<int, int>> offsets) 
{
    curandState *devStates;
    int i;

    int blockSide = 16;
    int gridSide = randomGeneratorGridSize/blockSide;

    cudaMalloc((void **)&devStates, (randomGeneratorGridSize*randomGeneratorGridSize) *sizeof(curandState));

    // Seeds testadas 1234
    // Seed do R3: 9859
    // Seed do R4: 1312
    randomNumberGeneratorSetupKernel<<<dim3(gridSide,gridSide), dim3(blockSide,blockSide)>>>(devStates, 1312);
    for(i=0; i<offsets.size(); i++) {
        randomNumberGeneratorGenerateKernel<<<dim3(gridSide,gridSide), dim3(blockSide,blockSide)>>>(devStates, marker, mask, pitchMarker, pitchMask, offsets[i].first, offsets[i].second);
    }
    cudaDeviceSynchronize();

    cudaFree(devStates);
}