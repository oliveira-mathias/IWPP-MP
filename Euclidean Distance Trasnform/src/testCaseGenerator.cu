#include "testCaseGenerator.cuh"


void __global__ randomNumberGeneratorSetupKernel(curandState *state, int seed) {
    int myRow = blockIdx.x*blockDim.x + threadIdx.x;
    int myCol = blockIdx.y*blockDim.y + threadIdx.y;

    int id = (myRow)*(gridDim.y*blockDim.y) + myCol;
    // printf("(%d, %d, %d),\n", myRow, myCol, id);
    /* Each thread gets a different seed, a maybe an equal sequence
        number, no offset */
    curand_init(seed + id, id%64 + 7, 0, &state[id]);   
}

void __global__ randomNumberGeneratorGenerateKernel(curandState *state, ushort2* voronoi, size_t pitchVoronoi, 
                    float backgroundProb, int rowOffset, int colOffset) 
{
    int myRow = blockIdx.x*blockDim.x + threadIdx.x;
    int myCol = blockIdx.y*blockDim.y + threadIdx.y;

    unsigned int id = (myRow)*(gridDim.y*blockDim.y) + myCol;

    ushort2* voronoiRow = (ushort2*)((char*)voronoi + (myRow + rowOffset) * pitchVoronoi);

    unsigned int generated;
    unsigned int threshHold = backgroundProb*MAX_UNSIGNED_INT;

    /* Copy state to local memory for efficiency */
    curandState localState = state[id];

    generated = curand(&localState);

    // We verify if the pixel will be a background or a foreground pixel
    if(generated<=threshHold) {
        voronoiRow[myCol + colOffset] = make_ushort2(myRow + rowOffset, myCol + colOffset);
    }
    else {
        voronoiRow[myCol + colOffset] = make_ushort2(MAX_UNSIGNED_SHORT, MAX_UNSIGNED_SHORT);
    }

    /* Copy state back to global memory */
    state[id] = localState;
}

void InitRandomTestCase(ushort2* voronoi, size_t pitchVoronoi, int randomGeneratorGridSize, float backgroundProb, std::vector<std::pair<int, int>> offsets) {
    curandState *devStates;

    int blockSide = 16;
    int gridSide = randomGeneratorGridSize/blockSide;
    int i;
    

    cudaMalloc((void **)&devStates, (randomGeneratorGridSize*randomGeneratorGridSize) *sizeof(curandState));

    // R1 seed: 2441
    // R2 seed: 1701
    // R3 seed: 5443
    // R4 seed: 4111
    randomNumberGeneratorSetupKernel<<<dim3(gridSide,gridSide), dim3(blockSide,blockSide)>>>(devStates, 4111);
    for(i=0; i<offsets.size(); i++) {
        randomNumberGeneratorGenerateKernel<<<dim3(gridSide,gridSide), dim3(blockSide,blockSide)>>>(devStates, voronoi, pitchVoronoi, backgroundProb, offsets[i].first, offsets[i].second);
    }
    
    cudaDeviceSynchronize();

    cudaFree(devStates);
}

void __global__ extractImage(unsigned char* image, int pitchImage, ushort2* voronoi, size_t pitchVoronoi) {
    int myRow = blockIdx.x * blockDim.x + threadIdx.x;
    int myCol = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned char* imageRow = (unsigned char*)((char*)image + (myRow) * pitchImage);;
    ushort2* voronoiRow = (ushort2*)((char*)voronoi + (myRow)*pitchVoronoi);

    // Aqui o pixel é foreground
    if(voronoiRow[myCol].x==MAX_UNSIGNED_SHORT) {
        imageRow[myCol] = 255;
    }
    // Aqui o pixel é background
    else {
        imageRow[myCol] = 0;
    }
}

int __device__ euclideanDistance(int x1, int y1, int x2, int y2);
void __global__ initCustomTestCase(ushort2* voronoi, size_t pitchVoronoi) {
    int myRow = blockIdx.x * blockDim.x + threadIdx.x;
    int myCol = blockIdx.y * blockDim.y + threadIdx.y;

    ushort2* voronoiRow = (ushort2*)((char*)voronoi + (myRow)*pitchVoronoi);

    // corners if((blockIdx.x==31 and blockIdx.y==31) or (blockIdx.x==31 and blockIdx.y==225) or (blockIdx.x==225 and blockIdx.y==31) or (blockIdx.x==225 and blockIdx.y==225)) {
    
    //circle if(euclideanDistance(myRow, myCol, 2047, 2047)==1048576){  

		//thicker_circle if(euclideanDistance(myRow, myCol, 2047, 2047)>=1048000 and euclideanDistance(myRow, myCol, 2047, 2047)<=1049000){  
		
		// thicker_circles if((euclideanDistance(myRow, myCol, 1023, 1023)>=261000 and euclideanDistance(myRow, myCol, 1023, 1023)<=263000)
		//		or (euclideanDistance(myRow, myCol, 1023, 3071)>=261000 and euclideanDistance(myRow, myCol, 1023, 3071)<=263000)
		//		or (euclideanDistance(myRow, myCol, 3071, 1023)>=261000 and euclideanDistance(myRow, myCol, 3071, 1023)<=263000)
		//		or (euclideanDistance(myRow, myCol, 3071, 3071)>=261000 and euclideanDistance(myRow, myCol, 3071, 3071)<=263000)){  
		/*16thick_circles if(
				(euclideanDistance(myRow, myCol, 511, 511)>=16300 and euclideanDistance(myRow, myCol, 511, 511)<=16400)
				or (euclideanDistance(myRow, myCol, 511, 1535)>=16300 and euclideanDistance(myRow, myCol, 511, 1535)<=16400)
				or (euclideanDistance(myRow, myCol, 511, 2559)>=16300 and euclideanDistance(myRow, myCol, 511, 2559)<=16400)
				or (euclideanDistance(myRow, myCol, 511, 3585)>=16300 and euclideanDistance(myRow, myCol, 511, 3585)<=16400)
				or (euclideanDistance(myRow, myCol, 1535, 511)>=16300 and euclideanDistance(myRow, myCol, 1535, 511)<=16400)
				or (euclideanDistance(myRow, myCol, 1535, 1535)>=16300 and euclideanDistance(myRow, myCol,1535, 1535)<=16400)
				or (euclideanDistance(myRow, myCol, 1535, 2559)>=16300 and euclideanDistance(myRow, myCol, 1535, 2559)<=16400)
				or (euclideanDistance(myRow, myCol, 1535, 3585)>=16300 and euclideanDistance(myRow, myCol, 1535, 3585)<=16400)
				or (euclideanDistance(myRow, myCol, 2559, 511)>=16300 and euclideanDistance(myRow, myCol, 2559, 511)<=16400)
				or (euclideanDistance(myRow, myCol, 2559, 1535)>=16300 and euclideanDistance(myRow, myCol, 2559, 1535)<=16400)
				or (euclideanDistance(myRow, myCol, 2559, 2559)>=16300 and euclideanDistance(myRow, myCol, 2559, 2559)<=16400)
				or (euclideanDistance(myRow, myCol, 2559, 3585)>=16300 and euclideanDistance(myRow, myCol, 2559, 3585)<=16400)
				or (euclideanDistance(myRow, myCol, 3585, 511)>=16300 and euclideanDistance(myRow, myCol, 3585, 511)<=16400)
				or (euclideanDistance(myRow, myCol, 3585, 1535)>=16300 and euclideanDistance(myRow, myCol, 3585, 1535)<=16400)
				or (euclideanDistance(myRow, myCol, 3585, 2559)>=16300 and euclideanDistance(myRow, myCol, 3585, 2559)<=16400)
				or (euclideanDistance(myRow, myCol, 3585, 3585)>=16300 and euclideanDistance(myRow, myCol, 3585, 3585)<=16400)
		){*/

		// diag if(myRow==myCol) {

		// X if(myRow==myCol or (myCol==(4095-myRow))) {  
		
		// single_corner if(myRow==4095 and myCol==0) {
		// quad_corner if((myRow==0 and myCol==0) or (myRow==0 and myCol==4095) or (myRow==4095 and myCol==0) or (myRow==4095 and myCol==4095)) {
		// big_thick_circle if((euclideanDistance(4095, 2047, myRow, myCol) <= 4194500) and (euclideanDistance(4095, 2047, myRow, myCol) >= 4194100)) {
        // big_thick_circle_8k if((euclideanDistance(8191, 4095, myRow, myCol) <= 16770000) and (euclideanDistance(8191, 4095, myRow, myCol) >= 16760000)) {
	if(
        (euclideanDistance(myRow, myCol, 1023, 1023)>=65400 and euclideanDistance(myRow, myCol, 1023, 1023)<=65700)
        or (euclideanDistance(myRow, myCol, 1023, 3071)>=65400 and euclideanDistance(myRow, myCol, 1023, 3071)<=65700)
        or (euclideanDistance(myRow, myCol, 1023, 5119)>=65400 and euclideanDistance(myRow, myCol, 1023, 5119)<=65700)
        or (euclideanDistance(myRow, myCol, 1023, 7167)>=65400 and euclideanDistance(myRow, myCol, 1023, 7167)<=65700)
        or (euclideanDistance(myRow, myCol, 3071, 1023)>=65400 and euclideanDistance(myRow, myCol, 3071, 1023)<=65700)
        or (euclideanDistance(myRow, myCol, 3071, 3071)>=65400 and euclideanDistance(myRow, myCol,3071, 3071)<=65700)
        or (euclideanDistance(myRow, myCol, 3071, 5119)>=65400 and euclideanDistance(myRow, myCol, 3071, 5119)<=65700)
        or (euclideanDistance(myRow, myCol, 3071, 7167)>=65400 and euclideanDistance(myRow, myCol, 3071, 7167)<=65700)
        or (euclideanDistance(myRow, myCol, 5119, 1023)>=65400 and euclideanDistance(myRow, myCol, 5119, 1023)<=65700)
        or (euclideanDistance(myRow, myCol, 5119, 3071)>=65400 and euclideanDistance(myRow, myCol, 5119, 3071)<=65700)
        or (euclideanDistance(myRow, myCol, 5119, 5119)>=65400 and euclideanDistance(myRow, myCol, 5119, 5119)<=65700)
        or (euclideanDistance(myRow, myCol, 5119, 7167)>=65400 and euclideanDistance(myRow, myCol, 5119, 7167)<=65700)
        or (euclideanDistance(myRow, myCol, 7167, 1023)>=65400 and euclideanDistance(myRow, myCol, 7167, 1023)<=65700)
        or (euclideanDistance(myRow, myCol, 7167, 3071)>=65400 and euclideanDistance(myRow, myCol, 7167, 3071)<=65700)
        or (euclideanDistance(myRow, myCol, 7167, 5119)>=65400 and euclideanDistance(myRow, myCol, 7167, 5119)<=65700)
        or (euclideanDistance(myRow, myCol, 7167, 7167)>=65400 and euclideanDistance(myRow, myCol, 7167, 7167)<=65700)
    ){		
            voronoiRow[myCol] = make_ushort2(myRow, myCol);
    }
    else {
        voronoiRow[myCol] = make_ushort2(MAX_UNSIGNED_SHORT, MAX_UNSIGNED_SHORT);
    }
}