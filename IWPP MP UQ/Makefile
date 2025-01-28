CC = /usr/bin/g++-10
# CC = /usr/bin/g++-11
NVCC = nvcc

# GENCODE_FLAGS = -Xptxas -dlcm=cg -arch=compute_86 -gencode arch=compute_86,code=sm_86
# GENCODE_FLAGS = -arch=compute_50 -gencode arch=compute_50,code=sm_50
GENCODE_FLAGS = -arch=compute_86 -gencode arch=compute_86,code=sm_86

# Flag para dump de informação do PTX
# --ptxas-options=-v 

EXTRA_GLOBAL_FLAGS = -O2

COMPUTE_SANITIZER_BIN =/usr/local/cuda-11.8.0/compute-sanitizer/compute-sanitizer

# OPENCV = /usr/local/include/opencv4/
OPENCV = /home/all_home/oliveira.mathias/conda_env/include/opencv4/
INCLUDE = ./include
BUILD = ./build
SRC = ./src
TARGET = exec

$(TARGET): $(BUILD)/main.o $(BUILD)/globalQueue.o $(BUILD)/MorphologicalReconstruction.o #$(BUILD)/testCaseGenerator.o
	$(NVCC) -lineinfo $(GENCODE_FLAGS) $(BUILD)/*.o  $(EXTRA_GLOBAL_FLAGS) -o $(TARGET) -I $(INCLUDE) -L/home/all_home/oliveira.mathias/conda_env/lib -lcurand -lopencv_core -lopencv_imgcodecs

$(BUILD)/main.o: $(SRC)/main.cu
	$(NVCC) -x cu -lineinfo $(GENCODE_FLAGS)  $(EXTRA_GLOBAL_FLAGS) -dc $(SRC)/main.cu -ccbin $(CC) -o $(BUILD)/main.o -I $(INCLUDE) -I$(OPENCV)

$(BUILD)/globalQueue.o: $(SRC)/globalQueue.cu $(INCLUDE)/globalQueue.cuh
	$(NVCC) -x cu -lineinfo $(GENCODE_FLAGS) $(EXTRA_GLOBAL_FLAGS) -dc $(SRC)/globalQueue.cu -ccbin $(CC) -o $(BUILD)/globalQueue.o -I $(INCLUDE) 

$(BUILD)/MorphologicalReconstruction.o: $(SRC)/MorphologicalReconstruction.cu $(INCLUDE)/MorphologicalReconstruction.cuh
	$(NVCC) -x cu --ptxas-options=-v -lineinfo $(GENCODE_FLAGS) $(EXTRA_GLOBAL_FLAGS) -dc $(SRC)/MorphologicalReconstruction.cu -ccbin $(CC) -o $(BUILD)/MorphologicalReconstruction.o -I $(INCLUDE) 

#$(BUILD)/testCaseGenerator.o: $(SRC)/testCaseGenerator.cu $(INCLUDE)/testCaseGenerator.cuh
#	$(NVCC) -x cu -lineinfo $(GENCODE_FLAGS) $(EXTRA_GLOBAL_FLAGS) -dc $(SRC)/testCaseGenerator.cu -ccbin $(CC) -o $(BUILD)/testCaseGenerator.o -I $(INCLUDE) 

run-25: $(TARGET)
	echo "25%---------------------"
	./$(TARGET) ./Imagens_de_Teste/25-percent-marker.pgm ./Imagens_de_Teste/25-percent-mask.jpg ./Imagens_de_Teste/res-cuda-25.tiff 0

run-50: $(TARGET)
	echo "50%---------------------"
	./$(TARGET) ./Imagens_de_Teste/50-percent-marker.jpg ./Imagens_de_Teste/50-percent-mask.jpg ./Imagens_de_Teste/res-cuda-50.tiff 0

run-75: $(TARGET)
	echo "75%---------------------"
	./$(TARGET) ./Imagens_de_Teste/75-percent-marker.jpg ./Imagens_de_Teste/75-percent-mask.jpg ./Imagens_de_Teste/res-cuda-75.tiff 0

run-100: $(TARGET)
	echo "100%---------------------"
	./$(TARGET) ./Imagens_de_Teste/100-percent-marker.jpg ./Imagens_de_Teste/100-percent-mask.jpg ./Imagens_de_Teste/res-cuda-100.tiff 0

memcheck:
	$(COMPUTE_SANITIZER_BIN) --tool memcheck --launch-timeout 0 $(TARGET) ./Imagens_de_Teste/75-percent-marker.jpg ./Imagens_de_Teste/75-percent-mask.jpg ./Imagens_de_Teste/res-cuda-75.tiff 0

racecheck:
	$(COMPUTE_SANITIZER_BIN) --tool racecheck $(TARGET)

initcheck:
	$(COMPUTE_SANITIZER_BIN) --tool initcheck $(TARGET)

synccheck:
	$(COMPUTE_SANITIZER_BIN) --tool synccheck $(TARGET)

visual-profiler:
	nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java



clean:
	rm -rf $(TARGET)
	rm -rf $(BUILD)/*.o 
