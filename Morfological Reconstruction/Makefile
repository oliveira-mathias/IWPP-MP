CC = /usr/bin/g++-10
NVCC = nvcc

GENCODE_FLAGS = -arch=compute_50 -gencode arch=compute_50,code=sm_50

EXTRA_GLOBAL_FLAGS = -g

COMPUTE_SANITIZER_BIN = /usr/local/cuda/compute-sanitizer/compute-sanitizer

OPENCV = /usr/local/include/opencv4/
INCLUDE = ./include
BUILD = ./build
SRC = ./src
TARGET = exec

$(TARGET): $(BUILD)/main.o $(BUILD)/globalQueue.o $(BUILD)/MorphologicalReconstruction.o $(BUILD)/testCaseGenerator.o
	$(NVCC) $(GENCODE_FLAGS) $(BUILD)/*.o  $(EXTRA_GLOBAL_FLAGS) -o $(TARGET) -I $(INCLUDE) -lcurand -lopencv_core -lopencv_imgcodecs

$(BUILD)/main.o: $(SRC)/main.cu
	$(NVCC) -x cu $(GENCODE_FLAGS)  $(EXTRA_GLOBAL_FLAGS) -dc $(SRC)/main.cu -ccbin $(CC) -o $(BUILD)/main.o -I $(INCLUDE) -I$(OPENCV)

$(BUILD)/globalQueue.o: $(SRC)/globalQueue.cu $(INCLUDE)/globalQueue.cuh
	$(NVCC) -x cu $(GENCODE_FLAGS) $(EXTRA_GLOBAL_FLAGS) -dc $(SRC)/globalQueue.cu -ccbin $(CC) -o $(BUILD)/globalQueue.o -I $(INCLUDE) 

$(BUILD)/MorphologicalReconstruction.o: $(SRC)/MorphologicalReconstruction.cu $(INCLUDE)/MorphologicalReconstruction.cuh
	$(NVCC) -x cu $(GENCODE_FLAGS) $(EXTRA_GLOBAL_FLAGS) -dc $(SRC)/MorphologicalReconstruction.cu -ccbin $(CC) -o $(BUILD)/MorphologicalReconstruction.o -I $(INCLUDE) 

$(BUILD)/testCaseGenerator.o: $(SRC)/testCaseGenerator.cu $(INCLUDE)/testCaseGenerator.cuh
	$(NVCC) -x cu $(GENCODE_FLAGS) $(EXTRA_GLOBAL_FLAGS) -dc $(SRC)/testCaseGenerator.cu -ccbin $(CC) -o $(BUILD)/testCaseGenerator.o -I $(INCLUDE) 

mem-check:
	$(COMPUTE_SANITIZER_BIN) --tool memcheck --launch-timeout 0 $(TARGET)

race-check:
	$(COMPUTE_SANITIZER_BIN) --tool racecheck $(TARGET)

init-check:
	$(COMPUTE_SANITIZER_BIN) --tool initcheck $(TARGET)

sync-check:
	$(COMPUTE_SANITIZER_BIN) --tool synccheck $(TARGET)

visual-profiler:
	nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java



clean:
	rm -rf $(TARGET)
	rm -rf $(BUILD)/*.o 
