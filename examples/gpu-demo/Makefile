
SRC_DIR = src/
MY_INCLUDE = -Isrc/

WITH_CUDA = true

default:
	g++ ${MY_INCLUDE} -c -g main.cpp -o main.o
	g++ main.o -o run-cpu.exe
	
gpu:
	nvcc ${MY_INCLUDE} -o run-gpu.exe main.cu -Xcompiler -std=c++11

clean:
	rm *.o || echo "no .o files to remove"
	rm *.exe || echo "remove .exe files"