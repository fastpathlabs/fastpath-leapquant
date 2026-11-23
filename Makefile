# Makefile for LeapQuant: the reference core, the batched kernel and their tests.
#
#   make                    build the reference tools and the CPU kernel libraries
#   make test               reference core tests (about 10 seconds)
#   make selftest           check the trace recorder and the replay tool against each other
#   make kernel-test        batched kernel against the reference core
#   make kernel-race-check  batched kernel as an emulated GPU thread block, under ThreadSanitizer
#   make kernel-sanitize    kernel test under AddressSanitizer and UndefinedBehaviorSanitizer
#   make kernel-selftest    the Python tools and the engine glue against the reference (needs torch, numpy)
#   make check              all of the above
#   make cuda               the GPU kernel library (needs nvcc; run this on the GPU machine)
#   make cuda-syntax-check  parse the GPU source with clang (no nvcc needed)
#   make clean              remove what was built
#
# Everything is built with -ffp-contract=off. With it, a*b+c is computed as a
# rounded multiply and a rounded add on every compiler and CPU, so the reference
# core gives the same numbers everywhere, and the batched kernel can be required
# to match it bit for bit.

CXX       ?= g++
CXXFLAGS  ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic -Wshadow
FPFLAGS    = -ffp-contract=off
PYTHON    ?= python3

# GPU build. FMAD=false keeps the GPU from fusing multiply and add, like FPFLAGS
# above. Try FMAD=true for speed once the kernel is verified, and verify again.
NVCC      ?= nvcc
FMAD      ?= false
# -arch=sm_90: Hopper (H100, H200). Stated, so that the binary does not depend on nvcc's default.
CUDA_ARCH ?= sm_90
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH)
# Tuning options of the GPU build (kernel/tune_kernel.py tries the combinations):
#   CUDA_THREADS  threads per block: 64, 128 or 256
#   WORD_LOADS    1 = read the int8 matrix 8 bytes at a time
#   CUDA_LIB      output file, so that several variants can sit side by side
CUDA_THREADS ?= 128
WORD_LOADS   ?= 1
CUDA_LIB     ?= $(BUILD)/libleapquant_cuda.so
#   FOLD_SHARED   1 = the fold keeps its matrix in the block's shared memory (default), 0 = in scratch
FOLD_SHARED  ?= 1
CUDA_DEFINES  = -DLQ_CUDA_THREADS_PER_BLOCK=$(CUDA_THREADS) -DLQ_FOLD_IN_SHARED=$(FOLD_SHARED) $(if $(filter 1,$(WORD_LOADS)),-DLQ_WORD_LOADS,)

BUILD = build
RESULTS ?= results/run-creative-franklin
ifeq ($(shell uname -s),Darwin)
LIB = dylib
else
LIB = so
endif

CORE = leapquant.cpp reference_heads.cpp
HEADERS = leapquant.h reference_heads.h leapquant_small_matrix.h
KERNEL_HEADERS = kernel/leapquant_kernel.h kernel/leapquant_kernel_blocks.h
KERNEL_TEST_SOURCES = $(CORE) kernel/leapquant_kernel_cpu.cpp kernel/kernel_test.cpp
KERNEL_TEST_FLAGS = $(CXXFLAGS) $(FPFLAGS) -DLQ_EXPECT_BIT_IDENTICAL -pthread

all: leapquant_test leapquant_replay kernel

# ---- reference core ----------------------------------------------------------
leapquant_test: leapquant_test.cpp $(CORE) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(FPFLAGS) $(CORE) leapquant_test.cpp -o leapquant_test

leapquant_replay: leapquant_replay.cpp $(CORE) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(FPFLAGS) $(CORE) leapquant_replay.cpp -o leapquant_replay

test: leapquant_test
	./leapquant_test

selftest: leapquant_replay
	$(PYTHON) record_trace.py --self-test selftest.bin
	./leapquant_replay selftest.bin

# ---- batched kernel, CPU -------------------------------------------------------
kernel: $(BUILD)/libleapquant_cpu.$(LIB) $(BUILD)/libleapquant_reference.$(LIB)

$(BUILD)/libleapquant_cpu.$(LIB): kernel/leapquant_kernel_cpu.cpp $(KERNEL_HEADERS)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(FPFLAGS) -fPIC -shared -pthread kernel/leapquant_kernel_cpu.cpp -o $@

$(BUILD)/libleapquant_reference.$(LIB): kernel/reference_shim.cpp $(CORE) $(HEADERS)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(FPFLAGS) -fPIC -shared $(CORE) kernel/reference_shim.cpp -o $@

$(BUILD)/kernel_test: $(KERNEL_TEST_SOURCES) $(HEADERS) $(KERNEL_HEADERS)
	@mkdir -p $(BUILD)
	$(CXX) $(KERNEL_TEST_FLAGS) $(KERNEL_TEST_SOURCES) -o $@

# The same test with the tuning option LQ_WORD_LOADS: it must stay bit-identical.
$(BUILD)/kernel_test_words: $(KERNEL_TEST_SOURCES) $(HEADERS) $(KERNEL_HEADERS)
	@mkdir -p $(BUILD)
	$(CXX) $(KERNEL_TEST_FLAGS) -DLQ_WORD_LOADS $(KERNEL_TEST_SOURCES) -o $@

kernel-test: $(BUILD)/kernel_test $(BUILD)/kernel_test_words
	$(BUILD)/kernel_test
	$(BUILD)/kernel_test_words | tail -1

$(BUILD)/kernel_test_tsan: $(KERNEL_TEST_SOURCES) $(HEADERS) $(KERNEL_HEADERS)
	@mkdir -p $(BUILD)
	$(CXX) $(KERNEL_TEST_FLAGS) -O1 -g -fsanitize=thread $(KERNEL_TEST_SOURCES) -o $@

kernel-race-check: $(BUILD)/kernel_test_tsan
	$(NO_ASLR) $(BUILD)/kernel_test_tsan --emulate-only

$(BUILD)/kernel_test_asan: $(KERNEL_TEST_SOURCES) $(HEADERS) $(KERNEL_HEADERS)
	@mkdir -p $(BUILD)
	$(CXX) $(KERNEL_TEST_FLAGS) -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined $(KERNEL_TEST_SOURCES) -o $@

# On Linux the sanitized binaries run with address-space randomisation off (setarch -R). With the
# kernel's default of 32 random bits, AddressSanitizer of gcc 13 never starts: it prints
# "DEADLYSIGNAL" for ever. Seen on Ubuntu 22.04 with kernel 6.8, where it hung the GPU suite.
ifeq ($(shell uname -s),Linux)
NO_ASLR = setarch -R
endif
kernel-sanitize: $(BUILD)/kernel_test_asan
	$(NO_ASLR) $(BUILD)/kernel_test_asan

# The Python tools against the reference, on the synthetic trace. Needs torch and numpy.
kernel-selftest: kernel leapquant_replay
	$(PYTHON) record_trace.py --self-test selftest.bin
	./leapquant_replay selftest.bin --sweeps 2 --write-outputs $(BUILD)/selftest_outputs.bin > /dev/null
	$(PYTHON) kernel/verify_kernel.py selftest.bin --device cpu --rows 3 --reference-outputs $(BUILD)/selftest_outputs.bin
	$(PYTHON) kernel/gates_test.py
	$(PYTHON) kernel/engine_test.py
	$(PYTHON) kernel/block_engine_test.py
	$(PYTHON) kernel/engine_test.py --fused
	$(PYTHON) kernel/block_engine_test.py --fused

check: test selftest kernel-test kernel-sanitize kernel-race-check kernel-selftest

# ---- batched kernel, GPU ---------------------------------------------------------
cuda:
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) -fmad=$(FMAD) $(CUDA_DEFINES) -shared -Xcompiler -fPIC kernel/leapquant_kernel_cuda.cu -o $(CUDA_LIB)

# The serving path's fold, written by hand for sm_90 (kernel/leapquant_fast_fold.cu; used when
# LEAPQUANT_FOLD_BACKEND=cuda). Not bit-exact by design, so FMA contraction stays on. -Xptxas -v prints
# registers, spills and shared memory per kernel: the resource record of every build.
FASTFOLD_LIB ?= $(BUILD)/libleapquant_fastfold.so
fast-fold:
	@mkdir -p $(BUILD)
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) -lineinfo -Xptxas -v -shared -Xcompiler -fPIC \
		kernel/leapquant_fast_fold.cu kernel/leapquant_fast_decode.cu -o $(FASTFOLD_LIB) > $(BUILD)/fast_fold_resources.txt 2>&1; \
		status=$$?; cat $(BUILD)/fast_fold_resources.txt; exit $$status

# Parses the CUDA source with clang, for machines without nvcc. See the script.
cuda-syntax-check:
	kernel/check_cuda_syntax.sh

clean-cuda:
	rm -f $(BUILD)/libleapquant_cuda.so

clean:
	rm -f leapquant_test leapquant_replay selftest.bin
	rm -rf $(BUILD)

.PHONY: all test selftest kernel kernel-test kernel-race-check kernel-sanitize kernel-selftest check cuda fast-fold cuda-syntax-check clean-cuda clean

reports:
	python3 reports/generate_reports.py $(RESULTS)
