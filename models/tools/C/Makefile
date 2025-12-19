SRC_DIR      := src
INC_DIR      := include
BUILD_DIR    := build
CPFLOAT      := dependencies/cpfloat/src
LIBOMP_PREFIX := $(shell brew --prefix libomp)
MATLAB_BIN="/Applications/MATLAB_R2026a.app/bin/maca64"

MEX_NAME     := gemm
MEX_SRC      := $(SRC_DIR)/gemm.c

SOURCES      := $(wildcard $(SRC_DIR)/*.c)

MEX          = /Applications/MATLAB_R2026a.app/bin/mex

INCLUDES     := -I$(INC_DIR) -I$(CPFLOAT) -I"${LIBOMP_PREFIX}/include"

MEXFLAGS     := -R2018a

CFLAGS_OPT   := COPTIMFLAGS="\$$COPTIMFLAGS -O3 -DNDEBUG"

CFLAGS_OMP   := CFLAGS="\$$CFLAGS -Xpreprocessor -fopenmp" LDFLAGS="\$$LDFLAGS -Wl,-rpath, ${MATLAB_BIN} -L${MATLAB_BIN} -lomp"

# Final flag set (append optional OpenMP if you enabled it)
EXTRA_FLAGS  := $(CFLAGS_OPT) $(CFLAGS_OMP)

OUTFILE      := $(BUILD_DIR)/$(MEX_NAME)

.PHONY: all clean

all: $(OUTFILE)

# Build target: use mex to compile and link all sources into a single MEX.
# Mex will write build/gemm.mex* (platform-specific extension).
$(OUTFILE): $(SOURCES) $(INC_DIR)/gemm.h | $(BUILD_DIR)
	$(MEX) $(MEXFLAGS) $(INCLUDES) $(EXTRA_FLAGS) -outdir $(BUILD_DIR) -output $(MEX_NAME) $(SOURCES)

# Ensure build directory exists
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Clean: remove build artifacts
clean:
	@rm -rf $(BUILD_DIR)
