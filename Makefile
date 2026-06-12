# Makefile — FPGA MPS Quantum Circuit Simulator
#
# Targets:
#   make xo         — compile HLS kernel → .xo object
#   make xclbin     — link .xo → .xclbin (hw)
#   make sw_emu     — compile + link for software emulation
#   make hw_emu     — compile + link for hardware emulation
#   make test       — run Python test suite (cpu mode, no FPGA required)
#   make benchmark  — run benchmark.py (cpu mode)
#   make clean      — remove build artefacts
#   make cleanall   — remove build artefacts + .xclbin
#
# Author: Nasir Ali
# Organization: C-DAC Noida
# Date: June 2026
# Target: Xilinx Alveo U55C, Vitis 2023.2

# ── Platform + tool configuration ────────────────────────────────────────────
PLATFORM   ?= xilinx_u55c_gen3x16_xdma_3_202210_1
VITIS      ?= v++
FREQ       ?= 300
TARGET     ?= hw

KERNEL_SRC  = fpga_mps_simulator.cpp
KERNEL_XO   = fpga_mps_simulator.xo
KERNEL_NAME = fpga_mps_simulator

XCLBIN      = fpga_mps_simulator.xclbin
LINK_CFG    = connectivity.cfg

# ── v++ flags ────────────────────────────────────────────────────────────────
VPP_CFLAGS = \
    --platform $(PLATFORM) \
    --target $(TARGET) \
    --compile \
    --kernel $(KERNEL_NAME) \
    --hls.clock $(FREQ)MHz:$(KERNEL_NAME) \
    --advanced.param compiler.enableHBM=true \
    --advanced.param compiler.worstNegativeSlack=-0.100 \
    --save-temps \
    -I. \
    -O3

VPP_LFLAGS = \
    --platform $(PLATFORM) \
    --target $(TARGET) \
    --link \
    --config $(LINK_CFG) \
    --kernel_frequency $(FREQ) \
    --vivado.prop run.impl_1.strategy=Congestion_SpreadLogic_high \
    --vivado.prop run.synth_1.strategy=Flow_AreaOptimized_high \
    --save-temps

# ── Build targets ─────────────────────────────────────────────────────────────

.PHONY: all xo xclbin sw_emu hw_emu test benchmark clean cleanall

all: xclbin

xo: $(KERNEL_XO)

$(KERNEL_XO): $(KERNEL_SRC)
	$(VITIS) $(VPP_CFLAGS) --output $@ $<

xclbin: $(XCLBIN)

$(XCLBIN): $(KERNEL_XO) $(LINK_CFG)
	$(VITIS) $(VPP_LFLAGS) --output $@ $(KERNEL_XO)

sw_emu: TARGET=sw_emu
sw_emu: XCLBIN=fpga_mps_simulator_sw_emu.xclbin
sw_emu: KERNEL_XO=fpga_mps_simulator_sw_emu.xo
sw_emu:
	$(VITIS) $(VPP_CFLAGS) --target sw_emu --output fpga_mps_simulator_sw_emu.xo $(KERNEL_SRC)
	$(VITIS) $(VPP_LFLAGS) --target sw_emu --output fpga_mps_simulator_sw_emu.xclbin \
		fpga_mps_simulator_sw_emu.xo

hw_emu: TARGET=hw_emu
hw_emu:
	$(VITIS) $(VPP_CFLAGS) --target hw_emu --output fpga_mps_simulator_hw_emu.xo $(KERNEL_SRC)
	$(VITIS) $(VPP_LFLAGS) --target hw_emu --output fpga_mps_simulator_hw_emu.xclbin \
		fpga_mps_simulator_hw_emu.xo

# ── Software targets ──────────────────────────────────────────────────────────

test:
	python3 -m pytest tests/ -v --tb=short 2>&1 | tee test_results.txt

benchmark:
	python3 benchmarks/benchmark.py 2>&1 | tee benchmark_results.txt

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean:
	rm -rf _x .Xil *.log *.jou *.str *.info *.dat *.pb \
	       *.ltx *.xsa *.xo.* *.compile_summary *.link_summary \
	       test_results.txt benchmark_results.txt __pycache__ \
	       .pytest_cache */.pytest_cache

cleanall: clean
	rm -f $(KERNEL_XO) $(XCLBIN) \
	      fpga_mps_simulator_sw_emu.xo fpga_mps_simulator_sw_emu.xclbin \
	      fpga_mps_simulator_hw_emu.xo fpga_mps_simulator_hw_emu.xclbin
