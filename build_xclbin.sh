#!/usr/bin/env bash
# build_xclbin.sh — Full hardware build script for fpga_mps_simulator
#
# Usage:
#   ./build_xclbin.sh           (hw — full Vivado impl, 2-4 hours)
#   ./build_xclbin.sh sw_emu    (software emulation, ~5 min)
#   ./build_xclbin.sh hw_emu    (hardware emulation, ~30 min)
#
# Produces:
#   build/fpga_mps_simulator.xo      (HLS kernel object)
#   fpga_mps_simulator.xclbin        (final deployable bitstream)
#
# Author: Nasir Ali
# Organization: C-DAC Noida
# Date: June 2026

set -euo pipefail

# ── Environment ────────────────────────────────────────────────────────────────
source /opt/xilinx/xrt/setup.sh
source /home/abhishek/tools/Vitis/2023.2/settings64.sh

PLATFORM="xilinx_u55c_gen3x16_xdma_3_202210_1"
FREQ=300
TARGET="${1:-hw}"
KERNEL=fpga_mps_simulator
SRC="${KERNEL}.cpp"
XO_DIR="build"
XO="${XO_DIR}/${KERNEL}.xo"
XCLBIN="${KERNEL}.xclbin"

echo "========================================================"
echo " FPGA MPS Simulator — Build"
echo " Target    : ${TARGET}"
echo " Platform  : ${PLATFORM}"
echo " Clock     : ${FREQ} MHz"
echo " Started   : $(date)"
echo "========================================================"

mkdir -p ${XO_DIR}

# ── Step 1: HLS Compile → .xo ─────────────────────────────────────────────────
echo ""
echo ">>> STEP 1: HLS compile (v++ --compile) → ${XO}"
echo "    Started: $(date)"

v++ \
    --compile \
    --target ${TARGET} \
    --platform ${PLATFORM} \
    --kernel ${KERNEL} \
    --hls.clock ${FREQ}MHz:${KERNEL} \
    --advanced.param compiler.enableHBM=true \
    --advanced.param compiler.worstNegativeSlack=-0.100 \
    --save-temps \
    -I. \
    -O3 \
    --output ${XO} \
    ${SRC} \
    2>&1 | tee ${XO_DIR}/compile.log

echo "    Compile done: $(date)"

# ── Step 2: Link → .xclbin ────────────────────────────────────────────────────
echo ""
echo ">>> STEP 2: Link (v++ --link) → ${XCLBIN}"
echo "    Started: $(date)"
echo "    NOTE: hw target Vivado implementation takes 2-4 hours."

v++ \
    --link \
    --target ${TARGET} \
    --platform ${PLATFORM} \
    --config connectivity.cfg \
    --kernel_frequency ${FREQ} \
    --save-temps \
    --output ${XCLBIN} \
    ${XO} \
    2>&1 | tee ${XO_DIR}/link.log

echo ""
echo "========================================================"
echo " Build COMPLETE"
echo " Output : ${XCLBIN}"
echo " Ended  : $(date)"
echo "========================================================"

ls -lh ${XCLBIN}
