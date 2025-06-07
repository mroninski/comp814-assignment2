#!/bin/bash

# Build script for comp814.tex
# This script compiles the LaTeX document with bibliography support

# Note: We don't use 'set -e' because LaTeX warnings shouldn't stop the build
# set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Build configuration
BUILD_DIR="build"
OUTPUT_DIR="output"
MAIN_FILE="comp814"

echo -e "${YELLOW}Building ${MAIN_FILE}.tex...${NC}"

# Check if required files exist
if [ ! -f "${MAIN_FILE}.tex" ]; then
    echo -e "${RED}Error: ${MAIN_FILE}.tex not found${NC}"
    exit 1
fi

if [ ! -f "references.bib" ]; then
    echo -e "${RED}Error: references.bib not found${NC}"
    exit 1
fi

# Create build and output directories
echo -e "${YELLOW}Setting up build directories...${NC}"
mkdir -p "${BUILD_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Clean previous builds
echo -e "${YELLOW}Cleaning previous build files...${NC}"
rm -rf "${BUILD_DIR}"/*
rm -f "${OUTPUT_DIR}/${MAIN_FILE}.pdf"

# Copy source files to build directory
cp "${MAIN_FILE}.tex" "${BUILD_DIR}/"
cp "references.bib" "${BUILD_DIR}/"

# Copy figures if they exist
for img in *.png *.jpg *.jpeg *.pdf *.eps; do
    if [ -f "$img" ]; then
        cp "$img" "${BUILD_DIR}/"
    fi
done 2>/dev/null || true

# Change to build directory
cd "${BUILD_DIR}"

# First LaTeX run
echo -e "${YELLOW}Running pdflatex (1st pass)...${NC}"
pdflatex -interaction=nonstopmode "${MAIN_FILE}.tex"
LATEX_EXIT_CODE=$?
if [ $LATEX_EXIT_CODE -ne 0 ] && [ ! -f "${MAIN_FILE}.pdf" ]; then
    echo -e "${RED}Fatal error in first LaTeX pass - no PDF generated${NC}"
    cd ..
    exit 1
elif [ $LATEX_EXIT_CODE -ne 0 ]; then
    echo -e "${YELLOW}LaTeX warnings/errors in first pass, but PDF generated - continuing...${NC}"
fi

# Process bibliography with biber (for biblatex)
echo -e "${YELLOW}Processing bibliography with biber...${NC}"
biber "${MAIN_FILE}"
BIBER_EXIT_CODE=$?
if [ $BIBER_EXIT_CODE -ne 0 ]; then
    echo -e "${YELLOW}Biber warnings/errors - this is normal for first run${NC}"
fi

# Second LaTeX run (incorporate bibliography)
echo -e "${YELLOW}Running pdflatex (2nd pass)...${NC}"
pdflatex -interaction=nonstopmode "${MAIN_FILE}.tex"
LATEX_EXIT_CODE=$?
if [ $LATEX_EXIT_CODE -ne 0 ] && [ ! -f "${MAIN_FILE}.pdf" ]; then
    echo -e "${RED}Fatal error in second LaTeX pass - no PDF generated${NC}"
    cd ..
    exit 1
elif [ $LATEX_EXIT_CODE -ne 0 ]; then
    echo -e "${YELLOW}LaTeX warnings/errors in second pass, but PDF generated - continuing...${NC}"
fi

# Third LaTeX run (fix cross-references)
echo -e "${YELLOW}Running pdflatex (3rd pass)...${NC}"
pdflatex -interaction=nonstopmode "${MAIN_FILE}.tex"
LATEX_EXIT_CODE=$?
if [ $LATEX_EXIT_CODE -ne 0 ] && [ ! -f "${MAIN_FILE}.pdf" ]; then
    echo -e "${RED}Fatal error in third LaTeX pass - no PDF generated${NC}"
    cd ..
    exit 1
elif [ $LATEX_EXIT_CODE -ne 0 ]; then
    echo -e "${YELLOW}LaTeX warnings/errors in third pass, but PDF generated - continuing...${NC}"
fi

# Move final PDF to output directory
if [ -f "${MAIN_FILE}.pdf" ]; then
    mv "${MAIN_FILE}.pdf" "../${OUTPUT_DIR}/"
    cd ..
    echo -e "${GREEN}✓ Successfully built ${MAIN_FILE}.pdf${NC}"
    echo -e "${GREEN}File size: $(ls -lh ${OUTPUT_DIR}/${MAIN_FILE}.pdf | awk '{print $5}')${NC}"
    echo -e "${GREEN}Output location: ${OUTPUT_DIR}/${MAIN_FILE}.pdf${NC}"
else
    cd ..
    echo -e "${RED}✗ Failed to create ${MAIN_FILE}.pdf${NC}"
    exit 1
fi

# Optional: Open PDF if on macOS
if command -v open &> /dev/null && [[ "$OSTYPE" == "darwin"* ]]; then
    read -p "Open PDF? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open "${OUTPUT_DIR}/${MAIN_FILE}.pdf"
    fi
fi

echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${GREEN}Clean source directory maintained - all build files in '${BUILD_DIR}' folder${NC}" 