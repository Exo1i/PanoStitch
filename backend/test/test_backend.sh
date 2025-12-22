#!/bin/bash
# Backend Testing Script

set -e

echo "======================================"
echo "PanoStitch Backend Testing Script"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
API_URL="http://localhost:5000/api"
TEST_IMAGES_DIR="./imgs/tree"
TIMEOUT=300

# Function to check if server is running
check_server() {
    echo -e "${YELLOW}[1/5] Checking if server is running...${NC}"
    
    if ! curl -s "$API_URL/health" > /dev/null 2>&1; then
        echo -e "${RED}✗ Server is not running!${NC}"
        echo "Start the server with: python app.py"
        exit 1
    fi
    echo -e "${GREEN}✓ Server is running${NC}"
    echo ""
}

# Function to test health endpoint
test_health() {
    echo -e "${YELLOW}[2/5] Testing /api/health endpoint...${NC}"
    
    response=$(curl -s "$API_URL/health")
    echo "$response" | grep -q "healthy"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Health check passed${NC}"
        echo "Response: $response"
    else
        echo -e "${RED}✗ Health check failed${NC}"
        echo "Response: $response"
        exit 1
    fi
    echo ""
}

# Function to test info endpoint
test_info() {
    echo -e "${YELLOW}[3/5] Testing /api/info endpoint...${NC}"
    
    response=$(curl -s "$API_URL/info")
    echo "$response" | grep -q "PanoStitch"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Info endpoint working${NC}"
    else
        echo -e "${RED}✗ Info endpoint failed${NC}"
        exit 1
    fi
    echo ""
}

# Function to test stitch endpoint
test_stitch() {
    echo -e "${YELLOW}[4/5] Testing /api/stitch endpoint...${NC}"
    
    if [ ! -d "$TEST_IMAGES_DIR" ]; then
        echo -e "${RED}✗ Test images directory not found: $TEST_IMAGES_DIR${NC}"
        echo "Make sure you have images in the imgs/tree directory"
        exit 1
    fi
    
    # Find image files
    images=$(find "$TEST_IMAGES_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | head -3)
    
    if [ -z "$images" ]; then
        echo -e "${RED}✗ No images found in $TEST_IMAGES_DIR${NC}"
        exit 1
    fi
    
    echo "Found images:"
    echo "$images" | while read img; do
        echo "  - $(basename $img)"
    done
    echo ""
    
    # Build curl command
    cmd="curl -X POST \"$API_URL/stitch\" --max-time $TIMEOUT"
    
    while IFS= read -r img; do
        cmd="$cmd -F \"files=@$img\""
    done <<< "$images"
    
    cmd="$cmd -F \"use_dnn=false\" -F \"resize=800\""
    
    echo "Sending stitching request..."
    echo "(This may take a few minutes...)"
    echo ""
    
    response=$(eval $cmd)
    
    echo "$response" | grep -q "success"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Stitching request successful${NC}"
        echo ""
        echo "Response:"
        echo "$response" | python -m json.tool 2>/dev/null || echo "$response"
    else
        echo -e "${RED}✗ Stitching request failed${NC}"
        echo "Response:"
        echo "$response" | python -m json.tool 2>/dev/null || echo "$response"
        exit 1
    fi
    echo ""
}

# Function to test with Python client
test_python_client() {
    echo -e "${YELLOW}[5/5] Testing with Python client...${NC}"
    
    if command -v python3 &> /dev/null; then
        python3 api_client.py health
        echo ""
    else
        echo -e "${YELLOW}⊘ Python not available for client test${NC}"
    fi
}

# Main execution
main() {
    check_server
    test_health
    test_info
    test_stitch
    test_python_client
    
    echo -e "${GREEN}======================================"
    echo "All tests completed successfully!"
    echo "======================================${NC}"
}

# Run main function
main
