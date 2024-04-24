#!/bin/bash

# Define paths and configuration
CONFIG_PATH="/P/bolt/data/bradhutc/vll-test/run/dnnntup.json"
OUTPUT_BASE="/P/bolt/data/bradhutc/vlq_framework/Truth_Analysis/ntuples"
SOURCE_DIR="/P/bolt/data/bradhutc/vll-test/source"

# Create the base output directory if it does not exist
mkdir -p "$OUTPUT_BASE"

# Setup environment and build
echo "Setting up the environment and building the application..."
cd /P/bolt/data/bradhutc/vll-test
mkdir -p build
cd build
source ../setup.sh
cmake "$SOURCE_DIR"
make -j12
source /P/bolt/data/bradhutc/vll-test/x86_64-centos7-gcc11-opt/setup.sh
cd ..

# Initialize a counter or timestamp
TIMESTAMP=$(date +%s)

# Loop over each line in filelist.txt
while IFS=',' read -r DAODFILE OUTPUT_NAME; do
    echo "Processing $DAODFILE as $OUTPUT_NAME..."
    cd /P/bolt/data/bradhutc/vll-test/
    # Run ana and wait a bit to ensure files are written
    ana -c "$CONFIG_PATH" -I "$DAODFILE" || { echo "ana command failed"; exit 1; }
    sleep 2  # Wait to make sure the filesystem updates (adjust time as needed)

    # Find the most recently created submitDir after running ana
    NEW_DIR=$(find /P/bolt/data/bradhutc/vll-test/ -maxdepth 1 -type d -name 'submitDir-*' -newermt @$TIMESTAMP -print | sort | tail -n 1)
    NEW_ROOT_FILE="$NEW_DIR/data-ANALYSIS/file.root"

    if [ -f "$NEW_ROOT_FILE" ]; then
        # Create the named output directory
        FINAL_DIR="$OUTPUT_BASE/$OUTPUT_NAME"
        mkdir -p "$FINAL_DIR"

        # Copy the ROOT file to the new directory
        cp "$NEW_ROOT_FILE" "$FINAL_DIR/"
        echo "ROOT file moved to $FINAL_DIR"
    else
        echo "No ROOT file found in $NEW_ROOT_FILE"
    fi

    # Update timestamp for the next iteration
    TIMESTAMP=$(date +%s)
done < /P/bolt/data/bradhutc/vlq_framework/Truth_Analysis/filelist.txt
