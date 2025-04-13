#!/usr/bin/env bash
set -e

# Function to check if dataset exists and download if needed
check_and_download_dataset() {
    if [ ! -d "../../data/17flowers" ]; then
        echo "⚠️ Flower dataset not found."
        read -p "Do you want to download the dataset now? [Y/n]: " download_dataset
        
        if [[ "$download_dataset" == "" || "$download_dataset" == "y" || "$download_dataset" == "Y" ]]; then
            echo "🔄 Downloading dataset..."
            cd ../
            PYTHONPATH=. uv run -m assignment_1.download_data
            status=$?
            cd - > /dev/null
            
            if [ $status -ne 0 ]; then
                echo "❌ Failed to download dataset."
                return 1
            fi
        else
            echo "❌ Dataset download declined. Assignment 1 requires the dataset to run."
            return 1
        fi
    fi
    return 0
}

# Run the check
check_and_download_dataset
exit $?