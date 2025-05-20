#!/usr/bin/env bash

# Check if setup has already been run
if [ ! -d ".venv" ]; then
    echo ""
    echo "╔═════════════════════════════════════════════════════════╗"
    echo "║  🎉  W E L C O M E  T O  V I S U A L  A N A L Y T I C S ║"
    echo "╠═════════════════════════════════════════════════════════╣"
    echo "║           (\_/)                                         ║"
    echo "║          (>‿◠)♡     First time setup needed!            ║"
    echo "║           / |       Let's get your environment          ║"
    echo "║          /  \       ready for analysis!                 ║"
    echo "║                                                         ║"
    echo "║          [Y] Yes, set up now     [n] No, exit           ║"
    echo "╚═════════════════════════════════════════════════════════╝"
    read -p " → Your choice [Y/n]: " run_setup
    
    if [[ "$run_setup" == "" || "$run_setup" == "y" || "$run_setup" == "Y" ]]; then
        echo "🔄 Running initial setup..."
        bash ./setup.sh
        
        source .venv/bin/activate

    else
        echo "❌ Setup declined. Cannot proceed without virtual environment."
        echo "You can run setup manually later with: ./setup.sh"
        exit 1
    fi
else
    echo "✅ Setup already completed."
fi

# Activate the virtual environment
source .venv/bin/activate

# Main menu function
show_menu() {
    echo ""
    echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
    echo "┃                Visual Analytics                       ┃"
    echo "┃                Assignment Runner                      ┃"
    echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
    echo ""
    echo "Please select an assignment to run:"
    echo "1) Assignment 1"
    echo "2) Assignment 2"
    echo "3) Assignment 3"
    echo "4) Assignment 4"
    echo "q) Quit"
    echo ""
}

# Display assignment header
show_assignment_header() {
    echo ""
    echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
    echo "┃                      Assignment $1 Output                          ┃"
    echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
    echo ""
}

# Display assignment footer
show_assignment_footer() {
    echo ""
    echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
    echo "┃                    End of Assignment $1 Output                     ┃"
    echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
    echo ""
}

# Add this function near the top of the file with other display functions
show_dataset_missing() {
    local dataset=$1
    local message=$2
    
    echo ""
    echo "┌─────────────────────────────────────────────┐"
    echo "│  📦  Dataset Required: $dataset              │"
    echo "└─────────────────────────────────────────────┘"
    echo ""
    echo "$message"
    echo ""
}

# Run the selected assignment
run_assignment() {
    case $1 in
        1)
            show_assignment_header $1
            echo "🚀 Running Assignment 1 with default configuration..."
            
            # Check if flower dataset directory exists
            if [ ! -d "data/17flowers" ]; then
                show_dataset_missing "Flowers" "The 17 category flower dataset is required for this assignment."
                
                read -p "Do you want to download the dataset now? [Y/n]: " download_choice
                if [[ "$download_choice" == "" || "$download_choice" == "y" || "$download_choice" == "Y" ]]; then
                    (cd src && PYTHONPATH=. uv run -m assignment_1.scripts.download_data)
                    if [ $? -ne 0 ]; then
                        show_assignment_footer $1
                        return
                    fi
                else
                    echo "❌ Dataset download declined. Cannot proceed without dataset."
                    show_assignment_footer $1
                    return
                fi
            fi
            
            (cd src && uv run -m assignment_1.main)
            show_assignment_footer $1
            ;;
        2)
            show_assignment_header $1
            echo "🚀 Running Assignment 2 with default configuration..."
            (cd src && uv run -m assignment_2.main)
            show_assignment_footer $1
            ;;
        3)
            show_assignment_header $1
            echo "🚀 Running Assignment 3 with default configuration..."
            
            # Check if Lego dataset directory exists
            if [ ! -d "data/lego" ]; then
                show_dataset_missing "Lego" "Please download the Lego dataset from UCloud and place it in the data/lego directory."
                show_assignment_footer $1
                return
            fi
            
            (cd src && uv run -m assignment_3.main)
            show_assignment_footer $1
            ;;
        4)
            show_assignment_header $1
            echo "🚀 Running Assignment 4 with default configuration..."
            
            # Check if newspaper dataset directory exists
            if [ ! -d "data/newspapers/images" ]; then
                show_dataset_missing "Newspapers" "Please download the Swiss newspapers dataset from Zenodo and place it in the data/newspapers/images directory.\nDataset URL: https://zenodo.org/records/3706863"
                show_assignment_footer $1
                return
            fi
            
            (cd src && uv run -m assignment_4.main)
            show_assignment_footer $1
            ;;
        *)
            echo "❌ Invalid selection."
            ;;
    esac
}

# Main program
while true; do
    show_menu
    read -p "Enter your choice: " choice
    
    if [[ "$choice" == "q" || "$choice" == "Q" ]]; then
        echo "👋 Goodbye!"
        break
    fi
    
    run_assignment $choice
done
