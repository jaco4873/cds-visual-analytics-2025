#!/usr/bin/env bash
set -e

# Check if setup has already been run
if [ ! -d ".venv" ]; then
    echo "⚠️ Virtual environment not found."
    read -p "Do you want to run the setup script now? [Y/n]: " run_setup
    
    if [[ "$run_setup" == "" || "$run_setup" == "y" || "$run_setup" == "Y" ]]; then
        echo "🔄 Running initial setup..."
        bash ./setup.sh
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

# Run the selected assignment
run_assignment() {
    case $1 in
        1)
            show_assignment_header $1
            echo "🚀 Running Assignment 1 with default configuration..."
            
            # Check and download dataset if needed
            (cd src/assignment_1 && ./check_and_download_data.sh)
            if [ $? -ne 0 ]; then
                show_assignment_footer $1
                return
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
                echo "⚠️ Lego dataset not found in data/lego directory."
                echo "Please download the Lego dataset from UCloud and place it in the data/lego directory."
                show_assignment_footer $1
                return
            fi
            
            (cd src && uv run -m assignment_3.main)
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
