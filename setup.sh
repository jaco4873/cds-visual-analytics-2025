#!/usr/bin/env bash
echo "🚀 Starting Visual Analytics setup..."

# Detect the operating system
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📱 macOS detected"
    OS_TYPE="macos"
elif [[ "$OSTYPE" == "linux"* ]]; then
    echo "🐧 Linux detected"
    OS_TYPE="linux"
else
    echo "❌ Unsupported operating system: $OSTYPE"
    echo "This script only supports macOS and Linux."
    exit 1  
fi

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to the current PATH
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Verify uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Failed to install uv. Please install it manually:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1  
fi

echo "✅ uv is installed: $(uv --version)"

# Install system-level dependencies
echo "📦 Checking system dependencies..."

# Function to check if dependencies are already installed
check_dependencies() {
    # Check for OpenCV
    if pkg-config --exists opencv4 || pkg-config --exists opencv; then
        echo "✅ OpenCV is already installed"
        OPENCV_INSTALLED=true
    else
        OPENCV_INSTALLED=false
    fi
    
    # Check for Tesseract
    if command -v tesseract &> /dev/null; then
        echo "✅ Tesseract is already installed"
        TESSERACT_INSTALLED=true
    else
        TESSERACT_INSTALLED=false
    fi
    
    # Return true if both are installed
    if [[ "$OPENCV_INSTALLED" == true && "$TESSERACT_INSTALLED" == true ]]; then
        return 0
    else
        return 1
    fi
}

# Check if dependencies are already installed
if check_dependencies; then
    echo "✅ All required dependencies are already installed"
else
    echo "📦 Some dependencies need to be installed"
    
    # Check if Homebrew needs to be installed on macOS
    if ! command -v brew &> /dev/null && [[ "$OS_TYPE" == "macos" ]]; then
        read -p "Homebrew is not installed. Do you want to install it? (y/n): " install_brew
        if [[ "$install_brew" == "y" || "$install_brew" == "Y" ]]; then
            echo "📦 Installing Homebrew package manager..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        else
            echo "⚠️ Homebrew installation declined. You will need to install missing dependencies manually."
        fi
    fi
    
    # Install packages based on available package manager
    if command -v brew &> /dev/null; then
        echo "📦 Using Homebrew package manager"
        if [[ "$OPENCV_INSTALLED" == false ]]; then
            echo "Installing OpenCV..."
            brew install opencv
        fi
        if [[ "$TESSERACT_INSTALLED" == false ]]; then
            echo "Installing Tesseract..."
            brew install tesseract
        fi
    elif [[ "$OS_TYPE" == "linux" ]] && command -v apt-get &> /dev/null; then
        echo "📦 Using apt package manager"
        sudo apt-get update
        if [[ "$OPENCV_INSTALLED" == false || "$TESSERACT_INSTALLED" == false ]]; then
            sudo apt-get install -y tesseract-ocr libtesseract-dev libgl1
        fi
    else
        # This will trigger for macOS without Homebrew or Linux without apt-get
        echo "⚠️ No supported package manager found. Please install these dependencies manually:"
        if [[ "$OPENCV_INSTALLED" == false ]]; then
            echo "- OpenCV"
        fi
        if [[ "$TESSERACT_INSTALLED" == false ]]; then
            echo "- Tesseract"
        fi
        
        if [[ "$OS_TYPE" == "macos" ]]; then
            echo "On macOS, you can install them using Homebrew, MacPorts, or from source."
        elif [[ "$OS_TYPE" == "linux" ]]; then
            echo "On Linux, you can use your distribution's package manager."
        fi
        
        read -p "Press Enter to continue setup after you've installed the dependencies..."
    fi
    
    # Verify dependencies after installation
    echo "Verifying dependencies..."
    check_dependencies
fi

# Create virtual environment with uv
echo "🔨 Creating virtual environment with Python 3.12..."
uv venv --python=3.12

# Activate the environment directly
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies with uv
echo "📚 Installing project dependencies..."
uv sync

# Create necessary directories
echo "📁 Creating required directories..."
mkdir -p data

echo "✅ Setup completed successfully!"

echo ""
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
echo "┃                                                                ┃"
echo "┃   🌟 🌟 🌟  SETUP COMPLETE  🌟 🌟 🌟                           ┃"
echo "┃                                                                ┃"
echo "┃   ⚠️  For future sessions, activate with:                       ┃"
echo "┃      source .venv/bin/activate                                 ┃"
echo "┃                                                                ┃"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
echo ""
