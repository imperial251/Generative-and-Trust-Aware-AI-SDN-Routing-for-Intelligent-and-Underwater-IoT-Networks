#!/bin/bash

# =============================================================================
# NS-3 Network Simulator Installation Script for SecureRouteX
# =============================================================================
# This script installs NS-3 and all dependencies for IoT network simulation
# Supports: Ubuntu 20.04+, Debian 11+, CentOS 8+, Fedora 35+
# Author: SecureRouteX Research Team
# Date: October 2025
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Detect OS
detect_os() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS=$NAME
        VERSION=$VERSION_ID
    elif type lsb_release >/dev/null 2>&1; then
        OS=$(lsb_release -si)
        VERSION=$(lsb_release -sr)
    elif [[ -f /etc/redhat-release ]]; then
        OS="CentOS"
        VERSION=$(cat /etc/redhat-release | grep -oE '[0-9]+\.[0-9]+')
    else
        print_error "Cannot detect operating system"
        exit 1
    fi
    
    print_info "Detected OS: $OS $VERSION"
}

# Check if running as root
check_sudo() {
    if [[ $EUID -eq 0 ]]; then
        print_warning "Running as root. Consider using a regular user with sudo privileges."
    fi
    
    if ! sudo -n true 2>/dev/null; then
        print_info "This script requires sudo privileges for package installation."
        sudo -v || { print_error "Sudo access required"; exit 1; }
    fi
}

# Update system packages
update_system() {
    print_header "UPDATING SYSTEM PACKAGES"
    
    case "$OS" in
        *"Ubuntu"*|*"Debian"*)
            sudo apt update && sudo apt upgrade -y
            print_success "System updated (APT)"
            ;;
        *"CentOS"*|*"Red Hat"*|*"Rocky"*)
            sudo yum update -y
            print_success "System updated (YUM)"
            ;;
        *"Fedora"*)
            sudo dnf update -y
            print_success "System updated (DNF)"
            ;;
        *)
            print_warning "Unknown package manager. Please update manually."
            ;;
    esac
}

# Install basic dependencies
install_basic_deps() {
    print_header "INSTALLING BASIC DEPENDENCIES"
    
    case "$OS" in
        *"Ubuntu"*|*"Debian"*)
            sudo apt install -y \
                build-essential \
                gcc \
                g++ \
                cmake \
                make \
                git \
                wget \
                curl \
                unzip \
                vim \
                python3 \
                python3-pip \
                python3-dev \
                python3-setuptools \
                libssl-dev \
                libffi-dev \
                zlib1g-dev \
                libbz2-dev \
                libreadline-dev \
                libsqlite3-dev \
                libncurses5-dev \
                libncursesw5-dev \
                xz-utils \
                tk-dev
            print_success "Basic dependencies installed (APT)"
            ;;
        *"CentOS"*|*"Red Hat"*|*"Rocky"*)
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y \
                gcc \
                gcc-c++ \
                cmake \
                make \
                git \
                wget \
                curl \
                unzip \
                vim \
                python3 \
                python3-pip \
                python3-devel \
                openssl-devel \
                libffi-devel \
                zlib-devel \
                bzip2-devel \
                readline-devel \
                sqlite-devel \
                ncurses-devel
            print_success "Basic dependencies installed (YUM)"
            ;;
        *"Fedora"*)
            sudo dnf groupinstall -y "Development Tools"
            sudo dnf install -y \
                gcc \
                gcc-c++ \
                cmake \
                make \
                git \
                wget \
                curl \
                unzip \
                vim \
                python3 \
                python3-pip \
                python3-devel \
                openssl-devel \
                libffi-devel \
                zlib-devel \
                bzip2-devel \
                readline-devel \
                sqlite-devel \
                ncurses-devel
            print_success "Basic dependencies installed (DNF)"
            ;;
    esac
}

# Install NS-3 specific dependencies
install_ns3_deps() {
    print_header "INSTALLING NS-3 SPECIFIC DEPENDENCIES"
    
    case "$OS" in
        *"Ubuntu"*|*"Debian"*)
            sudo apt install -y \
                gir1.2-goocanvas-2.0 \
                gir1.2-gtk-3.0 \
                libboost-all-dev \
                libgtk-3-dev \
                libxml2 \
                libxml2-dev \
                libxmlsec1-dev \
                qtbase5-dev \
                qtchooser \
                qt5-qmake \
                qtbase5-dev-tools \
                openmpi-bin \
                openmpi-common \
                openmpi-doc \
                libopenmpi-dev \
                autoconf \
                cvs \
                bzr \
                unrar \
                gdb \
                valgrind \
                gsl-bin \
                libgsl-dev \
                libgslcblas0 \
                wireshark \
                tcpdump \
                sqlite3 \
                libsqlite3-dev \
                libxml2-utils \
                libgtk2.0-0 \
                libgtk2.0-dev \
                vtun \
                lxc-utils \
                lxc-templates \
                vtun \
                uml-utilities \
                ebtables \
                bridge-utils \
                libpcap-dev \
                doxygen \
                graphviz \
                imagemagick \
                texlive \
                texlive-extra-utils \
                texlive-latex-extra \
                texlive-font-utils \
                dvipng \
                latexmk
            print_success "NS-3 dependencies installed (APT)"
            ;;
        *"CentOS"*|*"Red Hat"*|*"Rocky"*)
            # Enable EPEL repository
            sudo yum install -y epel-release
            
            sudo yum install -y \
                boost-devel \
                gtk3-devel \
                libxml2 \
                libxml2-devel \
                qt5-qtbase-devel \
                openmpi \
                openmpi-devel \
                autoconf \
                cvs \
                bzr \
                gdb \
                valgrind \
                gsl-devel \
                wireshark \
                tcpdump \
                sqlite \
                sqlite-devel \
                libpcap-devel \
                doxygen \
                graphviz \
                ImageMagick \
                texlive
            print_success "NS-3 dependencies installed (YUM)"
            ;;
        *"Fedora"*)
            sudo dnf install -y \
                boost-devel \
                gtk3-devel \
                libxml2 \
                libxml2-devel \
                qt5-qtbase-devel \
                openmpi \
                openmpi-devel \
                autoconf \
                cvs \
                bzr \
                gdb \
                valgrind \
                gsl-devel \
                wireshark \
                tcpdump \
                sqlite \
                sqlite-devel \
                libpcap-devel \
                doxygen \
                graphviz \
                ImageMagick \
                texlive
            print_success "NS-3 dependencies installed (DNF)"
            ;;
    esac
}

# Install Python packages for NS-3
install_python_packages() {
    print_header "INSTALLING PYTHON PACKAGES FOR NS-3 AND SECUREROUTEX"
    
    # Upgrade pip
    python3 -m pip install --upgrade pip
    
    # Essential packages for NS-3
    python3 -m pip install --user \
        pybindgen \
        pygccxml \
        cppyy \
        numpy \
        scipy \
        matplotlib \
        pandas \
        seaborn \
        networkx \
        pyserial \
        lxml \
        pygraphviz \
        pydot \
        cython \
        jupyter \
        ipython
    
    # SecureRouteX specific packages
    python3 -m pip install --user \
        torch \
        torchvision \
        tensorflow \
        scikit-learn \
        plotly \
        dash \
        flask \
        requests \
        pyyaml \
        toml \
        click \
        colorama \
        tqdm \
        psutil
    
    print_success "Python packages installed"
}

# Download and install NS-3
install_ns3() {
    print_header "DOWNLOADING AND INSTALLING NS-3"
    
    # Create workspace
    NS3_DIR="$HOME/ns3-workspace"
    mkdir -p "$NS3_DIR"
    cd "$NS3_DIR"
    
    # Download NS-3 (latest stable version)
    NS3_VERSION="3.41"
    print_info "Downloading NS-3 version $NS3_VERSION..."
    
    if [[ ! -d "ns-allinone-$NS3_VERSION" ]]; then
        wget "https://www.nsnam.org/releases/ns-allinone-$NS3_VERSION.tar.bz2"
        tar -xjf "ns-allinone-$NS3_VERSION.tar.bz2"
        rm "ns-allinone-$NS3_VERSION.tar.bz2"
    fi
    
    cd "ns-allinone-$NS3_VERSION"
    
    # Build NS-3
    print_info "Building NS-3... This may take 20-30 minutes."
    ./build.py --enable-examples --enable-tests
    
    # Configure NS-3
    cd "ns-$NS3_VERSION"
    ./ns3 configure --enable-examples --enable-tests --enable-python
    
    # Build with optimizations
    ./ns3 build
    
    print_success "NS-3 installed successfully"
    
    # Add to PATH
    echo "export PATH=\$PATH:$NS3_DIR/ns-allinone-$NS3_VERSION/ns-$NS3_VERSION" >> ~/.bashrc
    echo "export NS3_HOME=$NS3_DIR/ns-allinone-$NS3_VERSION/ns-$NS3_VERSION" >> ~/.bashrc
    
    print_info "NS-3 path added to ~/.bashrc"
}

# Install additional simulation tools
install_simulation_tools() {
    print_header "INSTALLING ADDITIONAL SIMULATION TOOLS"
    
    case "$OS" in
        *"Ubuntu"*|*"Debian"*)
            # Install additional network simulation tools
            sudo apt install -y \
                mininet \
                openvswitch-switch \
                openvswitch-testcontroller \
                ryu-manager \
                iperf3 \
                netperf \
                hping3 \
                nmap \
                wireshark-qt
            print_success "Additional simulation tools installed (APT)"
            ;;
        *"CentOS"*|*"Red Hat"*|*"Rocky"*|*"Fedora"*)
            # Note: Some tools may need manual installation on RHEL-based systems
            print_warning "Some simulation tools may need manual installation on RHEL-based systems"
            ;;
    esac
    
    # Install Python simulation packages
    python3 -m pip install --user \
        mininet \
        ryu \
        scapy \
        paramiko \
        fabric \
        ansible
    
    print_success "Python simulation tools installed"
}

# Setup SecureRouteX simulation environment
setup_secureroutex_env() {
    print_header "SETTING UP SECUREROUTEX SIMULATION ENVIRONMENT"
    
    # Create simulation directories
    SIMULATION_DIR="$HOME/secureroutex-simulation"
    mkdir -p "$SIMULATION_DIR"/{scenarios,results,logs,configs}
    
    # Copy SecureRouteX files if they exist
    CURRENT_DIR=$(pwd)
    if [[ -f "$CURRENT_DIR/secureroutex_gan_model.py" ]]; then
        cp "$CURRENT_DIR"/*.py "$SIMULATION_DIR/"
        cp "$CURRENT_DIR"/*.csv "$SIMULATION_DIR/"
        print_success "SecureRouteX files copied to simulation directory"
    fi
    
    # Create sample NS-3 scenario file
    cat > "$SIMULATION_DIR/secureroutex-ns3-scenario.cc" << 'EOF'
/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * SecureRouteX NS-3 Simulation Scenario
 * Multi-domain IoT Network with Trust-aware Routing
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-apps-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("SecureRouteXSimulation");

int main (int argc, char *argv[])
{
  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);
  
  Time::SetResolution (Time::NS);
  LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
  LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);

  // Create healthcare IoT nodes
  NodeContainer healthcareNodes;
  healthcareNodes.Create (10);

  // Create transportation IoT nodes  
  NodeContainer transportNodes;
  transportNodes.Create (15);

  // Create underwater IoT nodes
  NodeContainer underwaterNodes;
  underwaterNodes.Create (8);

  // Create SDN controller node
  NodeContainer controllerNode;
  controllerNode.Create (1);

  // Setup network topology
  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

  // Configure WiFi for mobile nodes
  WifiHelper wifi;
  wifi.SetRemoteStationManager ("ns3::AarfWifiManager");

  // Install Internet stack
  InternetStackHelper stack;
  stack.Install (healthcareNodes);
  stack.Install (transportNodes);
  stack.Install (underwaterNodes);
  stack.Install (controllerNode);

  // Assign IP addresses
  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");

  // Setup applications for testing
  UdpEchoServerHelper echoServer (9);
  ApplicationContainer serverApps = echoServer.Install (controllerNode.Get (0));
  serverApps.Start (Seconds (1.0));
  serverApps.Stop (Seconds (10.0));

  // Run simulation
  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}
EOF
    
    print_success "Sample NS-3 scenario created"
    
    # Set environment variables
    echo "export SECUREROUTEX_SIM_DIR=$SIMULATION_DIR" >> ~/.bashrc
    
    print_info "SecureRouteX simulation environment ready at: $SIMULATION_DIR"
}

# Verify installation
verify_installation() {
    print_header "VERIFYING INSTALLATION"
    
    # Check NS-3
    if command -v ns3 &> /dev/null; then
        print_success "NS-3 command found"
    else
        print_warning "NS-3 command not found in PATH. You may need to source ~/.bashrc"
    fi
    
    # Check Python packages
    python3 -c "import numpy, scipy, matplotlib, pandas, torch" 2>/dev/null && \
        print_success "Essential Python packages available" || \
        print_warning "Some Python packages may be missing"
    
    # Check compilers
    gcc --version &> /dev/null && print_success "GCC compiler available"
    g++ --version &> /dev/null && print_success "G++ compiler available"
    
    # Check build tools
    cmake --version &> /dev/null && print_success "CMake available"
    make --version &> /dev/null && print_success "Make available"
    
    print_info "Installation verification complete"
}

# Main installation function
main() {
    print_header "SECUREROUTEX NS-3 INSTALLATION SCRIPT"
    print_info "This script will install NS-3 and all dependencies for network simulation"
    print_info "Estimated time: 30-45 minutes depending on internet speed"
    
    # Confirm installation
    read -p "Do you want to proceed with the installation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Installation cancelled by user"
        exit 0
    fi
    
    # Run installation steps
    detect_os
    check_sudo
    update_system
    install_basic_deps
    install_ns3_deps
    install_python_packages
    install_ns3
    install_simulation_tools
    setup_secureroutex_env
    verify_installation
    
    print_header "INSTALLATION COMPLETE"
    print_success "NS-3 and SecureRouteX simulation environment installed successfully!"
    print_info "Please run 'source ~/.bashrc' or start a new terminal session"
    print_info "NS-3 workspace: $HOME/ns3-workspace"
    print_info "SecureRouteX simulation: $HOME/secureroutex-simulation"
    
    echo
    print_info "Next steps:"
    echo "1. source ~/.bashrc"
    echo "2. cd \$NS3_HOME"
    echo "3. ./ns3 run hello-simulator"
    echo "4. cd \$SECUREROUTEX_SIM_DIR"
    echo "5. Start developing your SecureRouteX simulations"
}

# Run main function
main "$@"