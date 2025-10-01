# SecureRouteX NS-3 Installation Guide

## Quick Installation (One Command)

```bash
# Download and run the installation script
wget https://raw.githubusercontent.com/your-repo/secureroutex/main/install_ns3_dependencies.sh
chmod +x install_ns3_dependencies.sh
./install_ns3_dependencies.sh
```

## Step-by-Step Manual Installation

### 1. **System Requirements**

**Minimum Requirements:**
- Ubuntu 20.04+ / Debian 11+ / CentOS 8+ / Fedora 35+
- 4GB RAM (8GB recommended)
- 10GB free disk space
- Internet connection for downloads

**Recommended:**
- 8GB+ RAM for large simulations
- SSD storage for faster builds
- Multi-core processor

### 2. **Pre-Installation Steps**

```bash
# Update your system
sudo apt update && sudo apt upgrade -y  # Ubuntu/Debian
# OR
sudo yum update -y                       # CentOS/RHEL
# OR  
sudo dnf update -y                       # Fedora

# Install git if not already installed
sudo apt install git -y                 # Ubuntu/Debian
sudo yum install git -y                 # CentOS/RHEL
sudo dnf install git -y                 # Fedora
```

### 3. **Install Build Dependencies**

#### For Ubuntu/Debian:
```bash
sudo apt install -y \
    build-essential \
    gcc g++ \
    python3 python3-pip python3-dev \
    cmake make \
    libssl-dev libffi-dev \
    git wget curl unzip
```

#### For CentOS/RHEL:
```bash
sudo yum groupinstall -y "Development Tools"
sudo yum install -y \
    gcc gcc-c++ \
    python3 python3-pip python3-devel \
    cmake make \
    openssl-devel libffi-devel \
    git wget curl unzip
```

#### For Fedora:
```bash
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y \
    gcc gcc-c++ \
    python3 python3-pip python3-devel \
    cmake make \
    openssl-devel libffi-devel \
    git wget curl unzip
```

### 4. **Install NS-3 Specific Dependencies**

#### For Ubuntu/Debian:
```bash
sudo apt install -y \
    gir1.2-goocanvas-2.0 \
    gir1.2-gtk-3.0 \
    libboost-all-dev \
    libgtk-3-dev \
    libxml2 libxml2-dev \
    qtbase5-dev \
    openmpi-bin openmpi-common libopenmpi-dev \
    gsl-bin libgsl-dev \
    sqlite3 libsqlite3-dev \
    libpcap-dev \
    doxygen graphviz \
    wireshark tcpdump
```

#### For CentOS/RHEL:
```bash
sudo yum install -y epel-release
sudo yum install -y \
    boost-devel \
    gtk3-devel \
    libxml2 libxml2-devel \
    qt5-qtbase-devel \
    openmpi openmpi-devel \
    gsl-devel \
    sqlite sqlite-devel \
    libpcap-devel \
    doxygen graphviz \
    wireshark tcpdump
```

### 5. **Install Python Packages**

```bash
# Upgrade pip
python3 -m pip install --upgrade pip

# Essential NS-3 Python packages
python3 -m pip install --user \
    pybindgen \
    pygccxml \
    cppyy \
    numpy scipy matplotlib \
    pandas seaborn \
    networkx \
    lxml \
    cython

# SecureRouteX specific packages
python3 -m pip install --user \
    torch torchvision \
    tensorflow \
    scikit-learn \
    plotly dash \
    pyyaml requests \
    tqdm psutil
```

### 6. **Download and Build NS-3**

```bash
# Create workspace
mkdir -p ~/ns3-workspace
cd ~/ns3-workspace

# Download NS-3 (version 3.41 - latest stable)
wget https://www.nsnam.org/releases/ns-allinone-3.41.tar.bz2
tar -xjf ns-allinone-3.41.tar.bz2
cd ns-allinone-3.41

# Build NS-3 (this takes 20-30 minutes)
./build.py --enable-examples --enable-tests

# Configure NS-3
cd ns-3.41
./ns3 configure --enable-examples --enable-tests --enable-python

# Build with optimizations
./ns3 build
```

### 7. **Install Additional Simulation Tools**

#### For Ubuntu/Debian:
```bash
sudo apt install -y \
    mininet \
    openvswitch-switch \
    openvswitch-testcontroller \
    ryu-manager \
    iperf3 netperf \
    hping3 nmap
```

#### Python simulation tools:
```bash
python3 -m pip install --user \
    mininet \
    ryu \
    scapy \
    paramiko
```

### 8. **Setup Environment Variables**

```bash
# Add to ~/.bashrc
echo 'export PATH=$PATH:~/ns3-workspace/ns-allinone-3.41/ns-3.41' >> ~/.bashrc
echo 'export NS3_HOME=~/ns3-workspace/ns-allinone-3.41/ns-3.41' >> ~/.bashrc

# Reload environment
source ~/.bashrc
```

### 9. **Verify Installation**

```bash
# Test NS-3 installation
cd $NS3_HOME
./ns3 run hello-simulator

# Check Python packages
python3 -c "import numpy, scipy, matplotlib, torch; print('All packages OK')"

# Check tools
gcc --version
python3 --version
cmake --version
```

### 10. **Setup SecureRouteX Simulation Environment**

```bash
# Create simulation directories
mkdir -p ~/secureroutex-simulation/{scenarios,results,logs,configs}

# Set environment variable
echo 'export SECUREROUTEX_SIM_DIR=~/secureroutex-simulation' >> ~/.bashrc
source ~/.bashrc

# Copy your SecureRouteX files
cp /path/to/your/secureroutex/*.py ~/secureroutex-simulation/
```

## **Troubleshooting Common Issues**

### Issue 1: Build fails with missing dependencies
```bash
# Install missing packages based on error messages
sudo apt install <missing-package>  # Ubuntu/Debian
sudo yum install <missing-package>  # CentOS/RHEL
```

### Issue 2: Python binding errors
```bash
# Reinstall Python packages
python3 -m pip install --user --upgrade pybindgen pygccxml
```

### Issue 3: Permission errors
```bash
# Ensure proper permissions
chmod +x install_ns3_dependencies.sh
# Don't run as root - use regular user with sudo
```

### Issue 4: NS-3 not found in PATH
```bash
# Manually add to PATH
export PATH=$PATH:~/ns3-workspace/ns-allinone-3.41/ns-3.41
# Or restart terminal after adding to ~/.bashrc
```

## **Testing Your Installation**

### Basic NS-3 Test:
```bash
cd $NS3_HOME
./ns3 run hello-simulator
./ns3 run scratch-simulator
```

### Python Integration Test:
```bash
cd $NS3_HOME
python3 -c "import ns; print('NS-3 Python bindings OK')"
```

### SecureRouteX Integration Test:
```bash
cd $SECUREROUTEX_SIM_DIR
python3 -c "
import numpy as np
import torch
import matplotlib.pyplot as plt
print('SecureRouteX dependencies OK')
"
```

## **Performance Optimization Tips**

1. **Parallel Builds:**
```bash
# Use multiple cores for faster compilation
./ns3 build -j$(nproc)
```

2. **Optimized Builds:**
```bash
# Release mode for better performance
./ns3 configure --build-profile=release
./ns3 build
```

3. **Memory Settings:**
```bash
# For large simulations, increase memory limits
export NS_GLOBAL_VALUE="RngSeed=1"
```

## **Next Steps After Installation**

1. **Learn NS-3 Basics:**
   - Run tutorial examples
   - Read NS-3 documentation
   - Practice with simple topologies

2. **Integrate SecureRouteX:**
   - Adapt your GAN models for NS-3
   - Implement trust-aware routing
   - Create multi-domain scenarios

3. **Validation and Testing:**
   - Run performance benchmarks
   - Validate against literature
   - Generate publication graphs

## **Estimated Installation Time**

- **Basic dependencies:** 10-15 minutes
- **NS-3 download and build:** 20-30 minutes  
- **Additional tools:** 5-10 minutes
- **Configuration and testing:** 5-10 minutes

**Total: 40-65 minutes** (depending on internet speed and system performance)

## **Support and Resources**

- **NS-3 Documentation:** https://www.nsnam.org/documentation/
- **NS-3 Wiki:** https://www.nsnam.org/wiki/
- **NS-3 Mailing List:** https://groups.google.com/forum/#!forum/ns-3-users
- **SecureRouteX Issues:** Check your project repository

---

**Ready to simulate! 🚀**