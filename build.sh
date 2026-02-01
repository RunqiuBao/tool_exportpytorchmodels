# 1. Create and enter build directory
mkdir build || cd build

# 2. Configure with the desired install path (e.g., a 'dist' folder in your project)
cmake .. -DCMAKE_INSTALL_PREFIX=../dist

# 3. Build the libraries
make -j$(nproc)

# 4. Install them to the ../dist folder
make install