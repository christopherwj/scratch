@echo off
echo Quick build test...
if not exist build mkdir build
cd build
cmake .. -G "MinGW Makefiles"
cmake --build .
cd ..
echo Build complete! 