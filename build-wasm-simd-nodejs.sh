#!/usr/bin/env bash
# Copyright (c)  2024  Xiaomi Corporation
#
# This script is to build sherpa-onnx for WebAssembly (NodeJS)
#
# Please use NodeJS >= 18

set -ex

if [ x"$EMSCRIPTEN" == x"" ]; then
  if ! command -v emcc &> /dev/null; then
    echo "Please install emscripten first"
    echo ""
    echo "You can use the following commands to install it:"
    echo ""
    echo "git clone https://github.com/emscripten-core/emsdk.git"
    echo "cd emsdk"
    echo "git pull"
    echo "./emsdk install 3.1.53"
    echo "./emsdk activate 3.1.53"
    echo "source ./emsdk_env.sh"
    exit 1
  else
    EMSCRIPTEN=$(dirname $(realpath $(which emcc)))
    emcc --version
  fi
fi

export EMSCRIPTEN=$EMSCRIPTEN
echo "EMSCRIPTEN: $EMSCRIPTEN"
if [ ! -f $EMSCRIPTEN/cmake/Modules/Platform/Emscripten.cmake ]; then
  echo "Cannot find $EMSCRIPTEN/cmake/Modules/Platform/Emscripten.cmake"
  echo "Please make sure you have installed emsdk correctly"
  echo "Hint: emsdk 3.1.53 is known to work. Other versions may not work"
  exit 1
fi

mkdir -p build-wasm-simd-nodejs
pushd build-wasm-simd-nodejs

export SHERPA_ONNX_IS_USING_BUILD_WASM_SH=ON

cmake \
  -DCMAKE_INSTALL_PREFIX=./install \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$EMSCRIPTEN/cmake/Modules/Platform/Emscripten.cmake \
  \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
  -DSHERPA_ONNX_ENABLE_TESTS=OFF \
  -DSHERPA_ONNX_ENABLE_CHECK=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_JNI=OFF \
  -DSHERPA_ONNX_ENABLE_C_API=ON \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
  -DSHERPA_ONNX_ENABLE_GPU=OFF \
  -DSHERPA_ONNX_ENABLE_WASM=ON \
  -DSHERPA_ONNX_ENABLE_WASM_NODEJS=ON \
  -DSHERPA_ONNX_ENABLE_BINARY=OFF \
  -DSHERPA_ONNX_LINK_LIBSTDCPP_STATICALLY=OFF \
  ..
make -j3
make install

ls -lh install/bin/wasm/nodejs
