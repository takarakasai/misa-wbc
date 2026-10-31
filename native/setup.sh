#!/usr/bin/env bash
set -euo pipefail
root="${MISA_QP_NATIVE_ROOT:-/tmp/articara-native-qp}"
mkdir -p "$root"
fetch() {
  if [ ! -d "$root/$1/.git" ]; then git clone "$2" "$root/$1"; fi
  git -C "$root/$1" checkout --detach "$3"
}
fetch proxsuite https://github.com/Simple-Robotics/proxsuite.git e701292db6b6f0941c2ddf68be3772d0fae30754
fetch eiquadprog https://github.com/stack-of-tasks/eiquadprog.git ec402b4dbcce32fd936fd39a3c6fc32f08b35a54
fetch qpoases https://github.com/coin-or/qpOASES.git 51d3fbea30142d3acbf40cf7a1c519efc27ea67b
fetch osqp https://github.com/osqp/osqp.git 0dd00a578cf1c2691c5c379965d504c75bf6cfad
fetch hpipm https://github.com/giaf/hpipm.git bad1123a96136fce3cdd8bbcbe24b853d012b840
fetch blasfeo https://github.com/giaf/blasfeo.git 60493284741b3e4e5b0bf25155221d4b5f0b2232
git -C "$root/osqp" submodule update --init --recursive
mkdir -p "$root/generated/proxsuite" "$root/generated/eiquadprog"
cat > "$root/generated/proxsuite/config.hpp" <<'CONFIG'
#pragma once
#define PROXSUITE_MAJOR_VERSION 0
#define PROXSUITE_MINOR_VERSION 7
#define PROXSUITE_PATCH_VERSION 2
#define PROXSUITE_VERSION_AT_LEAST(a,b,c) ((a)<0 || ((a)==0 && ((b)<7 || ((b)==7 && (c)<=2))))
CONFIG
cat > "$root/generated/eiquadprog/config.hpp" <<'CONFIG'
#pragma once
#define EIQUADPROG_DLLAPI
CONFIG
cmake -S "$root/osqp" -B "$root/osqp/build" -DCMAKE_BUILD_TYPE=Release -DUNITTESTS=OFF -DDLONG=OFF -DPRINTING=OFF -DPROFILING=ON
cmake --build "$root/osqp/build" --parallel 2
cmake -S "$root/blasfeo" -B "$root/blasfeo/build" -DCMAKE_BUILD_TYPE=Release -DTARGET=GENERIC -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX="$root/install"
cmake --build "$root/blasfeo/build" --parallel 2
cmake --install "$root/blasfeo/build"
cmake -S "$root/hpipm" -B "$root/hpipm/build" -DCMAKE_BUILD_TYPE=Release -DTARGET=GENERIC -DBUILD_SHARED_LIBS=OFF -DBLASFEO_PATH="$root/install" -DCMAKE_INSTALL_PREFIX="$root/install"
cmake --build "$root/hpipm/build" --parallel 2
cmake --install "$root/hpipm/build"
