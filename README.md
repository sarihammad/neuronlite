# NeuronLite

A C++20/23 ML runtime and execution engine focused on fast kernels, explicit memory control, and built-in profiling.

## Highlights

- SIMD-optimized kernels (GEMM, Softmax, activations)
- Arena allocator and buffer pools
- Zero-copy tensor views
- Per-operator profiling with trace export

## Build

Requirements: C++20 compiler, CMake 3.20+, AVX2-capable CPU.

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

Optional flags:

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON \
  -DBUILD_BENCHMARKS=ON \
  -DBUILD_EXAMPLES=ON \
  -DENABLE_PROFILING=ON
```

## Quick Start

```cpp
#include "neuronlite/tensor/tensor.hpp"
#include "neuronlite/kernels/gemm.hpp"

using namespace neuronlite;

Tensor A({128, 512}, DataType::Float32);
Tensor B({512, 256}, DataType::Float32);
Tensor C({128, 256}, DataType::Float32);

A.fill(1.0f);
B.fill(2.0f);

kernels::GEMM::matmul(A, B, C);
```

## Examples, Benchmarks, Tests

```bash
./build/examples/neuronlite_examples
./benchmarks/neuronlite_benchmarks
ctest --output-on-failure
```

## Layout

```
include/neuronlite/   Public headers
src/                  Implementations
examples/             Usage examples
benchmarks/           Microbenchmarks
```

## Contributing

See CONTRIBUTING.md.

## License

MIT. See LICENSE.
