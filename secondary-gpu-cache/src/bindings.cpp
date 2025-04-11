#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "secondary_gpu_cache.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<SecondaryGPUCache>(m, "handle")
        .def(py::init<size_t, size_t, size_t, torch::Tensor, size_t, size_t, float, int>())
        .def("swap_out", &SecondaryGPUCache::swap_out)
        .def("swap_in", &SecondaryGPUCache::swap_in)
        .def("grow_cache", &SecondaryGPUCache::grow_cache)
        .def("shrink_cache", &SecondaryGPUCache::shrink_cache);
}
