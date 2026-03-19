#include <torch/extension.h>

bool ptx_backend_scaffold_ready() {
    return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ptx_backend_scaffold_ready", &ptx_backend_scaffold_ready, "PTX scaffold availability");
}
