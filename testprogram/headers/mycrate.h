#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

struct GPUResources;

template<typename T = void>
struct Option;

struct GPUHandle {
  GPUResources *gpu_resources;
  uintptr_t width;
  uintptr_t height;
  Option<uint16_t*> dark_map_data;
  Option<uint16_t*> gain_map_data;
  Option<uint16_t*> defect_map_data;
};

extern "C" {

GPUHandle *create_gpu_handle(uintptr_t width, uintptr_t height);

void set_dark_map(GPUHandle *gpu_handle, uint16_t *dark_map_data);

void set_gain_map(GPUHandle *gpu_handle, uint16_t *gain_map_data);

void set_defect_map(GPUHandle *gpu_handle, uint16_t *defect_map_data);

void run_corrections(GPUHandle *gpu_handle, uint16_t *data);

void free_gpu_handle(GPUHandle *handle);

} // extern "C"
