#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

struct Corrections;

struct GPUHandle {
  Corrections *correction_context;
};

extern "C" {

GPUHandle *create_gpu_handle(uint32_t width, uint32_t height, uint32_t buffer_count);

void set_dark_map(GPUHandle *gpu_handle, uint16_t *dark_map_data, uint32_t width, uint32_t height);

void set_gain_map(GPUHandle *gpu_handle, float *gain_map_data, uint32_t width, uint32_t height);

void set_defect_map(GPUHandle *gpu_handle,
                    uint16_t *defect_map_data,
                    uint32_t width,
                    uint32_t height);

void process_image(GPUHandle *gpu_handle, uint16_t *data, uint32_t width, uint32_t height);

void free_gpu_handle(GPUHandle *handle);

} // extern "C"
