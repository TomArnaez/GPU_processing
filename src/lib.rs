pub mod core;
pub mod ffi;

use std::borrow::Cow;

use futures::executor::block_on;
use wgpu::{util::DeviceExt, Device, Instance, Queue};

struct GPUResources {
    device: Device,
    queue: Queue,
}

fn initialize_gpu_resources() -> Result<GPUResources, &'static str> {
    let instance = Instance::default();

    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .ok_or("Failed to find an appropriate adapter")?;

    let (device, queue) = block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::default(),
            label: None,
        },
        None,
    ))
    .map_err(|_| "Failed to create device")?;

    Ok(GPUResources { device, queue })
}

#[cfg(test)]
mod tests {
    use crate::initialize_gpu_resources;

    #[test]
    fn test_initialize_gpu_resources() {
        let result = initialize_gpu_resources();
        assert!(result.is_ok());
    }
}
