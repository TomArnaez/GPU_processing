use std::ptr::NonNull;

use crate::{initialize_gpu_resources, GPUResources};

#[repr(C)]
pub struct GPUHandle {
    gpu_resources: NonNull<GPUResources>,
}

#[no_mangle]
pub extern "C" fn create_gpu_handle() -> *mut GPUHandle {
    // Allocate GPUResources and check for errors
    let gpu_resources = match initialize_gpu_resources() {
        Ok(resources) => Box::new(resources),
        Err(_) => return std::ptr::null_mut(),
    };

    // Create a GPUHandle with a pointer to GPUResources
    let handle = Box::new(GPUHandle {
        gpu_resources: NonNull::new(Box::into_raw(gpu_resources)).unwrap(),
    });

    Box::into_raw(handle)
}

#[no_mangle]
pub extern "C" fn free_gpu_handle(handle: *mut GPUHandle) {
    if !handle.is_null() {
        // Convert the raw pointer back to a Box to ensure proper deallocation
        let _handle = unsafe { Box::from_raw(handle) };
        // GPUResources will be dropped here
    }
}

#[no_mangle]
pub extern "C" fn test_fn() {
    println!("Hello world from Rust!")
}
