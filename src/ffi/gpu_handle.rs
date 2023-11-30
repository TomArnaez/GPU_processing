use std::ptr::NonNull;

use crate::core::core::{GPUResources, initialise_gpu_resources};

#[repr(C)]
pub struct GPUHandle {
    gpu_resources: NonNull<GPUResources>,
    width: usize,
    height: usize,
    dark_map_data: Option<*mut u16>,
    gain_map_data: Option<*mut u16>,
    defect_map_data: Option<*mut u16>,
}

#[no_mangle]
pub extern "C" fn create_gpu_handle(width: usize, height: usize) -> *mut GPUHandle {
    // Allocate GPUResources and check for errors
    let gpu_resources = match initialise_gpu_resources() {
        Ok(resources) => Box::new(resources),
        Err(_) => return std::ptr::null_mut(),
    };

    // Create a GPUHandle with a pointer to GPUResources
    let handle = Box::new(GPUHandle {
        gpu_resources: NonNull::new(Box::into_raw(gpu_resources)).unwrap(),
        width,
        height,
        defect_map_data: None,
        gain_map_data: None,
        dark_map_data: None,
    });

    Box::into_raw(handle)
}

#[no_mangle]
pub extern "C" fn set_dark_map(gpu_handle: *mut GPUHandle, dark_map_data: *mut u16) {
    if gpu_handle.is_null() || dark_map_data.is_null() {
        return;
    }

    let gpu_handle = unsafe { &mut *gpu_handle };
    gpu_handle.dark_map_data = if dark_map_data.is_null() { None } else { Some(dark_map_data) };
}

#[no_mangle]
pub extern "C" fn set_gain_map(gpu_handle: *mut GPUHandle, gain_map_data: *mut u16) {
    if gpu_handle.is_null() || gain_map_data.is_null() {
        return;
    }

    let gpu_handle = unsafe { &mut *gpu_handle };
    gpu_handle.gain_map_data = if gain_map_data.is_null() { None } else { Some(gain_map_data) };
}

#[no_mangle]
pub extern "C" fn set_defect_map(gpu_handle: *mut GPUHandle, defect_map_data: *mut u16) {
    if gpu_handle.is_null() || defect_map_data.is_null() {
        return;
    }

    let gpu_handle = unsafe { &mut *gpu_handle };
    gpu_handle.defect_map_data = if defect_map_data.is_null() { None } else { Some(defect_map_data) };
}

#[no_mangle]
pub extern "C" fn run_corrections(gpu_handle: *mut GPUHandle, data: *mut u16) {
    if gpu_handle.is_null() {
        return;
    }
}

#[no_mangle]
pub extern "C" fn free_gpu_handle(handle: *mut GPUHandle) {
    if !handle.is_null() {
        // Convert the raw pointer back to a Box to ensure proper deallocation
        let _handle = unsafe { Box::from_raw(handle) };
        // GPUResources will be dropped here
    }
}