use std::{ptr::NonNull, time::Instant};

use crate::core::core::{GPUResources, initialise_gpu_resources, CorrectionContext};

use super::power_preference::{self, CPowerPreference};

#[repr(C)]
pub struct GPUHandle {
    correction_context: NonNull<CorrectionContext>,
}

#[no_mangle]
pub extern "C" fn create_gpu_handle(width: u32, height: u32, power_preference: CPowerPreference) -> *mut GPUHandle {
    // Allocate GPUResources and check for errors
    let gpu_resources = match initialise_gpu_resources(power_preference.into()) {
        Ok(resources) => resources,
        Err(_) => return std::ptr::null_mut(),
    };

    let correction_context = Box::new(CorrectionContext::new(gpu_resources, width, height));

    let handle = Box::new(GPUHandle {
        correction_context: NonNull::new(Box::into_raw(correction_context)).unwrap()
    });

    Box::into_raw(handle)
}

#[no_mangle]
pub extern "C" fn set_dark_map(gpu_handle: *mut GPUHandle, dark_map_data: *mut u16, width: u32, height: u32) {
    if gpu_handle.is_null() || dark_map_data.is_null() {
        return;
    }

    let gpu_handle = unsafe { &mut *gpu_handle };
    let size = (width * height) as usize;
    let slice =  unsafe { std::slice::from_raw_parts(dark_map_data, size) };
    unsafe {gpu_handle.correction_context.as_mut().enable_offset_pipeline(slice, 300) };
}

#[no_mangle]
pub extern "C" fn set_gain_map(gpu_handle: *mut GPUHandle, gain_map_data: *mut f32, width: u32, height: u32) {
    if gpu_handle.is_null() || gain_map_data.is_null() {
        return;
    }

    let gpu_handle: &mut GPUHandle = unsafe { &mut *gpu_handle };
    let size = (width * height) as usize;
    let slice =  unsafe { std::slice::from_raw_parts(gain_map_data, size) };
    unsafe {gpu_handle.correction_context.as_mut().enable_gain_pipeline(slice) };}

#[no_mangle]
pub extern "C" fn process_image(gpu_handle: *mut GPUHandle, data: *mut u16, width: u32, height: u32) {
    if gpu_handle.is_null() {
        return;
    }

    let size = (width * height) as usize;
    let gpu_handle: &mut GPUHandle = unsafe { &mut *gpu_handle };
    let slice =  unsafe { std::slice::from_raw_parts(data, size) };
    let result = unsafe { pollster::block_on(gpu_handle.correction_context.as_mut().process_image(&slice)) };
}

#[no_mangle]
pub extern "C" fn free_gpu_handle(handle: *mut GPUHandle) {
    if !handle.is_null() {
        // Convert the raw pointer back to a Box to ensure proper deallocation
        let _handle = unsafe { Box::from_raw(handle) };
        // GPUResources will be dropped here
    }
}