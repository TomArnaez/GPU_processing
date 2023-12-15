use std::{ptr::NonNull, time::Instant};

use crate::core::core::{initialise_gpu_resources, Corrections};

#[repr(C)]
pub struct GPUHandle {
    correction_context: NonNull<Corrections>,
}

#[no_mangle]
pub extern "C" fn create_gpu_handle(
    width: u32,
    height: u32,
) -> *mut GPUHandle {
    // Allocate GPUResources and check for errors
    let gpu_resources = initialise_gpu_resources();

    let correction_context = Box::new(Corrections::new(
        gpu_resources.1.clone(),
        gpu_resources.0.clone(),
        width,
        height,
    ));

    let handle = Box::new(GPUHandle {
        correction_context: NonNull::new(Box::into_raw(correction_context)).unwrap(),
    });

    Box::into_raw(handle)
}

#[no_mangle]
pub extern "C" fn set_dark_map(
    gpu_handle: *mut GPUHandle,
    dark_map_data: *mut u16,
    width: u32,
    height: u32,
) {
    if gpu_handle.is_null() || dark_map_data.is_null() {
        return;
    }

    let gpu_handle = unsafe { &mut *gpu_handle };
    let size: usize = (width * height) as usize;
    let dark_map: Vec<u16> = unsafe {
        Vec::from_raw_parts(
            dark_map_data,
            size,
            size,
        )
    };
    unsafe {
        gpu_handle
            .correction_context
            .as_mut()
            .enable_dark_map_correction(dark_map, 300);
    };
}

#[no_mangle]
pub extern "C" fn set_gain_map(
    gpu_handle: *mut GPUHandle,
    gain_map_data: *mut f32,
    width: u32,
    height: u32,
) {
    if gpu_handle.is_null() || gain_map_data.is_null() {
        return;
    }

    let gpu_handle: &mut GPUHandle = unsafe { &mut *gpu_handle };
    let size = (width * height) as usize;
    let gain_map: Vec<f32> = unsafe {
        Vec::from_raw_parts(
            gain_map_data,
            (width * height) as usize,
            (width * height) as usize,
        )
    };
    unsafe {
        gpu_handle
            .correction_context
            .as_mut()
            .enable_gain_correction(gain_map);
    };
}

#[no_mangle]
pub extern "C" fn set_defect_map(
    gpu_handle: *mut GPUHandle,
    defect_map_data: *mut u16,
    width: u32,
    height: u32,
) {
    if gpu_handle.is_null() || defect_map_data.is_null() {
        return;
    }

    let gpu_handle = unsafe { &mut *gpu_handle };
    let size: usize = (width * height) as usize;
    let defect_map: Vec<u16> = unsafe {
        Vec::from_raw_parts(
            defect_map_data,
            (width * height) as usize,
            (width * height) as usize,
        )
    };
    unsafe {
        gpu_handle
            .correction_context
            .as_mut()
            .enable_defect_correction(defect_map);
    };
}

#[no_mangle]
pub extern "C" fn process_image(
    gpu_handle: *mut GPUHandle,
    data: *mut u16,
    width: u32,
    height: u32,
) {
    if gpu_handle.is_null() {
        return;
    }

    let size = (width * height) as usize;
    let gpu_handle: &mut GPUHandle = unsafe { &mut *gpu_handle };
    let image: Vec<u16> =
        unsafe { Vec::from_raw_parts(data, (width * height) as usize, (width * height) as usize) };
    let result = unsafe { gpu_handle.correction_context.as_mut().process_image(image) };
}

#[no_mangle]
pub extern "C" fn free_gpu_handle(handle: *mut GPUHandle) {
    if !handle.is_null() {
        // Convert the raw pointer back to a Box to ensure proper deallocation
        let _handle = unsafe { Box::from_raw(handle) };
        // GPUResources will be dropped here
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::{GPUHandle, create_gpu_handle, set_dark_map};


    #[test]
    fn test() {
        let image_width: u32 = 4800;
        let image_height: u32 = 5800;
        let offset = 300;
        let mut data = vec![1u16; (image_height * image_width) as usize];

        let handle = create_gpu_handle(image_width, image_height);
        //set_dark_map(handle, data.as_mut_ptr(), image_width, image_height);
    }
}
