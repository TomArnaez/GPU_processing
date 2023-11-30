use std::borrow::Cow;

use wgpu::{util::DeviceExt, BindGroupLayout, ComputePipeline};

use super::error::MyError;

fn create_offset_correction_pipeline(device: &wgpu::Device) -> (ComputePipeline, BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
            "shaders/offset_correction.wgsl"
        ))),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            // Dark map texture
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // Input image texture
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // Output image texture
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::R32Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
        label: Some("Bind Group Layout"),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "correct_image", // Name of the compute function in your WGSL
    });

    (pipeline, bind_group_layout)
}

async fn create_image_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[u16],
    width: u32,
    height: u32,
) -> Result<wgpu::Texture, MyError> {
    if width == 0 || height == 0 || data.is_empty() {
        return Err(MyError::InvalidTextureData);
    }

    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R16Uint,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        label: Some("dark_map_texture"),
        view_formats: &[wgpu::TextureFormat::R16Uint],
    });

    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Temp Buffer"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::COPY_SRC,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("texture_encoder"),
    });

    encoder.copy_buffer_to_texture(
        wgpu::ImageCopyBuffer {
            buffer: &buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * 2),
                rows_per_image: Some(height),
            },
        },
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        size,
    );

    queue.submit(Some(encoder.finish()));
    Ok(texture)
}

struct OffsetCorrectionContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    dark_map_texture: wgpu::Texture,
    bind_group_layout: wgpu::BindGroupLayout,
    width: u32,
    height: u32,
}

impl OffsetCorrectionContext {
    async fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        dark_map_data: &[u16],
        width: u32,
        height: u32,
    ) -> Result<Self, MyError> {
        let (pipeline, bind_group_layout) = create_offset_correction_pipeline(&device);
        let dark_map_texture =
            create_image_texture(&device, &queue, dark_map_data, width, height).await?;

        Ok(OffsetCorrectionContext {
            device,
            queue,
            pipeline,
            dark_map_texture,
            bind_group_layout,
            width,
            height,
        })
    }

    async fn process_image(
        &self,
        data: &[u16],
        width: u32,
        height: u32,
    ) -> Result<Vec<u32>, MyError> {
        if width == 0 || height == 0 || data.is_empty() {
            return Err(MyError::InvalidTextureData);
        }

        let workgroup_size_x = 8;
        let workgroup_size_y = 8;
        let dispatch_size_x = (width + workgroup_size_x - 1) / workgroup_size_x;
        let dispatch_size_y = (height + workgroup_size_y - 1) / workgroup_size_y;

        let input_texture =
            create_image_texture(&self.device, &self.queue, data, width, height).await?;

        let output_texture = &self.device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint, // Choose the appropriate format for your use case
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            label: Some("output_texture"),
            view_formats: &[wgpu::TextureFormat::R32Uint],
        });

        let dark_map_texture_view = self
            .dark_map_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let input_texture_view = input_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let output_texture_view =
            output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&dark_map_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&input_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&output_texture_view),
                },
            ],
            label: Some("image_correction_bind_group"),
        });

        let buffer_size = width * height * 4; // Calculate the appropriate size
        let readback_buffer = &self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Dispatch the compute pipeline
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("image_processing_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("image_processing_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(dispatch_size_x, dispatch_size_y, 1);
        }

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: readback_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(Some(encoder.finish()));

        // Map the buffer for reading
        let buffer_slice = readback_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        if let Ok(Ok(())) = receiver.recv_async().await {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            let result = bytemuck::cast_slice(&data).to_vec();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            readback_buffer.unmap(); // Unmaps buffer from memory
                                     // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                     //   delete myPointer;
                                     //   myPointer = NULL;
                                     // It effectively frees the memory

            // Returns data from buffer
            return Ok(result);
        } else {
            panic!("failed to run compute on gpu!")
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{core::core::create_image_texture, initialize_gpu_resources};

    use super::{create_offset_correction_pipeline, OffsetCorrectionContext};

    #[test]
    fn test_offset_correct() {
        let resources = initialize_gpu_resources().unwrap();
        let (pipeline, bind_group_layout) = create_offset_correction_pipeline(&resources.device);
    }

    #[test]
    fn test_create_image() {
        let resources = initialize_gpu_resources().unwrap();
        const WIDTH: usize = 256;
        const HEIGHT: usize = 256;
        let data = &[100; WIDTH * HEIGHT];
        let image_texture = pollster::block_on(create_image_texture(
            &resources.device,
            &resources.queue,
            data,
            WIDTH as u32,
            HEIGHT as u32,
        ))
        .unwrap();
    }

    #[test]
    fn test_process_image() {
        let resources = initialize_gpu_resources().unwrap();
        const WIDTH: usize = 256;
        const HEIGHT: usize = 256;
        let data = &[100; WIDTH * HEIGHT];
        let context = pollster::block_on(OffsetCorrectionContext::new(
            resources.device,
            resources.queue,
            data,
            WIDTH as u32,
            HEIGHT as u32,
        ))
        .unwrap();

        let image_data = &[100; WIDTH * HEIGHT];
        for i in 0..1000 {
            println!("{i}");
            pollster::block_on(context.process_image(image_data, WIDTH as u32, HEIGHT as u32));
        }
    }
}
