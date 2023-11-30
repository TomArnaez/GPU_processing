use std::{borrow::Cow, time::Instant};

use pollster::block_on;
use wgpu::{util::DeviceExt, BindGroupLayout, ComputePipeline, Instance};

use super::{error::MyError, offset_correction::OffsetCorrectionResources, gain_correction_resources::GainCorrectionResources};

pub struct GPUResources {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

pub fn initialise_gpu_resources() -> Result<GPUResources, &'static str> {
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

struct CorrectionContext {
    gpu_resources: GPUResources,
    width: u32,
    height: u32,
    offset_resources: Option<OffsetCorrectionResources>,
    gain_resources: Option<GainCorrectionResources>,
}

impl CorrectionContext {
    pub fn new(gpu_resources: GPUResources, width: u32, height: u32) -> Self {
        CorrectionContext { gpu_resources, width, height, offset_resources: None, gain_resources: None }
    }

    pub fn enable_offset_pipeline(&mut self, dark_map: Vec<u16>) {
        self.offset_resources = Some(OffsetCorrectionResources::new(&self.gpu_resources.device, &self.gpu_resources.queue, &dark_map, self.width, self.height));
    }

    pub fn enable_gain_pipeline(&mut self, gain_map: Vec<u16>) {
        self.gain_resources = Some(GainCorrectionResources::new(&self.gpu_resources.device, &self.gpu_resources.queue, &gain_map, self.width, self.height));
    }

    pub async fn process_image(&mut self, input_image: &Vec<u16>) -> Result<Vec<u16>, MyError> {
        let input_texture = block_on(create_image_texture(&self.gpu_resources.device, &self.gpu_resources.queue, &input_image, "input_image", self.width, self.height))?;

        let output_texture = self.gpu_resources.device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            label: Some("output_texture"),
            view_formats: &[wgpu::TextureFormat::R32Uint],
        });

        let buffer_size = self.width * self.height * 4;

        let readback_buffer = &self.gpu_resources.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let workgroup_size_x = 8;
        let workgroup_size_y = 8;
        let dispatch_size_x = (self.width + workgroup_size_x - 1) / workgroup_size_x;
        let dispatch_size_y = (self.height + workgroup_size_y - 1) / workgroup_size_y;

        let mut encoder = self
        .gpu_resources.device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("image_correction_encoder"),
        });

        {
            let bind_group = self.offset_resources.as_ref().unwrap().get_bind_group(&self.gpu_resources.device, &input_texture, &output_texture);
            let pipeline = &self.offset_resources.as_ref().unwrap().pipeline;

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("image_correction_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(dispatch_size_x, dispatch_size_y, 1);
        }

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: readback_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.width * 4),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );


        self.gpu_resources.queue.submit(Some(encoder.finish()));

        let buffer_slice = readback_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.gpu_resources.device.poll(wgpu::Maintain::Wait);
        if let Ok(Ok(())) = receiver.recv_async().await {
            let data = buffer_slice.get_mapped_range();
            let result = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            readback_buffer.unmap();

            // Returns data from buffer
            return Ok(result);
        } else {
            panic!("failed to run compute on gpu!")
        }
    }
}

pub async fn create_image_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &Vec<u16>,
    label: &str,
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
        label: Some(label),
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

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use pollster::block_on;
    use wgpu::Instance;

    use super::{initialise_gpu_resources, create_image_texture, CorrectionContext};

    #[test]
    fn test_initialize_gpu() {
        let gpu_resources = initialise_gpu_resources().unwrap();

        let mut correction_context = CorrectionContext::new(gpu_resources, 512, 512);

        let dark_data = vec![7000; 1650*1036];

        correction_context.enable_offset_pipeline(dark_data);

        let image_data = vec![8000; 2802*2400];
        let start = Instant::now();
        for i in 0..1000 {
            let data = block_on(correction_context.process_image(&image_data)).unwrap();
        }
        let duration = start.elapsed();
        println!("Average time elapsed for dark correction is: {:?}", duration / 1000);
    }
}
