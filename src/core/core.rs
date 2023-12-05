use std::time::Instant;

use pollster::block_on;
use wgpu::{Buffer, Instance};

use super::{
    corrections::{
        gain_correction_resources::GainCorrectionResources,
        offset_correction::OffsetCorrectionResources,
    },
    error::MyError,
};

pub struct GPUResources {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

pub fn initialise_gpu_resources() -> Result<GPUResources, &'static str> {
    let instance = Instance::default();

    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .ok_or("Failed to find an appropriate adapter")?;

    let (device, queue) = block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
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
    readback_buffer: Buffer,
    staging_buffer: Buffer,
    width: u32,
    height: u32,
    dispatch_size_x: u32,
    dispatch_size_y: u32,
    offset_resources: Option<OffsetCorrectionResources>,
    gain_resources: Option<GainCorrectionResources>,
}

impl CorrectionContext {
    pub fn new(gpu_resources: GPUResources, width: u32, height: u32) -> Self {
        let buffer_size = width * height * 2;

        let staging_buffer = gpu_resources.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("My Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readback_buffer = gpu_resources.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let workgroup_size_x = 16;
        let workgroup_size_y = 16;
        let dispatch_size_x = (width + workgroup_size_x - 1) / workgroup_size_x;
        let dispatch_size_y = (height + workgroup_size_y - 1) / workgroup_size_y;

        CorrectionContext {
            gpu_resources,
            width,
            height,
            offset_resources: None,
            gain_resources: None,
            dispatch_size_x,
            dispatch_size_y,
            readback_buffer,
            staging_buffer,
        }
    }

    pub fn enable_offset_pipeline(&mut self, dark_map: Vec<u16>, offset: u32) {
        self.offset_resources = Some(OffsetCorrectionResources::new(
            &self.gpu_resources.device,
            &self.gpu_resources.queue,
            &self.staging_buffer,
            &dark_map,
            self.width,
            self.height,
            offset,
        ));
    }

    pub fn enable_gain_pipeline(&mut self, gain_map: Vec<f32>) {
        self.gain_resources = Some(GainCorrectionResources::new(
            &self.gpu_resources.device,
            &self.gpu_resources.queue,
            &self.staging_buffer,
            &gain_map,
            self.width,
            self.height,
        ));
    }

    pub async fn process_image(&mut self, input_image: &Vec<u16>) -> Result<Vec<u16>, MyError> {
        let input_texture = block_on(create_image_texture(
            &self.gpu_resources.device,
            &self.gpu_resources.queue,
            &self.staging_buffer,
            bytemuck::cast_slice(input_image),
            wgpu::TextureFormat::R16Uint,
            "input_image",
            self.width,
            self.height,
            2,
        ))?;

        let input_texture_view = input_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.gpu_resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("image correction encoder"),
                });

        if let Some(offset_resources) = &self.offset_resources {
            let bind_group =
                offset_resources.get_bind_group(&self.gpu_resources.device, &input_texture_view);
            let pipeline = &offset_resources.pipeline;

            let mut compute_pass: wgpu::ComputePass<'_> =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("offset correction pass"),
                    timestamp_writes: None,
                });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(self.dispatch_size_x, self.dispatch_size_y, 1);
        }

        if let Some(gain_resources) = &self.gain_resources {
            let bind_group =
                gain_resources.get_bind_group(&self.gpu_resources.device, &input_texture_view);
            let pipeline = &gain_resources.pipeline;

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gain correction pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &input_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.readback_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.width * 2),
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

        let (sender, receiver) = flume::bounded(1);
        let buffer_slice = self.readback_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.gpu_resources.device.poll(wgpu::Maintain::Wait);
        if let Ok(Ok(())) = receiver.recv_async().await {
            let data = buffer_slice.get_mapped_range();
            //let reading_time = Instant::now();
            let result = bytemuck::cast_slice(&data).to_vec();
            //println!("Reading time: {:?}", reading_time.elapsed());
            drop(data);
            self.readback_buffer.unmap();

            return Ok(result);
        } else {
            panic!("failed to run compute on gpu!")
        }
    }
}

pub async fn create_image_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    staging_buffer: &wgpu::Buffer,
    data: &[u8],
    texture_format: wgpu::TextureFormat,
    label: &str,
    width: u32,
    height: u32,
    byte_size: u32,
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
        format: texture_format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::STORAGE_BINDING,
        label: Some(label),
        view_formats: &[texture_format],
    });

    queue.write_buffer(staging_buffer, 0, data);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("texture_encoder"),
    });

    encoder.copy_buffer_to_texture(
        wgpu::ImageCopyBuffer {
            buffer: &staging_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width * byte_size),
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

    use super::{initialise_gpu_resources, CorrectionContext, GPUResources};

    #[test]
    fn test_dark_correction() {
        let gpu_resources = initialise_gpu_resources().unwrap();

        let mut correction_context = CorrectionContext::new(gpu_resources, 6144, 6144);

        let mut dark_data = vec![0; 6144 * 6144];

        dark_data[0] = 1;

        correction_context.enable_offset_pipeline(dark_data, 300);

        let mut image_data = vec![0; 6144 * 6144];

        /*
        image_data[0] = 999;    image_data[1] = 999; image_data[2] = 999;
        image_data[256] = 100;                        image_data[258] = 200;
        image_data[513] = 0;    image_data[514] = 0; image_data[515] = 0;
        */

        image_data[1] = 200;


        let start = Instant::now();
        let data = block_on(correction_context.process_image(&image_data)).unwrap();

        println!{"{}", data[0]};
        println!("Total time {:?}", start.elapsed() / 100);
    }

    /*
    #[test]
    fn test_gain_correction() {
        println!("Running test gain correction");
        let gpu_resources = initialise_gpu_resources().unwrap();
        let mut correction_context = CorrectionContext::new(gpu_resources, 2048, 2048);

        let norm_map = vec![0.5; 2048 * 2048];
        correction_context.enable_gain_pipeline(norm_map);

        let image_data = vec![8000; 2048 * 2048];
        let start = Instant::now();
        for i in 0..1000 {
            let data = block_on(correction_context.process_image(&image_data)).unwrap();
        }
        println!("Total time {:?}", start.elapsed() / 1000)
    }

    #[test]
    fn test_two_corrections() {
        let gpu_resources = initialise_gpu_resources().unwrap();
        let mut correction_context = CorrectionContext::new(gpu_resources, 2048, 2048);

        let dark_data = vec![3000; 2048 * 2048];
        correction_context.enable_offset_pipeline(dark_data, 300);

        let norm_map = vec![0.5; 2048 * 2048];
        correction_context.enable_gain_pipeline(norm_map);

        let image_data = vec![8000; 2048 * 2048];
        let data = block_on(correction_context.process_image(&image_data)).unwrap();
    }
    */
}
