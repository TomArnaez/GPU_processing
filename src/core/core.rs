use std::time::Instant;

use pollster::block_on;
use wgpu::{Buffer, Instance, PowerPreference};
use wgpu_profiler::*;

use super::{
    corrections::{
        gain_correction_resources::GainCorrectionResources,
        offset_correction::OffsetCorrectionResources, defect_correction::DefectCorrectionResources,
    },
    error::MyError,
};

pub struct GPUResources {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

pub fn initialise_gpu_resources(
    power_preference: PowerPreference,
) -> Result<GPUResources, &'static str> {
    let instance = Instance::default();

    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference,
        ..Default::default()
    }))
    .ok_or("Failed to find an appropriate adapter")?;

    println!("GPU adapter: {:?}", adapter.get_info());

    let (device, queue) = block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES | wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
            limits: wgpu::Limits::default(),
            label: None,
        },
        None,
    ))
    .map_err(|_| "Failed to create device")?;

    Ok(GPUResources { device, queue })
}

fn scopes_to_console_recursive(results: &[GpuTimerQueryResult], indentation: u32) {
    for scope in results {
        if indentation > 0 {
            print!("{:<width$}", "|", width = 4);
        }

        println!(
            "{:.3}ms - {}",
            (scope.time.end - scope.time.start) * 1000.0,
            scope.label
        );

        if !scope.nested_queries.is_empty() {
            scopes_to_console_recursive(&scope.nested_queries, indentation + 1);
        }
    }
}

fn console_output(results: &Option<Vec<GpuTimerQueryResult>>) {
    print!("\x1B[2J\x1B[1;1H"); // Clear terminal and put cursor to first row first column
    println!("Welcome to wgpu_profiler demo!");
    println!();
    println!(
        "Press space to write out a trace file that can be viewed in chrome's chrome://tracing"
    );
    println!();
    match results {
        Some(results) => {
            scopes_to_console_recursive(results, 0);
        }
        None => println!("No profiling results available yet!"),
    }
}

pub struct CorrectionContext {
    gpu_resources: GPUResources,
    readback_buffer: Buffer,
    staging_buffer: Buffer,
    width: u32,
    height: u32,
    dispatch_size_x: u32,
    dispatch_size_y: u32,
    offset_resources: Option<OffsetCorrectionResources>,
    gain_resources: Option<GainCorrectionResources>,
    defect_resources: Option<DefectCorrectionResources>,
}

impl CorrectionContext {
    pub fn new(gpu_resources: GPUResources, width: u32, height: u32) -> Self {
        let buffer_size = width * height * 2;

        let staging_buffer = gpu_resources.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
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
            defect_resources: None,
            dispatch_size_x,
            dispatch_size_y,
            readback_buffer,
            staging_buffer,
        }
    }

    pub fn enable_offset_pipeline(&mut self, dark_map: &[u16], offset: u32) {
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

    pub fn enable_gain_pipeline(&mut self, gain_map: &[f32]) {
        self.gain_resources = Some(GainCorrectionResources::new(
            &self.gpu_resources.device,
            &self.gpu_resources.queue,
            &self.staging_buffer,
            &gain_map,
            self.width,
            self.height,
        ));
    }

    pub fn enable_defect_correction(&mut self, defect_map: &[u16]) {
        self.defect_resources = Some(DefectCorrectionResources::new(
            &self.gpu_resources.device,
            &self.gpu_resources.queue,
            &self.staging_buffer,
            &defect_map,
            self.width,
            self.height,
        ));
    }

    pub async fn process_image(&mut self, input_image: &[u16]) -> Result<Vec<u16>, MyError> {
        let mut encoder =
            self.gpu_resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("image correction encoder"),
                });
            
        let mut profiler = GpuProfiler::new(GpuProfilerSettings::default()).unwrap();
        {
            let mut scope = profiler.scope("correcting", &mut encoder, &self.gpu_resources.device);

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

            if let Some(offset_resources) = &self.offset_resources {
                let bind_group =
                    offset_resources.get_bind_group(&self.gpu_resources.device, &input_texture_view);
                let pipeline = &offset_resources.pipeline;

                let mut compute_pass = scope.scoped_compute_pass("offset correction pass", &self.gpu_resources.device);

                compute_pass.set_pipeline(pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch_workgroups(1, 1, 1);
            }

            if let Some(gain_resources) = &self.gain_resources {
                let bind_group =
                    gain_resources.get_bind_group(&self.gpu_resources.device, &input_texture_view);
                let pipeline = &gain_resources.pipeline;

                let mut compute_pass = scope.scoped_compute_pass("gain correction pass", &self.gpu_resources.device);

                compute_pass.set_pipeline(pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch_workgroups(16, 16, 1);
            }

            if let Some(defect_resources) = &self.defect_resources {
                let vertical_bind_group =
                defect_resources.get_bind_group(&self.gpu_resources.device, &input_texture_view, 1);
                //let horizontal_bind_group = defect_resources.get_bind_group(&self.gpu_resources.device, &input_texture_view, 0);

                let pipeline = &defect_resources.pipeline;

                let mut compute_pass = scope.scoped_compute_pass("defect correction pass", &self.gpu_resources.device);
                
                compute_pass.set_pipeline(pipeline);
                compute_pass.set_bind_group(0, &vertical_bind_group, &[]);
                compute_pass.dispatch_workgroups(1,1, 1);

                //compute_pass.set_bind_group(0, &horizontal_bind_group, &[]);
                //compute_pass.dispatch_workgroups(self.dispatch_size_x, self.dispatch_size_y, 1);
            }


            scope.copy_texture_to_buffer(
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


        }
        profiler.resolve_queries(&mut encoder);

        self.gpu_resources.queue.submit(Some(encoder.finish()));
        profiler.end_frame().unwrap();



        let (sender, receiver) = flume::bounded(1);
        let buffer_slice = self.readback_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.gpu_resources.device.poll(wgpu::Maintain::Wait);
        if let Ok(Ok(())) = receiver.recv_async().await {
            let data = buffer_slice.get_mapped_range();
            let time = Instant::now();
            let result = bytemuck::cast_slice(&data).to_vec();
            println!("Time to copy data {:?}", time.elapsed());
            //console_output(&profiler.process_finished_frame(self.gpu_resources.queue.get_timestamp_period()));

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
    fn test_defect_correction() {
        env_logger::init();
        let gpu_resources = initialise_gpu_resources(wgpu::PowerPreference::HighPerformance).unwrap();
        let mut correction_context = CorrectionContext::new(gpu_resources, 6144, 6144);

        let mut dark_data = vec![0; 6144 * 6144];

        dark_data[0] = 1;

        correction_context.enable_defect_correction(&dark_data);

        let mut image_data = vec![0; 6144 * 6144];

        image_data[0] = 10;
        image_data[1] = 10;
        image_data[2] = 10;


        let start = Instant::now();
        let data = block_on(correction_context.process_image(&image_data)).unwrap();
        println!("{}", data[1]);
        println!("{:?}", start.elapsed());
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
