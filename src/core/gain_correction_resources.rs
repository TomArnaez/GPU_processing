use std::borrow::Cow;

use pollster::block_on;

use super::core::create_image_texture;

pub struct GainCorrectionResources {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl GainCorrectionResources {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, map_data: &Vec<u16>, width: u32, height: u32) -> Self {
        block_on(perform_gain_normalisation(device, queue, map_data, width, height, 16));

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "shaders/normalise.wgsl"
            ))),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<u32>() as u64),
                    },    
                    count: None,
                }
            ],
            label: Some("bind_group_layout"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gain Correction Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
    
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gain Correction Pipeline Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        GainCorrectionResources {
            bind_group_layout,
            pipeline
        }
    }
}

async fn perform_gain_normalisation(device: &wgpu::Device, queue: &wgpu::Queue, map_data: &Vec<u16>, width: u32, height: u32, workgroup_size: u32) -> Result<Vec<u32>, ()> {
    let gain_map_texture = block_on(create_image_texture(&device, &queue, &map_data, "dark_map", width, height)).unwrap();
    let gain_map_texture_view = gain_map_texture
    .create_view(&wgpu::TextureViewDescriptor::default());

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
            "shaders/normalise.wgsl"
        ))),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
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
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<u32>() as u64),
                },    
                count: None,
            }
        ],
        label: Some("bind_group_layout"),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Gain Correction Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Gain Correction Pipeline Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    let atomic_min_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Compute Atomic Min Buffer"),
        size: std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Atomic Min Buffer"),
        size: std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&gain_map_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: atomic_min_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None
        });
    
        // 3. Set Pipeline and Bind Group
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
    
        // 4. Dispatch the Compute Shader
        // Assuming workgroup_size is defined elsewhere as u32
        let x_workgroups = (width + workgroup_size - 1) / workgroup_size;
        let y_workgroups = (height + workgroup_size - 1) / workgroup_size;
        compute_pass.dispatch_workgroups(x_workgroups, y_workgroups, 1);
    }

    encoder.copy_buffer_to_buffer(
        &atomic_min_buffer, 0,
        &readback_buffer, 0,
        std::mem::size_of::<u32>() as u64,
    );

    queue.submit(Some(encoder.finish()));


    let buffer_slice = readback_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(wgpu::Maintain::Wait);
    if let Ok(Ok(())) = receiver.recv_async().await {
        let data = buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        readback_buffer.unmap();

        println!("{:?}", result);

        return Ok(result);
    } else {
        panic!("failed to run compute on gpu!")
    }    
}