use std::borrow::Cow;

use pollster::block_on;
use wgpu::util::DeviceExt;

use crate::core::core::create_image_texture;

pub struct DefectCorrectionResources {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub defect_map_texture: wgpu::Texture,
    pub kernel_buffer: wgpu::Buffer,
}

impl DefectCorrectionResources {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, staging_buffer: &wgpu::Buffer, defect_map_data: &[u16], width: u32, height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/defect_correction_kernel.wgsl"
            ))),
        });

        let defect_map_texture = block_on(create_image_texture(&device, &queue, staging_buffer, bytemuck::cast_slice(defect_map_data), wgpu::TextureFormat::R16Uint, "defect_map", width, height, 2)).unwrap();

        let kernel_data: Vec<u32> = vec![1, 2, 3, 2, 1];
        let kernel_buffer_size = (kernel_data.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress;

        let kernel_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Defect Correction Kernel Buffer"),
            contents: bytemuck::cast_slice(&kernel_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                // Defect map texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: wgpu::TextureFormat::R16Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Input image texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
                        format: wgpu::TextureFormat::R16Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Kernel buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        min_binding_size: wgpu::BufferSize::new(kernel_buffer_size),
                        has_dynamic_offset: false,
                    },
                    count: None,
                },
                // Pass direction
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: wgpu::BufferSize::new(4) },
                    count: None
                }
            ],
            label: Some("Defect Correction Bind Group Layout"),
        });
    
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Defect Correction Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
    
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Defect Correction Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "correct_image", // Name of the compute function in your WGSL
        });

        DefectCorrectionResources {
            pipeline,
            bind_group_layout,
            defect_map_texture,
            kernel_buffer
        }
    }

    pub fn get_bind_group(&self, device: &wgpu::Device, input_texture_view: &wgpu::TextureView, vertical_pass: u32) -> wgpu::BindGroup {
        let dark_map_texture_view = self
        .defect_map_texture
        .create_view(&wgpu::TextureViewDescriptor::default());

        let pass_direction_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Defect Correction Kernel Buffer"),
            contents: bytemuck::cast_slice(&[vertical_pass]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
        });

        device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                    resource: self.kernel_buffer.as_entire_binding()
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pass_direction_buffer.as_entire_binding()
                }
            ],
            label: Some("Defect Correction Bind Group"),
        })
    }
}