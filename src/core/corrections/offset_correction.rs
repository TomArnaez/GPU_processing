use std::borrow::Cow;

use pollster::block_on;
use wgpu::util::DeviceExt;

use crate::core::core::create_image_texture;

pub struct OffsetCorrectionResources {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub dark_map_texture: wgpu::Texture,
    pub offset_uniform_buffer: wgpu::Buffer,
}

impl OffsetCorrectionResources {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, staging_buffer: &wgpu::Buffer, dark_map_data: &Vec<u16>, width: u32, height: u32, offset: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/offset_correction.wgsl"
            ))),
        });

        let dark_map_texture = block_on(create_image_texture(&device, &queue, staging_buffer, bytemuck::cast_slice(dark_map_data), wgpu::TextureFormat::R16Uint, "dark_map", width, height, 2)).unwrap();
        
        let offset_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Offset Uniform Buffer"),
            contents: bytemuck::cast_slice(&[offset]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
        });
    
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                // Dark map texture
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
                // Offset
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
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

        OffsetCorrectionResources {
            pipeline,
            bind_group_layout,
            dark_map_texture,
            offset_uniform_buffer
        }
    }

    pub fn get_bind_group(&self, device: &wgpu::Device, input_texture_view: &wgpu::TextureView) -> wgpu::BindGroup {
        let dark_map_texture_view = self
        .dark_map_texture
        .create_view(&wgpu::TextureViewDescriptor::default());

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
                    resource: self.offset_uniform_buffer.as_entire_binding()
                }
            ],
            label: Some("offset_correction_bind_group"),
        })
    }
}