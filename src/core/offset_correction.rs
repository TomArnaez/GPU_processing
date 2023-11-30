use std::borrow::Cow;

use pollster::block_on;
use super::core::create_image_texture;

pub struct OffsetCorrectionResources {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub dark_map_texture: wgpu::Texture,
}

impl OffsetCorrectionResources {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, dark_map_data: &Vec<u16>, width: u32, height: u32) -> Self {
        let dark_map_texture = block_on(create_image_texture(&device, &queue, &dark_map_data, "dark_map", width, height)).unwrap();

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

        OffsetCorrectionResources {
            pipeline,
            bind_group_layout,
            dark_map_texture
        }
    }

    pub fn get_bind_group(&self, device: &wgpu::Device, input_texture: &wgpu::Texture, output_texture: &wgpu::Texture) -> wgpu::BindGroup {
        let dark_map_texture_view = self
        .dark_map_texture
        .create_view(&wgpu::TextureViewDescriptor::default());

        let input_texture_view = input_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let output_texture_view =
            output_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
                    resource: wgpu::BindingResource::TextureView(&output_texture_view),
                },
            ],
            label: Some("offset_correction_bind_group"),
        })
    }
}