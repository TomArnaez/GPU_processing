use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RecordingCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    sync::{self, GpuFuture},
};

pub struct DarkMapBufferResources {
    pipeline: Arc<ComputePipeline>,
    dark_map_buffer: Subbuffer<[u16]>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl DarkMapBufferResources {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        dark_map: Vec<u16>,
        offset: u32,
        image_height: u32,
        image_width: u32,
    ) -> Self {
        let pipeline = {
            mod offset_correction_shader {
                vulkano_shaders::shader! {
                    ty: "compute",
                    src: r"
                            #version 450
                            #extension GL_EXT_shader_16bit_storage : require
                            #extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

                            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

                            layout(set = 0, binding = 0) buffer DarkMapData {
                                uint16_t darkMapData[];
                            };
                            layout(set = 0, binding = 1) buffer ImageData {
                                uint16_t imageData[];
                            };
                            layout(set = 0, binding = 2) buffer ResultData {
                                uint16_t resultData[];
                            };
        
                            void main() {
                                uint idx = gl_GlobalInvocationID.x;
                                resultData[idx] = imageData[idx] - darkMapData[idx] + uint16_t(300);
                            }
                        ",
                }
            }

            let cs = offset_correction_shader::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let dark_map_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            dark_map, /* number of elements, matching the image size */
        )
        .unwrap();

        let builder = RecordingCommandBuffer::primary(
            command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let command_buffer = builder.end().unwrap();

        let future = sync::now(device)
            .then_execute(queue, command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        DarkMapBufferResources {
            pipeline,
            dark_map_buffer,
            memory_allocator,
            descriptor_set_allocator,
        }
    }

    pub fn apply_pipeline(
        &self,
        builder: &mut RecordingCommandBuffer<PrimaryAutoCommandBuffer>,
        image_width: u32,
        image_height: u32,
        image_buffer: Subbuffer<[u16]>,
        result_buffer: Subbuffer<[u16]>,
    ) {
        let local_size_x = 64;

        let dispatch_size_x = (image_width * image_height + local_size_x - 1) / local_size_x;

        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, self.dark_map_buffer.clone()),
                WriteDescriptorSet::buffer(1, image_buffer),
                WriteDescriptorSet::buffer(2, result_buffer),
            ],
            [],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                set,
            )
            .unwrap()
            .dispatch([dispatch_size_x, 1, 1])
            .unwrap();
    }
}
