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

pub struct DefectMapBufferResources {
    pipeline: Arc<ComputePipeline>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    kernel_buffer: Subbuffer<[u16]>,
    defect_map_buffer: Subbuffer<[u16]>,
    direction_buffer: Subbuffer<[i32; 1]>,
}

impl DefectMapBufferResources {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        defect_map: &[u16],
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

                            #define KERNEL_SIZE 5

                            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

                            layout(set = 0, binding = 0) buffer DefectData {
                                uint16_t defectMapData[];
                            };

                            layout(set = 0, binding = 1) buffer ImageData {
                                uint16_t imageData[];
                            };

                            layout(set = 0, binding = 2) buffer ResultImage {
                                uint16_t resultData[];
                            };
                   
                            int kernel[5] = int[5](1, 2, 0, 2, 1);

                            // Define the weight kernel as a constant 2D array
                            const float weightKernel[KERNEL_SIZE][KERNEL_SIZE] = float[KERNEL_SIZE][KERNEL_SIZE](
                                float[KERNEL_SIZE](1.0, 2.0, 3.0, 2.0, 1.0),
                                float[KERNEL_SIZE](2.0, 3.0, 4.0, 3.0, 2.0),
                                float[KERNEL_SIZE](3.0, 4.0, 0.0, 4.0, 3.0),
                                float[KERNEL_SIZE](2.0, 3.0, 4.0, 3.0, 2.0),
                                float[KERNEL_SIZE](1.0, 2.0, 3.0, 2.0, 1.0)
                            );

                            void main() {
                                uint image_height = 5800;
                                uint image_width = 4800;

                                uint idx = gl_GlobalInvocationID.x;
                                float weightedSum = 0.0;
                                float totalWeight = 0.0;

                                if (defectMapData[idx] == 1) {
                                    for (int y = -KERNEL_SIZE / 2; y <= KERNEL_SIZE / 2; ++y) {
                                        for (int x = -KERNEL_SIZE / 2; x <= KERNEL_SIZE / 2; ++x) {
                                            int pixelX = int(idx % image_width) + x;
                                            int pixelY = int(idx / image_width) + y;

                                            if (pixelX >= 0 && pixelX < image_width && pixelY >= 0 && pixelY < image_height) {
                                                uint globalIndex = pixelY * image_width + pixelX;
                                                if (defectMapData[globalIndex] == 0) {
                                                    weightedSum += imageData[globalIndex] * weightKernel[y + KERNEL_SIZE / 2][x + KERNEL_SIZE / 2];
                                                    totalWeight += weightKernel[y + KERNEL_SIZE / 2][x + KERNEL_SIZE / 2];
                                                }
                                            }
                                        }
                                    }

                                    if (totalWeight > 0) {
                                        resultData[idx] = uint16_t(weightedSum / totalWeight);
                                    } else {
                                        resultData[idx] = imageData[idx];
                                    }
                                } else {
                                    resultData[idx] = imageData[idx];
                                }
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

        let defect_map_buffer = Buffer::new_slice(
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
            (image_height * image_width) as u64, /* number of elements, matching the image size */
        )
        .unwrap();

        defect_map_buffer.write().unwrap().copy_from_slice(defect_map);

        let kernel_buffer = Buffer::from_iter(
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
            vec![1u16, 2, 3, 2, 1],
        )
        .unwrap();

        let direction_buffer = Buffer::from_data(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST | BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            [0], // 0 for horizontal, 1 for vertical
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

        DefectMapBufferResources {
            pipeline,
            memory_allocator,
            descriptor_set_allocator,
            defect_map_buffer,
            kernel_buffer,
            direction_buffer,
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
                WriteDescriptorSet::buffer(0, self.defect_map_buffer.clone()),
                WriteDescriptorSet::buffer(1, image_buffer.clone()),
                WriteDescriptorSet::buffer(2, result_buffer.clone()),
                //WriteDescriptorSet::buffer(3, self.direction_buffer.clone()),
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
            .unwrap()
            .update_buffer(self.direction_buffer.clone(), &[1])
            .unwrap()
            .dispatch([dispatch_size_x, 1, 1])
            .unwrap();
    }
}
