use std::{sync::Arc, time::Instant, mem};

use log::debug;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, CommandBufferUsage, CopyBufferInfo,
        RecordingCommandBuffer,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Features, Queue,
        QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::{allocator::{
        AllocationCreateInfo, GenericMemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
    }, MemoryAllocateInfo},
    sync::{self, GpuFuture},
    VulkanLibrary,
};

use super::corrections::{
    dark_correction::DarkMapBufferResources, defect_correction::DefectMapBufferResources,
    gain_correction::GainMapBufferResources,
};

pub fn initialise_gpu_resources() -> (Arc<Queue>, Arc<Device>) {
    let library = VulkanLibrary::new().unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .unwrap();

    // Choose which physical device to use.
    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .unwrap();

    debug!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    for (index, heap) in physical_device
        .memory_properties()
        .memory_heaps
        .iter()
        .enumerate()
    {
        debug!(
            "Heap #{:?} has a capacity of {:?} bytes with flags {:?}",
            index, heap.size, heap.flags
        );
    }

    for ty in physical_device.memory_properties().memory_types.iter() {
        debug!(
            "Memory type belongs to heap #{:?}, with flags: {:?}",
            ty.heap_index, ty.property_flags
        );
    }

    let features = Features {
        storage_buffer16_bit_access: true,
        shader_int16: true,
        ..Features::default()
    };

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            enabled_features: features,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    (queue, device)
}

pub struct Corrections {
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    image_buffer: Subbuffer<[u16]>,
    result_buffer: Subbuffer<[u16]>,
    readback_buffer: Subbuffer<[u16]>,
    staging_buffer: Subbuffer<[u16]>,
    image_width: u32,
    image_height: u32,
    dark_map_resources: Option<DarkMapBufferResources>,
    defect_buffer_resources: Option<DefectMapBufferResources>,
    gain_map_resources: Option<GainMapBufferResources>,
}

impl Corrections {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        image_width: u32,
        image_height: u32,
    ) -> Self {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let staging_buffer = Buffer::new_slice::<u16>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC | BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            (image_height * image_width) as u64,
        )
        .unwrap();

        let image_buffer = Buffer::new_slice::<u16>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                    ..Default::default()
            },
            (image_height * image_width) as u64,
        )
        .unwrap();

        let readback_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        vec![0u16; (image_width*image_height) as usize] /* number of elements, matching the image size */,
    )
    .unwrap();

        let result_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vec![0u16; (image_width*image_height) as usize] /* number of elements, matching the image size */,
    )
    .unwrap();

        Corrections {
            device,
            queue,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            image_buffer,
            staging_buffer,
            readback_buffer,
            result_buffer,
            image_width,
            image_height,
            dark_map_resources: None,
            defect_buffer_resources: None,
            gain_map_resources: None,
        }
    }

    pub fn enable_dark_map_correction(&mut self, dark_map: Vec<u16>, offset: u32) {
        self.dark_map_resources = Some(DarkMapBufferResources::new(
            self.device.clone(),
            self.queue.clone(),
            self.command_buffer_allocator.clone(),
            self.memory_allocator.clone(),
            self.descriptor_set_allocator.clone(),
            dark_map,
            offset,
            self.image_height,
            self.image_width,
        ));
    }

    pub fn enable_gain_correction(&mut self, gain_map: Vec<f32>) {
        self.gain_map_resources = Some(GainMapBufferResources::new(
            self.device.clone(),
            self.queue.clone(),
            self.command_buffer_allocator.clone(),
            self.memory_allocator.clone(),
            self.descriptor_set_allocator.clone(),
            gain_map,
            self.image_height,
            self.image_width,
        ));
    }

    pub fn enable_defect_correction(&mut self, defect_map: Vec<u16>) {
        self.defect_buffer_resources = Some(DefectMapBufferResources::new(
            self.device.clone(),
            self.queue.clone(),
            self.command_buffer_allocator.clone(),
            self.memory_allocator.clone(),
            self.descriptor_set_allocator.clone(),
            defect_map,
            self.image_height,
            self.image_width,
        ))
    }

    pub fn process_image(&mut self, image: Vec<u16>) {
        let time = Instant::now();
        self.staging_buffer.write().unwrap().copy_from_slice(image.as_slice());


        let mut builder = RecordingCommandBuffer::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();


        builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.staging_buffer.clone(),
                self.image_buffer.clone(),
            ))
            .unwrap();



        if let Some(dark_map_resources) = &self.dark_map_resources {
            dark_map_resources.apply_pipeline(
                &mut builder,
                self.image_width,
                self.image_height,
                self.image_buffer.clone(),
            );
        }


        if let Some(gain_map_resources) = &self.gain_map_resources {
            gain_map_resources.apply_pipeline(
                &mut builder,
                self.image_width,
                self.image_height,
                self.image_buffer.clone(),
                self.result_buffer.clone(),
            );
        }


        if let Some(defect_map_resources) = &self.defect_buffer_resources {
            defect_map_resources.apply_pipeline(
                &mut builder,
                self.image_width,
                self.image_height,
                self.image_buffer.clone(),
                self.result_buffer.clone(),
            );
        }

        builder
        .copy_buffer(CopyBufferInfo::buffers(
            self.image_buffer.clone(),
            self.readback_buffer.clone(),
        ))
        .unwrap();


        //println!("2 {:?}", time.elapsed());


        let command_buffer = builder.end().unwrap();

        let time: Instant = Instant::now();

        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        let mut data = self.readback_buffer.read().unwrap().to_vec();
        let ptr = data.as_mut_ptr();
        println!("Building primary buffer {:?}", time.elapsed());

        mem::forget(data);
    }
}



#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::{initialise_gpu_resources, Corrections};

    #[test]
    fn test() {
        let gpu_resources = initialise_gpu_resources();
        let image_width: u32 = 4800;
        let image_height: u32 = 5800;
        let offset = 300;

        let mut correction_context =
            Corrections::new(gpu_resources.1.clone(), gpu_resources.0.clone(), image_width, image_height);

        let mut defect_map = vec![1u16; (image_height * image_width) as usize];
        let gain_map = vec![0.5f32; (image_height * image_width) as usize];
        let dark_map = vec![1u16; (image_height * image_width) as usize];

        defect_map[0] = 1;
        defect_map[1] = 0;

        correction_context.enable_dark_map_correction(dark_map, offset);
        //correction_context.enable_gain_correction(gain_map);
        //correction_context.enable_defect_correction(defect_map);

        for i in 0..100 {
            let time = Instant::now();
            let mut image = vec![0u16; (image_height * image_width) as usize];
            correction_context.process_image(image);
            println!("{:?}", time.elapsed());
        }
    

    }
}
