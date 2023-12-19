use std::{
    io, mem,
    os::windows::io::AsHandle,
    sync::{Arc, Mutex, RwLock},
    time::Instant,
};

use futures::lock;
use log::debug;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, CommandBufferUsage, RecordingCommandBuffer,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Features, Queue,
        QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    sync::{self, GpuFuture},
    Validated, VulkanError, VulkanLibrary,
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

pub struct CorrectionsInner {
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    image_buffers: Arc<Vec<Subbuffer<[u16]>>>,
    result_buffer: Vec<Vec<u16>>,
    width: u32,
    height: u32,
    dark_map_resources: Arc<Option<DarkMapBufferResources>>,
    head_index: usize,
}

pub struct Corrections {
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    result_buffer: Subbuffer<[u16]>,
    readback_buffer: Subbuffer<[u16]>,
    staging_buffers: Vec<Subbuffer<[u16]>>,
    image_width: u32,
    image_height: u32,
    defect_buffer_resources: Option<DefectMapBufferResources>,
    gain_map_resources: Option<GainMapBufferResources>,
    inner: Arc<RwLock<CorrectionsInner>>,
}

impl Corrections {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        image_width: u32,
        image_height: u32,
        buffer_count: u32,
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
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        vec![0u16; (image_width*image_height) as usize] /* number of elements, matching the image size */,
    )
    .unwrap();

        let mut staging_buffers = Vec::new();
        let mut image_buffers = Vec::new();

        for i in 0..buffer_count {
            staging_buffers.push(
                Buffer::new_slice::<u16>(
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
                .unwrap(),
            );

            image_buffers.push(
                Buffer::new_slice::<u16>(
                    memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER
                            | BufferUsage::TRANSFER_SRC
                            | BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    (image_height * image_width) as u64,
                )
                .unwrap(),
            );
        }
        Corrections {
            device: device.clone(),
            queue: queue.clone(),
            memory_allocator,
            descriptor_set_allocator,
            staging_buffers,
            readback_buffer,
            result_buffer,
            image_width,
            image_height,
            defect_buffer_resources: None,
            gain_map_resources: None,
            inner: Arc::new(RwLock::new(CorrectionsInner {
                queue: queue.clone(),
                device: device.clone(),
                image_buffers: Arc::new(image_buffers),
                result_buffer: Vec::new(),
                command_buffer_allocator,
                width: image_width,
                height: image_height,
                dark_map_resources: Arc::new(None),
                head_index: 0,
            })),
        }
    }

    pub fn enable_dark_map_correction(&mut self, dark_map: &[u16], offset: u32) {
        let mut inner_lock = self.inner.write().unwrap();
        inner_lock.dark_map_resources = Arc::new(Some(DarkMapBufferResources::new(
            self.device.clone(),
            self.queue.clone(),
            inner_lock.command_buffer_allocator.clone(),
            self.memory_allocator.clone(),
            self.descriptor_set_allocator.clone(),
            dark_map,
            offset,
            self.image_height,
            self.image_width,
        )));
    }

    pub fn enable_gain_correction(&mut self, gain_map: &[f32]) {
        let mut inner_lock = self.inner.write().unwrap();

        self.gain_map_resources = Some(GainMapBufferResources::new(
            self.device.clone(),
            self.queue.clone(),
            inner_lock.command_buffer_allocator.clone(),
            self.memory_allocator.clone(),
            self.descriptor_set_allocator.clone(),
            &gain_map,
            self.image_height,
            self.image_width,
        ));
    }

    pub fn enable_defect_correction(&mut self, defect_map: &[u16]) {
        let mut inner_lock = self.inner.write().unwrap();

        self.defect_buffer_resources = Some(DefectMapBufferResources::new(
            self.device.clone(),
            self.queue.clone(),
            inner_lock.command_buffer_allocator.clone(),
            self.memory_allocator.clone(),
            self.descriptor_set_allocator.clone(),
            defect_map,
            self.image_height,
            self.image_width,
        ))
    }

    pub fn process_image(&mut self) {
        let inner = self.inner.clone();

        tokio::spawn(async move {
            let time = Instant::now();
            println!("Running {:?}", time);

            let mut inner_lock = inner.write().unwrap();
            let head_index = inner_lock.head_index;
            inner_lock.head_index += 1;

            let device = inner_lock.device.clone();
            let queue = inner_lock.queue.clone();
            let command_buffer_allocator = inner_lock.command_buffer_allocator.clone();
            let image_buffers = inner_lock.image_buffers.clone();
            let width = inner_lock.width;
            let height = inner_lock.height;
            let dark_map_resources = inner_lock.dark_map_resources.clone();
            println!("Locking time {:?}", time.elapsed());
            drop(inner_lock);

            image_buffers[head_index].write().unwrap();

            let mut builder = RecordingCommandBuffer::primary(
                command_buffer_allocator.clone(),
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            if let Some(dark_map_resources) = dark_map_resources.as_ref() {
                println!("Applying dark correction");
                dark_map_resources.apply_pipeline(
                    &mut builder,
                    width,
                    height,
                    image_buffers[head_index].clone(),
                );
            }

            let command_buffer = builder.end().unwrap();

            let future = sync::now(device.clone())
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                .then_signal_fence_and_flush();

            let time = Instant::now();

            match future.map_err(Validated::unwrap) {
                Ok(future) => {
                    future.wait(None).unwrap();
                    println!(
                        "Receiving data for head index {}, took time {:?}",
                        head_index,
                        time.elapsed()
                    );
                    let data = image_buffers[head_index].read().unwrap().to_vec();
                    println!("Async task completed {:?}", time);
                }
                Err(e) => {}
            }
        });

        /*


        /*
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.staging_buffer.clone(),
                self.image_buffer.clone(),
            ))
            .unwrap();
        */

        if let Some(dark_map_resources) = &self.dark_map_resources {
            dark_map_resources.apply_pipeline(
                &mut builder,
                self.image_width,
                self.image_height,
                self.image_buffers[0]
                .clone(),
            );
        }

        if let Some(gain_map_resources) = &self.gain_map_resources {
            gain_map_resources.apply_pipeline(
                &mut builder,
                self.image_width,
                self.image_height,
                self.image_buffers[0].clone(),
                self.result_buffer.clone(),
            );
        }

        if let Some(defect_map_resources) = &self.defect_buffer_resources {
            defect_map_resources.apply_pipeline(
                &mut builder,
                self.image_width,
                self.image_height,
                self.image_buffers[0].clone(),
                self.result_buffer.clone(),
            );
        }

        if (self.defect_buffer_resources.is_some()) {
            /*
            builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.result_buffer.clone(),
                self.readback_buffer.clone(),
            ))
            .unwrap();
            */
        } else {
            /*
            builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.image_buffer.clone(),
                self.readback_buffer.clone(),
            ))
            .unwrap();
            */
        }

        let command_buffer = builder.end().unwrap();

        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush();

        let time = Instant::now();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                future.wait(None).unwrap();
                println!("Time to compute {:?}", time.elapsed());
                let time = Instant::now();
                image.copy_from_slice(&self.result_buffer.read().unwrap());
                println!("Time to write {:?}", time.elapsed());
            }
            Err(e) => {
                println!("failed to flush future: {e}");
            }
        }
        */
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::{initialise_gpu_resources, Corrections};

    #[tokio::test(flavor = "multi_thread")]
    async fn test() {
        let gpu_resources = initialise_gpu_resources();
        let image_width: u32 = 4800;
        let image_height: u32 = 5800;
        let offset = 300;
        let buffer_count = 10;

        let mut correction_context = Corrections::new(
            gpu_resources.1.clone(),
            gpu_resources.0.clone(),
            image_width,
            image_height,
            buffer_count,
        );
        let mut image = vec![10u16; (image_height * image_width) as usize];
        let mut defect_map = vec![1u16; (image_height * image_width) as usize];
        let gain_map = vec![0.5f32; (image_height * image_width) as usize];
        let dark_map = vec![1u16; (image_height * image_width) as usize];

        defect_map[0] = 1;
        defect_map[1] = 0;
        defect_map[2] = 0;
        image[1] = 20;
        image[2] = 10;

        correction_context.enable_dark_map_correction(&dark_map, offset);
        //correction_context.enable_gain_correction(&gain_map);
        //correction_context.enable_defect_correction(&defect_map);
        let time = Instant::now();

        for i in 0..buffer_count {
            correction_context.process_image();
        }
        println!("Time to process image {:?}", time.elapsed() / buffer_count);
        loop {}
    }
}
