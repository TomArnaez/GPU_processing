extern crate gpu_processing;

use criterion::{criterion_group, criterion_main, Criterion};
use gpu_processing::core::core::{Corrections, initialise_gpu_resources};

// You might need to replicate or import the `initialise_gpu_resources` function
// and other relevant dependencies.

fn corrections_benchmark(c: &mut Criterion) {
    c.bench_function("corrections_process_image", |b| {
        let gpu_resources = initialise_gpu_resources();
        let image_width: u32 = 4800;
        let image_height: u32 = 5800;

        let mut correction_context =
            Corrections::new(gpu_resources.1.clone(), gpu_resources.0.clone(), image_width, image_height);

        // Set up your image, defect_map, gain_map, and dark_map here as per your test

        b.iter(|| {
            let image = vec![0u16; (image_height * image_width) as usize];
            correction_context.process_image(image);
        });
    });
}

criterion_group!(benches, corrections_benchmark);
criterion_main!(benches);