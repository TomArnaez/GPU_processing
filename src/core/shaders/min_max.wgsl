const workgroup_size: u32 = 16u;

@group(0) @binding(0)
var gain_map: texture_2d<u32>;

@group(0) @binding(1)
var<storage, read_write> global_minimum: atomic<u32>;

@group(0) @binding(2)
var<storage, read_write> global_maximum: atomic<u32>;

var<workgroup> workgroup_minimum: atomic<u32>;
var<workgroup> workgroup_maximum: atomic<u32>;

@compute @workgroup_size(workgroup_size, workgroup_size)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    // Initialize workgroup minimum and maximum for the first thread of the workgroup
    if (global_id.x == 0u && global_id.y == 0u) {
        atomicStore(&workgroup_minimum, 0xFFFFFFFFu);
        atomicStore(&workgroup_maximum, 0u);
    }

    let pixel_value = textureLoad(gain_map, vec2<i32>(x, y), 0);
    atomicMin(&workgroup_minimum, pixel_value[0]);
    atomicMax(&workgroup_maximum, pixel_value[0]);

    workgroupBarrier();

    // Workgroup leader updates the global minimum
    if (global_id.x == 0u && global_id.y == 0u) {
        atomicStore(&global_minimum, 0xFFFFFFFFu);
        atomicMin(&global_minimum, workgroup_minimum);

        atomicStore(&global_maximum, 0u);
        atomicStore(&global_maximum, workgroup_maximum);
    }
}