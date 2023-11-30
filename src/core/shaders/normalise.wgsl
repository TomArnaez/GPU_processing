const workgroup_size: u32 = 16;

@group(0) @binding(0)
var gain_map: texture_2d<u32>;

@group(0) @binding(1)
var<workgroup> sharedMaxValues: array<u32, workgroup_size * workgroup_size>

@group(0) binding(2)
var<storage, read_write> localMaxima: array<u32>

@compute @workgroup_size(workgroup_size, workgroup_size)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index: u32 = global_id.x;
}