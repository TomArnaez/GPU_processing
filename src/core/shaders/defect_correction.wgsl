@group(0) @binding(0)
var input_image: texture_storage_2d<r16uint, read_write>;
@group(0) @binding(1)
var defect_map: texture_storage_2d<r16uint, read>;
@group(0) @binding(2)
var output_image: texture_storage_2d<r16uint, write>;

@compute @workgroup_size(16, 16)
fn defect_correct(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(glboal_id.x);
    let y = i32(global_id.y);
}