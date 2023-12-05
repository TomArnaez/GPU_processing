@group(0) @binding(0)
var dark_map: texture_storage_2d<r16uint, read>;
@group(0) @binding(1)
var input_image: texture_storage_2d<r16uint, read_write>;
@group(0) @binding(2)
var<uniform> offset: u32;

@compute @workgroup_size(1, 1)
fn correct_image(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    let input_pixel = textureLoad(input_image, vec2<i32>(x, y));
    let dark_pixel = textureLoad(dark_map, vec2<i32>(x, y));

    let corrected_pixel = input_pixel - dark_pixel +  vec4<u32>(offset, offset, offset, offset);

    textureStore(input_image, vec2<i32>(x, y), vec4<u32>(corrected_pixel));
}