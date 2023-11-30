@group(0) @binding(0)
var dark_map: texture_2d<u32>;
@group(0) @binding(1)
var input_image: texture_2d<u32>;
@group(0) @binding(2)
var output_image: texture_storage_2d<r32uint, write>;

@compute @workgroup_size(16, 16)
fn correct_image(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    let input_pixel = textureLoad(input_image, vec2<i32>(x, y), 0);
    let dark_pixel = textureLoad(dark_map, vec2<i32>(x, y), 0);

    let corrected_pixel = input_pixel - dark_pixel;

    textureStore(output_image, vec2<i32>(x, y), vec4<u32>(corrected_pixel[0], 0u, 0u, 1u));
}