const MAX_PIXEL_VAL: u32 = 16383u;
const MIN_PIXEL_VALUE = 0u;

@group(0) @binding(0)
var normalised_gain_map: texture_storage_2d<r32float, read>;
@group(0) @binding(1)
var input_image: texture_storage_2d<r16uint, read_write>;

@compute @workgroup_size(16, 16)
fn gain_correct(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    let pixel_value = textureLoad(input_image, vec2<i32>(x, y));

    if (pixel_value[0] < MAX_PIXEL_VAL) {
        let gain_value = textureLoad(normalised_gain_map, vec2<i32>(x, y));
        
        let casted = f32(pixel_value[0]) * gain_value[0];

        textureStore(input_image, vec2<i32>(x, y), vec4<u32>(u32(casted), 0u, 0u, 0u));
    }
}