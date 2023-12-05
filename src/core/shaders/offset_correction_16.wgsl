@group(0) @binding(0)
var dark_map: texture_storage_2d<r16uint, read>;
@group(0) @binding(1)
var<storage, read_write> image: array<u32>;
@group(0) @binding(2)
var<uniform> offset: u32;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    let dark_pixel = textureLoad(dark_map, vec2<i32>(x, y));

    let value = image[x + y * (3071)];
    let lower_u16 = (value & 0xFFFFu) - dark_pixel[0] + offset;
    var upper_u16 = ((value >> u32(16)) & 0xFFFFu) - dark_pixel[0] + offset;

    image[x + y * 3071] = (upper_u16 << u32(16)) | lower_u16;
}