@group(0) @binding(0)
var defect_map: texture_storage_2d<r16uint, read>;
@group(0) @binding(1)
var input_image: texture_storage_2d<r16uint, read_write>;
@group(0) @binding(2) 
var<storage> kernel: array<u32>;
@group(0) @binding(3)
var<uniform> is_vertical_pass: u32; // 0 for horizontal, non-zero for vertical

fn get_pass_coords(x: i32, y: i32, offset: i32) -> vec2<i32> {
    if (is_vertical_pass != 0u) {
        return vec2<i32>(x, y + offset); // Vertical pass
    } else {
        return vec2<i32>(x + offset, y); // Horizontal pass
    };
}

@compute @workgroup_size(16, 16)
fn correct_image(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    let kernel_size: u32 = arrayLength(&kernel);
    var result: u32 = 0u;

    for (var i: i32 = 0; i < 5; i++) {
        var offset: i32 = i - i32(kernel_size / 2u);

        let coords = get_pass_coords(x, y, offset);

        if (coords.x >= 0 && coords.x < 6144 && coords.y >= 0 && coords.y < 6144) {
            let value = textureLoad(input_image, coords).r;
            let defect_value = textureLoad(defect_map, coords).r;
            result += value * kernel[i] * (1u - defect_value);
        }
    }

    textureStore(input_image, vec2<i32>(x, y), vec4<u32>(result, 0u, 0u, 1u));
}