@group(0) @binding(0)
var defect_map: texture_storage_2d<r16uint, read>;
@group(0) @binding(1)
var input_image: texture_storage_2d<r16uint, read_write>;
@group(0) @binding(2)
var<uniform> offset: u32;

var<private> neighbour_pixels: array<u32, 8>;

fn calculate_contrast(a: u32, b: u32) -> u32 {
    return u32(f32(a + b) / f32(2));
}

fn is_inside_bounds(pos: vec2<i32>, dimensions: vec2<i32>) -> bool {
    return pos.x >= 0 && pos.x < dimensions.x && pos.y >= 0 && pos.y < dimensions.y;
}

fn get_offset(direction: u32) -> vec2<i32> {
    var offsets: array<vec2<i32>, 8> = array<vec2<i32>, 8>(
        vec2<i32>(1, 0),   // Right
        vec2<i32>(-1, 0),  // Left
        vec2<i32>(0, 1),   // Up
        vec2<i32>(0, -1),  // Down
        vec2<i32>(1, 1),   // Up-Right
        vec2<i32>(-1, 1),  // Up-Left
        vec2<i32>(1, -1),  // Down-Right
        vec2<i32>(-1, -1)  // Down-Left
    );
    return offsets[direction];
}

fn average_line(x: i32, y: i32, dimensions: vec2<i32>) -> u32 {
    var best_contrast = u32(99999); // Initial high value indicating failure
    var best_contrast_found = u32(0); // Flag to indicate if a valid contrast was found

    for (var i = 0; i < 8; i++) {
        var neighbour_pos1 = vec2<i32>(x, y) + get_offset(u32(i));
        var neighbour_pos2 = vec2<i32>(x, y) - get_offset(u32(i));

        let in_bounds1 = u32(is_inside_bounds(neighbour_pos1, dimensions));
        let in_bounds2 = u32(is_inside_bounds(neighbour_pos2, dimensions));

        let neighbour_defect1 = in_bounds1 * textureLoad(defect_map, neighbour_pos1).r;
        let neighbour_defect2 = in_bounds2 * textureLoad(defect_map, neighbour_pos2).r;

        let neighbour_pixel1 = in_bounds1 * textureLoad(input_image, neighbour_pos1).r;
        let neighbour_pixel2 = in_bounds2 * textureLoad(input_image, neighbour_pos2).r;

        let both_inside_bounds = in_bounds1 * in_bounds2;
        let both_free_of_defects = (1u - neighbour_defect1) * (1u - neighbour_defect2);

        let current_contrast = both_inside_bounds * both_free_of_defects * calculate_contrast(neighbour_pixel1, neighbour_pixel2);
        
        // Update best contrast if a lower contrast (and valid) is found
        let is_better_contrast = both_inside_bounds * both_free_of_defects * u32(current_contrast < best_contrast);
        best_contrast = is_better_contrast * current_contrast + (1u - is_better_contrast) * best_contrast;
        best_contrast_found |= is_better_contrast;
    }

    return best_contrast_found * best_contrast + (1u - best_contrast_found) * u32(99999); // Return 99999 if no valid contrast found
}

fn average_neighbours(x: i32, y: i32) -> u32 {
    var total = u32(0);
    var count = u32(0);

    for (var i = 0; i < 8; i++) {
        total += neighbour_pixels[i];
        count += 1u;
    }
    
    // Avoid division by zero
    if (count > 0u) {
        return total / count;
    }

    return u32(99999); // Indicate failure
}

@compute @workgroup_size(16, 16)
fn correct_image(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    let dimensions = vec2<i32>(6144, 6144);

    for (var i = 0; i < 8; i++) {
        var neighbor_pos = vec2<i32>(x, y) + get_offset(u32(i));     

        let in_bounds = u32(is_inside_bounds(neighbor_pos, dimensions));

        let neighbor_defect =   in_bounds * textureLoad(defect_map, neighbor_pos).r;
        let neighbor_pixel  =   in_bounds * textureLoad(input_image, neighbor_pos).r;

        neighbour_pixels[i] = (1u - neighbor_defect) * neighbor_pixel;
    }

    let neighbour_average = average_neighbours(x, y);
    let line_average = average_line(x, y, dimensions);

    let defect_pixel = textureLoad(defect_map, vec2<i32>(x, y));
    let is_defective = defect_pixel.r > 0u;

    let use_line_average = is_defective && (line_average != u32(99999));

    textureStore(input_image, vec2<i32>(x, y), vec4<u32>(neighbour_average, 0u, 0u, 1u));
}