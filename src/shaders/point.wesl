// Point shader package for distant object rendering
// All definitions inlined for proper Rust binding generation

// Common vertex input structure
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coord: vec2<f32>,
    @location(2) normal: vec3<f32>,
};

struct StandardMVPUniform {
    mvp_matrix: mat4x4<f32>,
    camera_position: vec3<f32>,
    _padding1: f32,
    camera_direction: vec3<f32>,
    _padding2: f32,
    log_depth_constant: f32,
    far_plane_distance: f32,
    near_plane_distance: f32,
    fc_constant: f32,
    mv_matrix: mat4x4<f32>,
};

fn model_to_clip_coordinates(
    position: vec4<f32>,
    mvp_matrix: mat4x4<f32>,
    depth_constant: f32,
    far_plane_distance: f32
) -> vec4<f32> {
    var clip = mvp_matrix * position;
    let fcoef = 1.0 / log2(far_plane_distance + 1.0);
    clip.z = log2(max(1e-6, 1.0 + clip.w)) * fcoef * clip.w;
    return clip;
}

struct PointUniform {
    point_color: vec4<f32>,     // RGBA color of the point
    point_size: f32,            // Point size in pixels
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
};


struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> mvp: StandardMVPUniform;

@group(1) @binding(0) var<uniform> point: PointUniform;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Use pre-computed MVP matrix (calculated with 64-bit precision on CPU)
    out.clip_position = model_to_clip_coordinates(
        vec4<f32>(input.position, 1.0),
        mvp.mvp_matrix,
        mvp.log_depth_constant,
        mvp.far_plane_distance
    );
    // Note: point size not supported in WGSL, needs to be handled differently
    out.color = point.point_color;
    
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Simple circular point rendering
    // Note: WGSL doesn't have gl_PointCoord, so we'll just return the color
    // Point rendering would need to be implemented differently in WebGPU
    return input.color;
}