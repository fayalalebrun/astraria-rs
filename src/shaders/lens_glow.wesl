// Lens glow shader package for stellar lens flare effects
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

struct LensGlowUniform {
    glow_size: f32,           // Size factor for the glow
    screen_width: f32,        // Screen width in pixels
    screen_height: f32,       // Screen height in pixels
    _padding: f32,            // Padding for 16-byte alignment (WebGL2 requirement)
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
};

@group(0) @binding(0) var<uniform> mvp: StandardMVPUniform;
@group(1) @binding(0) var<uniform> lens_glow: LensGlowUniform;
@group(1) @binding(1) var glow_texture: texture_2d<f32>;
@group(1) @binding(2) var glow_sampler: sampler;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // BILLBOARD BEHAVIOR: Transform the star center to clip space first
    let star_center = vec4<f32>(0.0, 0.0, 0.0, 1.0); // Model space center
    var center_clip = mvp.mvp_matrix * star_center;
    
    // Check if star is behind camera (negative w means behind camera in clip space)
    if (center_clip.w <= 0.0) {
        // Star is behind camera - don't render lens glow
        out.clip_position = vec4<f32>(-2.0, -2.0, -1.0, 1.0); // Off-screen
        return out;
    }
    
    // Perform perspective divide to get normalized device coordinates
    let center_ndc = center_clip.xyz / center_clip.w;
    
    // Add the quad offset exactly like Java does
    // Java: offsetX = (width/screenWidth), offsetY = (height/screenHeight)
    // Both width and height have the same value in Java: (size / aspect_ratio)
    let aspect_ratio = lens_glow.screen_width / lens_glow.screen_height;
    let size_with_aspect = lens_glow.glow_size / aspect_ratio;
    
    let offset_x = size_with_aspect / lens_glow.screen_width;
    let offset_y = size_with_aspect / lens_glow.screen_height;
    
    let screen_offset = vec2<f32>(input.position.x * offset_x, input.position.y * offset_y);
    
    // Final position: star center + screen-aligned offset
    out.clip_position = vec4<f32>(
        center_ndc.x + screen_offset.x,
        center_ndc.y + screen_offset.y,
        0.001, // Force to render on top (near plane)
        1.0
    );
    
    out.tex_coord = input.tex_coord;
    
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample the original lens glow texture (star_glow.png)
    let tex_color = textureSample(glow_texture, glow_sampler, input.tex_coord);
    
    // For now, use the original texture directly without modification
    // The additive blending will handle the final color mixing
    return vec4<f32>(tex_color.rgb, tex_color.a);
}