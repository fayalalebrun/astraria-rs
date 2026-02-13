use image::{DynamicImage, GenericImageView};
/// Asset loading and management system
/// Replaces LibGDX AssetManager with Rust-native implementation
use std::collections::HashMap;
use std::sync::{Arc, Weak};
use wgpu::{Buffer, Device, Queue, Texture, TextureView, util::DeviceExt};

use crate::{AstrariaError, AstrariaResult};

#[cfg(feature = "native")]
use crate::generated_shaders::common::VertexInput;

pub struct AssetManager {
    textures: HashMap<String, Arc<TextureAsset>>,
    models: HashMap<String, Arc<ModelAsset>>,
    cubemaps: HashMap<String, Arc<CubemapAsset>>,
    // Asset lifecycle tracking
    texture_handles: HashMap<String, Vec<Weak<TextureAsset>>>,
    model_handles: HashMap<String, Vec<Weak<ModelAsset>>>,
    cubemap_handles: HashMap<String, Vec<Weak<CubemapAsset>>>,
}

pub struct TextureAsset {
    pub texture: Texture,
    pub view: TextureView,
    pub width: u32,
    pub height: u32,
}

pub struct CubemapAsset {
    pub texture: Texture,
    pub view: TextureView,
    pub size: u32,
}

pub struct ModelAsset {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub num_indices: u32,
    pub num_vertices: u32,
    pub material_name: Option<String>,
}

impl AssetManager {
    pub async fn new() -> AstrariaResult<Self> {
        Ok(Self {
            textures: HashMap::new(),
            models: HashMap::new(),
            cubemaps: HashMap::new(),
            texture_handles: HashMap::new(),
            model_handles: HashMap::new(),
            cubemap_handles: HashMap::new(),
        })
    }

    /// Load a default white texture for testing
    pub fn create_default_texture(
        &mut self,
        device: &Device,
        queue: &Queue,
    ) -> AstrariaResult<Arc<TextureAsset>> {
        if let Some(texture) = self.textures.get("default_white") {
            return Ok(Arc::clone(texture));
        }

        // Create a simple 1x1 white texture
        let texture_size = wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Default White Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Write white pixel data
        let white_pixel = [255u8, 255u8, 255u8, 255u8];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &white_pixel,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            texture_size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_asset = TextureAsset {
            texture,
            view,
            width: 1,
            height: 1,
        };

        let texture_arc = Arc::new(texture_asset);
        self.textures
            .insert("default_white".to_string(), Arc::clone(&texture_arc));
        Ok(texture_arc)
    }

    pub async fn load_texture(
        &mut self,
        device: &Device,
        queue: &Queue,
        path: &str,
    ) -> AstrariaResult<Arc<TextureAsset>> {
        log::info!("AssetManager: Attempting to load texture: {}", path);

        if let Some(texture) = self.textures.get(path) {
            log::info!("AssetManager: Texture already cached: {}", path);
            return Ok(Arc::clone(texture));
        }

        // Resolve Java-style paths to actual file paths
        let resolved_path = Self::resolve_texture_path(path);
        log::info!(
            "AssetManager: Resolved path '{}' -> '{}'",
            path,
            resolved_path
        );

        // Check if we already loaded this resolved path
        if let Some(texture) = self.textures.get(&resolved_path).cloned() {
            // Cache under both original and resolved path for future lookups
            self.textures.insert(path.to_string(), Arc::clone(&texture));
            return Ok(texture);
        }

        // Load image data
        log::info!("AssetManager: Loading image file: {}", resolved_path);
        let img = Self::load_image_data(&resolved_path).await.map_err(|e| {
            log::error!(
                "AssetManager: Failed to load image {} (resolved from {}): {}",
                resolved_path,
                path,
                e
            );
            AstrariaError::AssetLoading(format!(
                "Failed to load image {} (resolved from {}): {}",
                resolved_path, path, e
            ))
        })?;

        let texture_asset =
            Self::create_texture_from_image(device, queue, &img, Some(&resolved_path))?;
        let texture_arc = Arc::new(texture_asset);

        // Cache under both original and resolved path
        self.textures
            .insert(path.to_string(), Arc::clone(&texture_arc));
        self.textures
            .insert(resolved_path.clone(), Arc::clone(&texture_arc));
        log::info!(
            "AssetManager: Successfully loaded and cached texture: {} -> {}",
            path,
            resolved_path
        );
        Ok(texture_arc)
    }

    #[cfg(feature = "native")]
    async fn load_image_data(path: &str) -> AstrariaResult<DynamicImage> {
        image::open(path).map_err(|e| {
            AstrariaError::AssetLoading(format!("Failed to load image {}: {}", path, e))
        })
    }

    #[cfg(feature = "web")]
    async fn load_image_data(path: &str) -> AstrariaResult<DynamicImage> {
        let bytes = Self::fetch_bytes(path).await?;

        // Determine format from file extension
        let format = Self::guess_image_format(path);

        match format {
            Some(fmt) => image::load_from_memory_with_format(&bytes, fmt).map_err(|e| {
                AstrariaError::AssetLoading(format!("Failed to decode image {}: {}", path, e))
            }),
            None => image::load_from_memory(&bytes).map_err(|e| {
                AstrariaError::AssetLoading(format!("Failed to decode image {}: {}", path, e))
            }),
        }
    }

    #[cfg(feature = "web")]
    fn guess_image_format(path: &str) -> Option<image::ImageFormat> {
        let path_lower = path.to_lowercase();
        if path_lower.ends_with(".png") {
            Some(image::ImageFormat::Png)
        } else if path_lower.ends_with(".jpg") || path_lower.ends_with(".jpeg") {
            Some(image::ImageFormat::Jpeg)
        } else if path_lower.ends_with(".gif") {
            Some(image::ImageFormat::Gif)
        } else if path_lower.ends_with(".bmp") {
            Some(image::ImageFormat::Bmp)
        } else if path_lower.ends_with(".webp") {
            Some(image::ImageFormat::WebP)
        } else {
            None
        }
    }

    fn create_texture_from_image(
        device: &Device,
        queue: &Queue,
        img: &DynamicImage,
        label: Option<&str>,
    ) -> AstrariaResult<TextureAsset> {
        // WebGL2 has a max texture dimension of 2048
        #[cfg(target_arch = "wasm32")]
        const MAX_TEXTURE_DIM: u32 = 2048;
        #[cfg(not(target_arch = "wasm32"))]
        const MAX_TEXTURE_DIM: u32 = 16384;

        let dimensions = img.dimensions();
        let img = if dimensions.0 > MAX_TEXTURE_DIM || dimensions.1 > MAX_TEXTURE_DIM {
            let scale = (MAX_TEXTURE_DIM as f32 / dimensions.0.max(dimensions.1) as f32).min(1.0);
            let new_width = (dimensions.0 as f32 * scale) as u32;
            let new_height = (dimensions.1 as f32 * scale) as u32;
            log::info!(
                "Resizing texture from {}x{} to {}x{} (max dimension: {})",
                dimensions.0, dimensions.1, new_width, new_height, MAX_TEXTURE_DIM
            );
            std::borrow::Cow::Owned(img.resize(new_width, new_height, image::imageops::FilterType::Triangle))
        } else {
            std::borrow::Cow::Borrowed(img)
        };

        let rgba = img.to_rgba8();
        let dimensions = img.dimensions();

        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            texture_size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Ok(TextureAsset {
            texture,
            view,
            width: dimensions.0,
            height: dimensions.1,
        })
    }

    #[cfg(feature = "native")]
    pub async fn load_scenario(&self, path: &str) -> AstrariaResult<String> {
        use std::fs;

        // Try to load from assets/examples/ directory first
        let full_path = format!("assets/examples/{}", path);

        match fs::read_to_string(&full_path) {
            Ok(content) => {
                log::info!("Loaded scenario file: {}", full_path);
                Ok(content)
            }
            Err(_) => {
                // Try direct path if not found in examples
                match fs::read_to_string(path) {
                    Ok(content) => {
                        log::info!("Loaded scenario file: {}", path);
                        Ok(content)
                    }
                    Err(e) => {
                        log::error!("Failed to load scenario file '{}': {}", path, e);
                        Err(crate::AstrariaError::AssetLoading(format!(
                            "Failed to load scenario file '{}': {}",
                            path, e
                        )))
                    }
                }
            }
        }
    }

    #[cfg(feature = "web")]
    pub async fn load_scenario(&self, path: &str) -> AstrariaResult<String> {
        // On web, fetch from HTTP
        let full_path = format!("assets/examples/{}", path);
        Self::fetch_text(&full_path).await
    }

    #[cfg(feature = "web")]
    async fn fetch_text(url: &str) -> AstrariaResult<String> {
        use wasm_bindgen::JsCast;
        use wasm_bindgen_futures::JsFuture;
        use web_sys::{Request, RequestInit, RequestMode, Response};

        let window = web_sys::window().ok_or_else(|| {
            AstrariaError::AssetLoading("No window object available".to_string())
        })?;

        let mut opts = RequestInit::new();
        opts.method("GET");
        opts.mode(RequestMode::Cors);

        let request = Request::new_with_str_and_init(url, &opts)
            .map_err(|e| AstrariaError::AssetLoading(format!("Failed to create request: {:?}", e)))?;

        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| AstrariaError::AssetLoading(format!("Fetch failed: {:?}", e)))?;

        let resp: Response = resp_value.dyn_into()
            .map_err(|_| AstrariaError::AssetLoading("Response cast failed".to_string()))?;

        let text = JsFuture::from(resp.text().map_err(|e| {
            AstrariaError::AssetLoading(format!("Failed to get response text: {:?}", e))
        })?)
        .await
        .map_err(|e| AstrariaError::AssetLoading(format!("Failed to read text: {:?}", e)))?;

        text.as_string()
            .ok_or_else(|| AstrariaError::AssetLoading("Response was not a string".to_string()))
    }

    #[cfg(feature = "web")]
    async fn fetch_bytes(url: &str) -> AstrariaResult<Vec<u8>> {
        use wasm_bindgen::JsCast;
        use wasm_bindgen_futures::JsFuture;
        use web_sys::{Request, RequestInit, RequestMode, Response};

        let window = web_sys::window().ok_or_else(|| {
            AstrariaError::AssetLoading("No window object available".to_string())
        })?;

        let mut opts = RequestInit::new();
        opts.method("GET");
        opts.mode(RequestMode::Cors);

        let request = Request::new_with_str_and_init(url, &opts)
            .map_err(|e| AstrariaError::AssetLoading(format!("Failed to create request: {:?}", e)))?;

        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| AstrariaError::AssetLoading(format!("Fetch failed: {:?}", e)))?;

        let resp: Response = resp_value.dyn_into()
            .map_err(|_| AstrariaError::AssetLoading("Response cast failed".to_string()))?;

        let array_buffer = JsFuture::from(resp.array_buffer().map_err(|e| {
            AstrariaError::AssetLoading(format!("Failed to get array buffer: {:?}", e))
        })?)
        .await
        .map_err(|e| AstrariaError::AssetLoading(format!("Failed to read array buffer: {:?}", e)))?;

        let uint8_array = js_sys::Uint8Array::new(&array_buffer);
        Ok(uint8_array.to_vec())
    }

    #[cfg(feature = "native")]
    pub async fn load_model(
        &mut self,
        device: &Device,
        path: &str,
    ) -> AstrariaResult<Arc<ModelAsset>> {
        if let Some(model) = self.models.get(path) {
            return Ok(Arc::clone(model));
        }

        log::info!("Loading OBJ model: {}", path);

        // Load OBJ file using tobj
        let (models, _materials) = tobj::load_obj(
            path,
            &tobj::LoadOptions {
                single_index: true,
                triangulate: true,
                ignore_points: true,
                ignore_lines: true,
            },
        )
        .map_err(|e| AstrariaError::AssetLoading(format!("Failed to load OBJ {}: {}", path, e)))?;

        if models.is_empty() {
            return Err(AstrariaError::AssetLoading(format!(
                "No models found in OBJ file: {}",
                path
            )));
        }

        // Use the first model for now
        let model = &models[0];
        let mesh = &model.mesh;

        // Convert to our vertex format using indices
        let mut vertices = Vec::new();
        let positions = &mesh.positions;
        let normals = &mesh.normals;
        let texcoords = &mesh.texcoords;

        // Build vertices based on the indices
        for &index in &mesh.indices {
            let pos_idx = (index as usize) * 3;
            let tex_idx = (index as usize) * 2;

            let vertex = VertexInput {
                position: if pos_idx + 2 < positions.len() {
                    glam::Vec3::new(
                        positions[pos_idx],
                        positions[pos_idx + 1],
                        positions[pos_idx + 2],
                    )
                } else {
                    glam::Vec3::ZERO
                },
                tex_coord: if tex_idx + 1 < texcoords.len() {
                    glam::Vec2::new(texcoords[tex_idx], texcoords[tex_idx + 1])
                } else {
                    glam::Vec2::ZERO
                },
                normal: if pos_idx + 2 < normals.len() {
                    glam::Vec3::new(normals[pos_idx], normals[pos_idx + 1], normals[pos_idx + 2])
                } else {
                    glam::Vec3::Y // Default up normal
                },
            };
            vertices.push(vertex);
        }

        // Create simple sequential indices since we've already expanded vertices
        let indices: Vec<u32> = (0..vertices.len() as u32).collect();

        // Create vertex buffer
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} Vertex Buffer", path)),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Create index buffer with our sequential indices
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} Index Buffer", path)),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let model_asset = ModelAsset {
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
            num_vertices: vertices.len() as u32,
            material_name: model.mesh.material_id.map(|id| format!("material_{}", id)),
        };

        log::info!(
            "Loaded OBJ model: {} vertices, {} indices",
            model_asset.num_vertices,
            model_asset.num_indices
        );

        let model_arc = Arc::new(model_asset);
        self.models.insert(path.to_string(), Arc::clone(&model_arc));
        Ok(model_arc)
    }

    #[cfg(feature = "web")]
    pub async fn load_model(
        &mut self,
        device: &Device,
        path: &str,
    ) -> AstrariaResult<Arc<ModelAsset>> {
        if let Some(model) = self.models.get(path) {
            return Ok(Arc::clone(model));
        }

        log::info!("Loading OBJ model from web: {}", path);

        // On web, we use a simple built-in sphere for now
        // A full OBJ parser could be added later
        let model_asset = Self::create_procedural_sphere(device, 32, 16)?;

        log::info!(
            "Created procedural sphere: {} vertices, {} indices",
            model_asset.num_vertices,
            model_asset.num_indices
        );

        let model_arc = Arc::new(model_asset);
        self.models.insert(path.to_string(), Arc::clone(&model_arc));
        Ok(model_arc)
    }

    #[cfg(feature = "web")]
    fn create_procedural_sphere(
        device: &Device,
        segments: u32,
        rings: u32,
    ) -> AstrariaResult<ModelAsset> {
        use std::f32::consts::PI;

        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut tex_coords = Vec::new();
        let mut indices = Vec::new();

        for ring in 0..=rings {
            let phi = PI * ring as f32 / rings as f32;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            for segment in 0..=segments {
                let theta = 2.0 * PI * segment as f32 / segments as f32;
                let sin_theta = theta.sin();
                let cos_theta = theta.cos();

                let x = cos_theta * sin_phi;
                let y = cos_phi;
                let z = sin_theta * sin_phi;

                positions.push(glam::Vec3::new(x, y, z));
                normals.push(glam::Vec3::new(x, y, z));
                tex_coords.push(glam::Vec2::new(
                    segment as f32 / segments as f32,
                    ring as f32 / rings as f32,
                ));
            }
        }

        for ring in 0..rings {
            for segment in 0..segments {
                let current = ring * (segments + 1) + segment;
                let next = current + segments + 1;

                indices.push(current);
                indices.push(next);
                indices.push(current + 1);

                indices.push(current + 1);
                indices.push(next);
                indices.push(next + 1);
            }
        }

        // Create vertices by combining position, normal, tex_coord
        let vertices: Vec<[f32; 8]> = positions
            .iter()
            .zip(normals.iter())
            .zip(tex_coords.iter())
            .map(|((pos, norm), tex)| {
                [pos.x, pos.y, pos.z, tex.x, tex.y, norm.x, norm.y, norm.z]
            })
            .collect();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Procedural Sphere Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Procedural Sphere Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Ok(ModelAsset {
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
            num_vertices: vertices.len() as u32,
            material_name: None,
        })
    }

    /// Get a loaded model by path
    pub fn get_model(&self, path: &str) -> Option<&ModelAsset> {
        self.models.get(path).map(|arc| arc.as_ref())
    }

    /// Get a shared reference to a loaded model
    pub fn get_model_handle(&self, path: &str) -> Option<Arc<ModelAsset>> {
        self.models.get(path).map(Arc::clone)
    }

    /// Get a shared reference to a loaded texture
    pub fn get_texture_handle(&self, path: &str) -> Option<Arc<TextureAsset>> {
        self.textures.get(path).map(Arc::clone)
    }

    /// Clean up unused assets (remove assets with no active handles)
    pub fn cleanup_unused_assets(&mut self) {
        // Clean up unused textures
        let mut unused_textures = Vec::new();
        for (path, texture_arc) in &self.textures {
            if Arc::strong_count(texture_arc) == 1 {
                unused_textures.push(path.clone());
            }
        }
        for path in unused_textures {
            self.textures.remove(&path);
            self.texture_handles.remove(&path);
            log::debug!("Cleaned up unused texture: {}", path);
        }

        // Clean up unused models
        let mut unused_models = Vec::new();
        for (path, model_arc) in &self.models {
            if Arc::strong_count(model_arc) == 1 {
                unused_models.push(path.clone());
            }
        }
        for path in unused_models {
            self.models.remove(&path);
            self.model_handles.remove(&path);
            log::debug!("Cleaned up unused model: {}", path);
        }

        // Clean up unused cubemaps
        let mut unused_cubemaps = Vec::new();
        for (name, cubemap_arc) in &self.cubemaps {
            if Arc::strong_count(cubemap_arc) == 1 {
                unused_cubemaps.push(name.clone());
            }
        }
        for name in unused_cubemaps {
            self.cubemaps.remove(&name);
            self.cubemap_handles.remove(&name);
            log::debug!("Cleaned up unused cubemap: {}", name);
        }
    }

    /// Load a cubemap from 6 face images
    pub async fn load_cubemap(
        &mut self,
        device: &Device,
        queue: &Queue,
        name: &str,
        face_paths: &[&str; 6], // +X, -X, +Y, -Y, +Z, -Z
    ) -> AstrariaResult<Arc<CubemapAsset>> {
        if let Some(cubemap) = self.cubemaps.get(name) {
            return Ok(Arc::clone(cubemap));
        }

        // WebGL2 has a max texture dimension of 2048
        #[cfg(target_arch = "wasm32")]
        const MAX_CUBEMAP_DIM: u32 = 2048;
        #[cfg(not(target_arch = "wasm32"))]
        const MAX_CUBEMAP_DIM: u32 = 16384;

        // Load all 6 face images
        let mut face_images = Vec::with_capacity(6);
        let mut cubemap_size = 0u32;

        for (i, path) in face_paths.iter().enumerate() {
            let img = Self::load_image_data(path).await.map_err(|e| {
                AstrariaError::AssetLoading(format!("Failed to load cubemap face {}: {}", path, e))
            })?;

            let (width, height) = img.dimensions();

            // Ensure all faces are square
            if width != height {
                return Err(AstrariaError::AssetLoading(format!(
                    "Cubemap face {} is not square: {}x{}",
                    path, width, height
                )));
            }

            // Resize if too large for WebGL2
            let img = if width > MAX_CUBEMAP_DIM {
                let new_size = MAX_CUBEMAP_DIM;
                log::info!(
                    "Resizing cubemap face from {}x{} to {}x{} (max dimension: {})",
                    width, height, new_size, new_size, MAX_CUBEMAP_DIM
                );
                img.resize_exact(new_size, new_size, image::imageops::FilterType::Triangle)
            } else {
                img
            };

            let rgba = img.to_rgba8();
            let (width, _height) = img.dimensions();

            if i == 0 {
                cubemap_size = width;
            } else if width != cubemap_size {
                return Err(AstrariaError::AssetLoading(format!(
                    "Cubemap face {} size mismatch: expected {}, got {}",
                    path, cubemap_size, width
                )));
            }

            face_images.push(rgba);
        }

        // Create cubemap texture
        let texture_size = wgpu::Extent3d {
            width: cubemap_size,
            height: cubemap_size,
            depth_or_array_layers: 6,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("Cubemap: {}", name)),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload each face
        for (i, face_data) in face_images.iter().enumerate() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    aspect: wgpu::TextureAspect::All,
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: i as u32,
                    },
                },
                face_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * cubemap_size),
                    rows_per_image: Some(cubemap_size),
                },
                wgpu::Extent3d {
                    width: cubemap_size,
                    height: cubemap_size,
                    depth_or_array_layers: 1,
                },
            );
        }

        // Create cube texture view
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("Cubemap View: {}", name)),
            format: None,
            dimension: Some(wgpu::TextureViewDimension::Cube),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
        });

        let cubemap_asset = CubemapAsset {
            texture,
            view,
            size: cubemap_size,
        };

        let cubemap_arc = Arc::new(cubemap_asset);
        self.cubemaps
            .insert(name.to_string(), Arc::clone(&cubemap_arc));

        log::info!("Loaded cubemap '{}' with size {}", name, cubemap_size);
        Ok(cubemap_arc)
    }

    /// Get a loaded cubemap by name
    pub fn get_cubemap(&self, name: &str) -> Option<&CubemapAsset> {
        self.cubemaps.get(name).map(|arc| arc.as_ref())
    }

    /// Get a shared reference to a loaded cubemap
    pub fn get_cubemap_handle(&self, name: &str) -> Option<Arc<CubemapAsset>> {
        self.cubemaps.get(name).map(Arc::clone)
    }

    /// Resolve Java-style texture paths to actual file system paths
    /// Converts "./Planet Textures/filename.jpg" -> "assets/Planet Textures/filename.jpg"
    #[cfg(feature = "native")]
    fn resolve_texture_path(path: &str) -> String {
        if path.starts_with("./") {
            // Java-style relative path, convert to assets directory
            let resolved = format!("assets/{}", &path[2..]);

            // Verify the file exists
            if std::path::Path::new(&resolved).exists() {
                resolved
            } else {
                // Try alternative asset directories
                let alternatives = [
                    format!("assets/{}", &path[2..]),
                    format!(
                        "assets/textures/{}",
                        &path[2..].replace("Planet Textures/", "")
                    ),
                    path.to_string(), // Try original path as-is
                ];

                for alt_path in &alternatives {
                    if std::path::Path::new(alt_path).exists() {
                        return alt_path.clone();
                    }
                }

                // Return the best guess if nothing exists
                resolved
            }
        } else if std::path::Path::new(path).exists() {
            // Already a valid path
            path.to_string()
        } else {
            // Try prepending assets directory
            let with_assets = format!("assets/{}", path);
            if std::path::Path::new(&with_assets).exists() {
                with_assets
            } else {
                // Return best guess
                with_assets
            }
        }
    }

    /// Resolve texture paths for web - just normalizes the path
    #[cfg(feature = "web")]
    fn resolve_texture_path(path: &str) -> String {
        if path.starts_with("./") {
            format!("assets/{}", &path[2..])
        } else if path.starts_with("assets/") {
            path.to_string()
        } else {
            format!("assets/{}", path)
        }
    }

    /// Load hardcoded atmospheric assets required by shaders
    pub async fn load_atmospheric_assets(
        &mut self,
        device: &Device,
        queue: &Queue,
    ) -> AstrariaResult<()> {
        log::info!("AssetManager: Loading atmospheric assets...");

        // Load atmospheric gradient texture (required for planet_atmo shader)
        if !self.textures.contains_key("atmoGradient.png")
            && !self.textures.contains_key("assets/atmoGradient.png")
        {
            match self.load_texture(device, queue, "atmoGradient.png").await {
                Ok(_) => {
                    log::info!("AssetManager: Successfully loaded atmospheric gradient texture")
                }
                Err(e) => log::warn!("AssetManager: Failed to load atmospheric gradient: {}", e),
            }
        } else {
            log::info!("AssetManager: Atmospheric gradient texture already loaded");
        }

        // Load star spectrum texture (required for star temperature mapping)
        if !self.textures.contains_key("star_spectrum_1D.png") {
            match self
                .load_texture(device, queue, "assets/star_spectrum_1D.png")
                .await
            {
                Ok(_) => log::info!("AssetManager: Successfully loaded star spectrum texture"),
                Err(e) => log::warn!("AssetManager: Failed to load star spectrum: {}", e),
            }
        } else {
            log::info!("AssetManager: Star spectrum texture already loaded");
        }

        log::info!("AssetManager: Atmospheric assets loading complete");
        Ok(())
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize, usize) {
        (self.textures.len(), self.models.len(), self.cubemaps.len())
    }
}
