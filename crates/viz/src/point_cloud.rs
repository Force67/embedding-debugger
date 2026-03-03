use bytemuck::{Pod, Zeroable};
use embedding_core::projection::ProjectedPoint;
use iced::event::Status;
use iced::widget::shader;
use iced::widget::shader::wgpu;
use iced::{Color, Point, Rectangle, Size, mouse, keyboard};
use iced_core::window;
use nalgebra::Vector4;

use std::time::Instant;

use crate::camera::ArcballCamera;

// ─── Public data types ────────────────────────────────────────────────────────

/// A single renderable point with position and color.
#[derive(Debug, Clone, Copy)]
pub struct PointData {
    pub position: [f32; 3],
    pub color: [f32; 4],
    pub size: f32,
    /// Index back into the original embedding set.
    pub index: usize,
}

impl PointData {
    pub fn from_projected(point: &ProjectedPoint, color: Color, size: f32) -> Self {
        Self {
            position: [point.x, point.y, point.z],
            color: [color.r, color.g, color.b, color.a],
            size,
            index: point.index,
        }
    }
}

/// High-level point cloud state stored in the application.
#[derive(Debug, Clone, Default)]
pub struct PointCloud {
    pub points: Vec<PointData>,
    /// Kept for API compatibility; the live interactive camera lives in
    /// `InteractionState` inside the shader widget.
    pub camera: ArcballCamera,
}

impl PointCloud {
    pub fn new(points: Vec<PointData>, camera: ArcballCamera) -> Self {
        Self { points, camera }
    }

    /// Update points from projected embeddings.
    pub fn set_points(&mut self, projected: &[ProjectedPoint], colors: &[Color]) {
        self.points = projected
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let color = colors.get(i).copied().unwrap_or(Color::WHITE);
                PointData::from_projected(p, color, 8.0)
            })
            .collect();
    }
}

// ─── Viewer events ────────────────────────────────────────────────────────────

/// Events emitted by the 3D viewer back to the application.
#[derive(Debug, Clone)]
pub enum ViewerEvent {
    /// A point was clicked (or deselected if `None`).
    PointSelected(Option<PointSelection>),
    /// The camera orientation changed.
    CameraChanged(ArcballCamera),
}

/// Information about a selected point.
#[derive(Debug, Clone)]
pub struct PointSelection {
    /// Index back into the original embedding set.
    pub index: usize,
    /// World-space position.
    pub position: [f32; 3],
}

// ─── GPU vertex / uniform types ───────────────────────────────────────────────

/// One vertex on the GPU.  Six of these form a billboard quad for each point.
///
/// Memory layout (stride = 48 bytes, all f32):
///   offset  0 – 11 : world_pos   (vec3)
///   offset 12 – 15 : _pad0
///   offset 16 – 31 : color       (vec4)
///   offset 32 – 35 : point_size  (f32)
///   offset 36 – 43 : quad_offset (vec2)
///   offset 44 – 47 : _pad1
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct GpuVertex {
    world_pos:   [f32; 3],
    _pad0:       f32,
    color:       [f32; 4],
    point_size:  f32,
    quad_offset: [f32; 2],
    _pad1:       f32,
}

/// Uniform block uploaded each frame (80 bytes, std140-compatible).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuUniforms {
    view_proj: [f32; 16], // 64 bytes
    viewport:  [f32; 2],  //  8 bytes – physical pixel dimensions of the widget
    _pad:      [f32; 2],  //  8 bytes padding → total 80, multiple of 16 ✓
}

// ─── Billboard quad expansion ─────────────────────────────────────────────────

/// Corner offsets for the two triangles that form a billboard quad.
const QUAD_CORNERS: [[f32; 2]; 6] = [
    [-1.0, -1.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0],
    [-1.0, -1.0],
    [ 1.0,  1.0],
    [-1.0,  1.0],
];

fn expand_to_vertices(points: &[PointData]) -> Vec<GpuVertex> {
    let mut verts = Vec::with_capacity(points.len() * 6);
    for p in points {
        for &qo in &QUAD_CORNERS {
            verts.push(GpuVertex {
                world_pos:   p.position,
                _pad0:       0.0,
                color:       p.color,
                point_size:  p.size,
                quad_offset: qo,
                _pad1:       0.0,
            });
        }
    }
    verts
}

// ─── WGSL shader source ───────────────────────────────────────────────────────

const SHADER_SRC: &str = r#"
struct Uniforms {
    view_proj : mat4x4<f32>,
    viewport  : vec2<f32>,
    _pad      : vec2<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertIn {
    @location(0) world_pos   : vec3<f32>,
    @location(1) color       : vec4<f32>,
    @location(2) point_size  : f32,
    @location(3) quad_offset : vec2<f32>,
}

struct VertOut {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0)       color    : vec4<f32>,
    @location(1)       uv       : vec2<f32>,
    @location(2)       fog      : f32,
}

@vertex
fn vs_main(in: VertIn) -> VertOut {
    var out: VertOut;

    let cc = u.view_proj * vec4<f32>(in.world_pos, 1.0);

    // Perspective size attenuation with slightly stronger scaling.
    let depth_attn = clamp(2.0 / max(0.3, cc.w * 0.45), 0.6, 3.5);
    let sz = in.point_size * depth_attn;

    let rx = sz * 2.0 / u.viewport.x * cc.w;
    let ry = sz * 2.0 / u.viewport.y * cc.w;

    out.clip_pos = vec4<f32>(
        cc.x + in.quad_offset.x * rx,
        cc.y + in.quad_offset.y * ry,
        cc.z,
        cc.w,
    );
    out.color = in.color;
    out.uv    = in.quad_offset;
    out.fog   = clamp(exp(-cc.w * 0.35), 0.12, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    let d = length(in.uv);
    if d > 1.0 { discard; }

    let base = in.color.rgb;
    let sphere_r = 0.7;

    var rgb: vec3<f32>;
    var alpha: f32;

    if d <= sphere_r {
        // ── Sphere impostor: compute normal from billboard UV ─────────────
        let s = d / sphere_r;
        let nz = sqrt(max(0.0, 1.0 - s * s));
        let normal = normalize(vec3<f32>(in.uv.x / sphere_r, in.uv.y / sphere_r, nz));

        // Two-light rig for richer shading.
        let key_dir  = normalize(vec3<f32>(0.5, 0.8, 0.6));
        let fill_dir = normalize(vec3<f32>(-0.4, -0.1, 0.7));
        let view_dir = vec3<f32>(0.0, 0.0, 1.0);

        // Diffuse (wrapped slightly for softer falloff).
        let key_diff  = max(dot(normal, key_dir), 0.0);
        let fill_diff = max(dot(normal, fill_dir), 0.0) * 0.25;

        // Blinn-Phong specular highlight.
        let half_v = normalize(key_dir + view_dir);
        let spec   = pow(max(dot(normal, half_v), 0.0), 64.0);

        // Fresnel rim gives silhouette definition.
        let fresnel = pow(1.0 - max(dot(normal, view_dir), 0.0), 3.0);

        // Ambient occlusion hint — darken edges.
        let ao = smoothstep(0.0, 0.35, nz) * 0.45 + 0.55;

        let ambient = 0.14;
        rgb = base * (ambient + key_diff * 0.68 + fill_diff) * ao
            + vec3<f32>(1.0) * spec * 0.50
            + base * fresnel * 0.22;

        // Anti-alias the sphere edge.
        alpha = smoothstep(sphere_r, sphere_r - 0.025, d);
    } else {
        // ── Soft glow halo around the sphere ──────────────────────────────
        let t    = (d - sphere_r) / (1.0 - sphere_r);
        let glow = pow(max(0.0, 1.0 - t), 2.6) * 0.30;
        rgb   = base * 0.45;
        alpha = glow;
    }

    // ── Depth fog: distant points dissolve into dark space ────────────────
    rgb = mix(vec3<f32>(0.03, 0.03, 0.09), rgb, in.fog);

    // ── Premultiplied alpha output ────────────────────────────────────────
    return vec4<f32>(rgb * alpha, alpha);
}
"#;

// ─── wgpu render pipeline wrapper ────────────────────────────────────────────

struct GpuPipeline {
    pipeline:       wgpu::RenderPipeline,
    uniform_buf:    wgpu::Buffer,
    vertex_buf:     wgpu::Buffer,
    bind_group:     wgpu::BindGroup,
    vertex_count:   u32,
    // Depth buffer for proper occlusion.
    depth_view:     Option<wgpu::TextureView>,
    depth_size:     (u32, u32),
}

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

impl GpuPipeline {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("point_cloud::shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("point_cloud::uniforms"),
            size:               std::mem::size_of::<GpuUniforms>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("point_cloud::bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty:         wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("point_cloud::bg"),
            layout:  &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        // Start with a small placeholder; will grow as needed.
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("point_cloud::vb"),
            size:               std::mem::size_of::<GpuVertex>() as u64 * 6,
            usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label:                Some("point_cloud::pl"),
                bind_group_layouts:   &[&bgl],
                push_constant_ranges: &[],
            });

        // Vertex attribute descriptors with explicit byte offsets.
        let vert_attrs = [
            wgpu::VertexAttribute {
                format:           wgpu::VertexFormat::Float32x3,
                offset:           0,
                shader_location:  0,
            },
            wgpu::VertexAttribute {
                format:           wgpu::VertexFormat::Float32x4,
                offset:           16,
                shader_location:  1,
            },
            wgpu::VertexAttribute {
                format:           wgpu::VertexFormat::Float32,
                offset:           32,
                shader_location:  2,
            },
            wgpu::VertexAttribute {
                format:           wgpu::VertexFormat::Float32x2,
                offset:           36,
                shader_location:  3,
            },
        ];

        let pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label:  Some("point_cloud::rp"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module:      &shader,
                    entry_point: "vs_main",
                    buffers:     &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<GpuVertex>() as u64,
                        step_mode:    wgpu::VertexStepMode::Vertex,
                        attributes:   &vert_attrs,
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module:      &shader,
                    entry_point: "fs_main",
                    targets:     &[Some(wgpu::ColorTargetState {
                        format,
                        blend:      Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology:  wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format:              DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare:       wgpu::CompareFunction::LessEqual,
                    stencil:             wgpu::StencilState::default(),
                    bias:                wgpu::DepthBiasState::default(),
                }),
                multisample:   wgpu::MultisampleState::default(),
                multiview:     None,
            });

        Self {
            pipeline,
            uniform_buf,
            vertex_buf,
            bind_group,
            vertex_count: 0,
            depth_view:   None,
            depth_size:   (0, 0),
        }
    }

    /// Ensure the depth texture matches the render target size.
    fn ensure_depth_texture(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        if self.depth_size == (width, height) && self.depth_view.is_some() {
            return;
        }
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("point_cloud::depth_tex"),
            size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          DEPTH_FORMAT,
            usage:           wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats:    &[],
        });
        self.depth_view = Some(tex.create_view(&wgpu::TextureViewDescriptor::default()));
        self.depth_size = (width, height);
    }

    fn update(
        &mut self,
        device:   &wgpu::Device,
        queue:    &wgpu::Queue,
        uniforms: &GpuUniforms,
        vertices: &[GpuVertex],
    ) {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(uniforms));

        let needed = (vertices.len() * std::mem::size_of::<GpuVertex>()) as u64;
        if needed == 0 {
            self.vertex_count = 0;
            return;
        }

        // Grow the buffer if required (round up to next power-of-two size).
        if self.vertex_buf.size() < needed {
            self.vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("point_cloud::vb"),
                size:               needed.next_power_of_two(),
                usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        queue.write_buffer(&self.vertex_buf, 0, bytemuck::cast_slice(vertices));
        self.vertex_count = vertices.len() as u32;
    }

    fn render(
        &self,
        target:   &wgpu::TextureView,
        viewport: Rectangle<u32>,
        encoder:  &mut wgpu::CommandEncoder,
    ) {
        if self.vertex_count == 0 {
            return;
        }

        let depth_view = match &self.depth_view {
            Some(v) => v,
            None => return,
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label:                    Some("point_cloud::pass"),
            color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
                view:           target,
                resolve_target: None,
                ops:            wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view:       depth_view,
                depth_ops:  Some(wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes:         None,
            occlusion_query_set:      None,
        });

        pass.set_viewport(
            viewport.x as f32,
            viewport.y as f32,
            viewport.width as f32,
            viewport.height as f32,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(
            viewport.x,
            viewport.y,
            viewport.width,
            viewport.height,
        );
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.draw(0..self.vertex_count, 0..1);
    }
}

// ─── iced Primitive ───────────────────────────────────────────────────────────

/// The data snapshot sent to the GPU each frame.
#[derive(Debug, Clone)]
pub struct PointCloudPrimitive {
    vertices: Vec<GpuVertex>,
    camera:   ArcballCamera,
}

impl shader::Primitive for PointCloudPrimitive {
    fn prepare(
        &self,
        format:       wgpu::TextureFormat,
        device:       &wgpu::Device,
        queue:        &wgpu::Queue,
        bounds:       Rectangle,
        target_size:  Size<u32>,
        scale_factor: f32,
        storage:      &mut shader::Storage,
    ) {
        if !storage.has::<GpuPipeline>() {
            storage.store(GpuPipeline::new(device, format));
        }
        let gpu = storage.get_mut::<GpuPipeline>().unwrap();

        // Ensure the depth texture matches the render target size.
        gpu.ensure_depth_texture(device, target_size.width, target_size.height);

        // Re-derive projection with the correct aspect ratio for this frame.
        let mut cam   = self.camera.clone();
        cam.aspect    = if bounds.height > 0.0 { bounds.width / bounds.height } else { 1.0 };
        let vp        = cam.view_projection();
        let mut vp_arr = [0.0f32; 16];
        vp_arr.copy_from_slice(vp.as_slice());

        let uniforms = GpuUniforms {
            view_proj: vp_arr,
            viewport:  [
                bounds.width  * scale_factor,
                bounds.height * scale_factor,
            ],
            _pad: [0.0; 2],
        };

        gpu.update(device, queue, &uniforms, &self.vertices);
    }

    fn render(
        &self,
        storage:      &shader::Storage,
        target:       &wgpu::TextureView,
        _target_size: Size<u32>,
        viewport:     Rectangle<u32>,
        encoder:      &mut wgpu::CommandEncoder,
    ) {
        if let Some(gpu) = storage.get::<GpuPipeline>() {
            gpu.render(target, viewport, encoder);
        }
    }
}

// ─── Depth-sort helper ────────────────────────────────────────────────────────

#[inline]
fn dist_sq(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

// ─── Point picking ────────────────────────────────────────────────────────────

/// Screen-space picking radius in pixels.
const PICK_THRESHOLD: f32 = 20.0;
/// Minimum cursor movement (px) before a click is treated as a drag.
const DRAG_THRESHOLD: f32 = 4.0;

/// Find the closest point to `click` in screen space, if any.
fn pick_point(
    camera: &ArcballCamera,
    points: &[PointData],
    click:  Point,
    bounds: Rectangle,
) -> Option<usize> {
    let mut cam = camera.clone();
    cam.aspect = if bounds.height > 0.0 { bounds.width / bounds.height } else { 1.0 };
    let vp = cam.view_projection();

    let threshold_sq = PICK_THRESHOLD * PICK_THRESHOLD;
    let mut best: Option<usize> = None;
    let mut best_d2 = threshold_sq;

    for (i, p) in points.iter().enumerate() {
        let clip = vp * Vector4::new(p.position[0], p.position[1], p.position[2], 1.0);
        if clip.w <= 0.001 { continue; }

        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;

        let sx = (ndc_x + 1.0) * 0.5 * bounds.width  + bounds.x;
        let sy = (1.0 - ndc_y) * 0.5 * bounds.height + bounds.y;

        let dx = sx - click.x;
        let dy = sy - click.y;
        let d2 = dx * dx + dy * dy;

        if d2 < best_d2 {
            best = Some(i);
            best_d2 = d2;
        }
    }
    best
}

// ─── iced Shader Program ─────────────────────────────────────────────────────

/// iced `Shader` program that drives the point cloud.
///
/// The program holds only the (immutable-per-frame) point data.
/// All interactive camera state lives in [`InteractionState`].
#[derive(Debug, Clone)]
pub struct PointCloudProgram {
    pub cloud: PointCloud,
}

// ── Per-widget interaction state (persists across frames) ──

/// Tracks the camera and mouse-drag state for the shader widget.
#[derive(Debug, Clone)]
pub struct InteractionState {
    pub camera: ArcballCamera,
    drag:       Option<DragState>,
    /// Timestamp of last click for double-click detection.
    last_click: Option<Instant>,
    /// Currently selected point (by embedding index).
    pub selected: Option<usize>,
}

impl Default for InteractionState {
    fn default() -> Self {
        Self {
            camera:     ArcballCamera::default(),
            drag:       None,
            last_click: None,
            selected:   None,
        }
    }
}

/// Double-click threshold in milliseconds.
const DOUBLE_CLICK_MS: u128 = 350;

#[derive(Debug, Clone)]
struct DragState {
    button:    mouse::Button,
    start_pos: Point,
    last_pos:  Point,
    has_moved: bool,
}

impl shader::Program<ViewerEvent> for PointCloudProgram {
    type State     = InteractionState;
    type Primitive = PointCloudPrimitive;

    fn draw(
        &self,
        state:   &Self::State,
        _cursor: mouse::Cursor,
        _bounds: Rectangle,
    ) -> Self::Primitive {
        let eye = state.camera.eye();
        let eye_pos = [eye.x, eye.y, eye.z];

        let mut sorted = self.cloud.points.clone();

        // Highlight the selected point (bigger + brighter).
        if let Some(sel_idx) = state.selected {
            if let Some(p) = sorted.iter_mut().find(|p| p.index == sel_idx) {
                p.size *= 1.8;
                p.color = [
                    (p.color[0] * 0.5 + 0.5).min(1.0),
                    (p.color[1] * 0.5 + 0.5).min(1.0),
                    (p.color[2] * 0.5 + 0.5).min(1.0),
                    p.color[3],
                ];
            }
        }

        sorted.sort_unstable_by(|a, b| {
            let da = dist_sq(&a.position, &eye_pos);
            let db = dist_sq(&b.position, &eye_pos);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });

        PointCloudPrimitive {
            vertices: expand_to_vertices(&sorted),
            camera:   state.camera.clone(),
        }
    }

    fn update(
        &self,
        state:  &mut Self::State,
        event:  shader::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
        shell:  &mut iced_core::Shell<'_, ViewerEvent>,
    ) -> (Status, Option<ViewerEvent>) {
        match event {
            // ── Mouse events ──────────────────────────────────────────────
            shader::Event::Mouse(mouse_event) => match mouse_event {
                mouse::Event::ButtonPressed(btn) => {
                    if let Some(abs) = cursor.position() {
                        if cursor.is_over(bounds) {
                            // Double-click → reset camera.
                            if btn == mouse::Button::Left {
                                let now = Instant::now();
                                if let Some(prev) = state.last_click {
                                    if now.duration_since(prev).as_millis() < DOUBLE_CLICK_MS {
                                        state.camera.reset();
                                        state.last_click = None;
                                        shell.request_redraw(window::RedrawRequest::NextFrame);
                                        return (Status::Captured, Some(ViewerEvent::CameraChanged(state.camera.clone())));
                                    }
                                }
                                state.last_click = Some(now);
                            }

                            state.drag = Some(DragState {
                                button:    btn,
                                start_pos: abs,
                                last_pos:  abs,
                                has_moved: false,
                            });
                            shell.request_redraw(window::RedrawRequest::NextFrame);
                            return (Status::Captured, None);
                        }
                    }
                    (Status::Ignored, None)
                }

                mouse::Event::ButtonReleased(_) => {
                    if let Some(drag) = state.drag.take() {
                        // Left-click without movement → point picking.
                        if drag.button == mouse::Button::Left && !drag.has_moved {
                            if let Some(vec_idx) = pick_point(
                                &state.camera,
                                &self.cloud.points,
                                drag.start_pos,
                                bounds,
                            ) {
                                let point = &self.cloud.points[vec_idx];
                                state.selected = Some(point.index);
                                shell.request_redraw(window::RedrawRequest::NextFrame);
                                return (Status::Captured, Some(ViewerEvent::PointSelected(
                                    Some(PointSelection {
                                        index:    point.index,
                                        position: point.position,
                                    })
                                )));
                            } else {
                                // Clicked empty space → deselect.
                                if state.selected.is_some() {
                                    state.selected = None;
                                    shell.request_redraw(window::RedrawRequest::NextFrame);
                                    return (Status::Captured, Some(ViewerEvent::PointSelected(None)));
                                }
                            }
                        } else if drag.has_moved {
                            return (Status::Captured, Some(ViewerEvent::CameraChanged(state.camera.clone())));
                        }
                        return (Status::Captured, None);
                    }
                    (Status::Ignored, None)
                }

                mouse::Event::CursorMoved { position } => {
                    if let Some(ref mut drag) = state.drag {
                        let dx = position.x - drag.last_pos.x;
                        let dy = position.y - drag.last_pos.y;
                        drag.last_pos = position;

                        let total_dx = position.x - drag.start_pos.x;
                        let total_dy = position.y - drag.start_pos.y;
                        if total_dx * total_dx + total_dy * total_dy > DRAG_THRESHOLD * DRAG_THRESHOLD {
                            drag.has_moved = true;
                        }

                        if drag.has_moved {
                            match drag.button {
                                mouse::Button::Left => {
                                    state.camera.rotate(dx * 0.005, dy * 0.005);
                                }
                                mouse::Button::Right | mouse::Button::Middle => {
                                    let scale = state.camera.distance * 0.0018;
                                    state.camera.pan(dx * scale, -dy * scale);
                                }
                                _ => {}
                            }
                            shell.request_redraw(window::RedrawRequest::NextFrame);
                            return (Status::Captured, Some(ViewerEvent::CameraChanged(state.camera.clone())));
                        }

                        shell.request_redraw(window::RedrawRequest::NextFrame);
                        return (Status::Captured, None);
                    }
                    (Status::Ignored, None)
                }

                mouse::Event::WheelScrolled { delta } => {
                    if cursor.is_over(bounds) {
                        let scroll = match delta {
                            mouse::ScrollDelta::Lines  { y, .. } => y,
                            mouse::ScrollDelta::Pixels { y, .. } => y * 0.02,
                        };
                        state.camera.zoom(scroll * 2.5);
                        shell.request_redraw(window::RedrawRequest::NextFrame);
                        return (Status::Captured, Some(ViewerEvent::CameraChanged(state.camera.clone())));
                    }
                    (Status::Ignored, None)
                }

                _ => (Status::Ignored, None),
            },

            // ── Keyboard events ───────────────────────────────────────────
            shader::Event::Keyboard(kb_event) => {
                if let keyboard::Event::KeyPressed { key, .. } = kb_event {
                    if !cursor.is_over(bounds) {
                        return (Status::Ignored, None);
                    }

                    let pan_step = state.camera.distance * 0.04;
                    let rot_step = 0.06_f32;

                    let handled = match &key {
                        keyboard::Key::Character(c) => match c.as_str() {
                            "w" => { state.camera.pan(0.0, pan_step); true }
                            "s" => { state.camera.pan(0.0, -pan_step); true }
                            "a" => { state.camera.pan(-pan_step, 0.0); true }
                            "d" => { state.camera.pan(pan_step, 0.0); true }
                            "q" => { state.camera.zoom(3.0); true }
                            "e" => { state.camera.zoom(-3.0); true }
                            "r" => { state.camera.reset(); true }
                            "f" => {
                                // Focus: reset target to origin, keep orientation.
                                state.camera.target = nalgebra::Point3::origin();
                                true
                            }
                            _ => false,
                        },
                        keyboard::Key::Named(named) => {
                            use keyboard::key::Named as N;
                            match named {
                                N::ArrowUp    => { state.camera.rotate(0.0, -rot_step); true }
                                N::ArrowDown  => { state.camera.rotate(0.0,  rot_step); true }
                                N::ArrowLeft  => { state.camera.rotate(-rot_step, 0.0); true }
                                N::ArrowRight => { state.camera.rotate( rot_step, 0.0); true }
                                N::Escape => {
                                    if state.selected.is_some() {
                                        state.selected = None;
                                        shell.request_redraw(window::RedrawRequest::NextFrame);
                                        return (Status::Captured, Some(ViewerEvent::PointSelected(None)));
                                    }
                                    false
                                }
                                _ => false,
                            }
                        }
                        _ => false,
                    };

                    if handled {
                        shell.request_redraw(window::RedrawRequest::NextFrame);
                        return (Status::Captured, Some(ViewerEvent::CameraChanged(state.camera.clone())));
                    }
                }
                (Status::Ignored, None)
            }

            _ => (Status::Ignored, None),
        }
    }

    fn mouse_interaction(
        &self,
        state:  &Self::State,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        if state.drag.is_some() {
            mouse::Interaction::Grabbing
        } else if cursor.is_over(bounds) {
            mouse::Interaction::Grab
        } else {
            mouse::Interaction::default()
        }
    }
}
