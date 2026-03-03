use bytemuck::{Pod, Zeroable};
use embedding_core::projection::ProjectedPoint;
use iced::event::Status;
use iced::widget::shader;
use iced::widget::shader::wgpu;
use iced::{Color, Point, Rectangle, Size, mouse};
use iced_core::window;

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
                PointData::from_projected(p, color, 5.0)
            })
            .collect();
    }
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
}

@vertex
fn vs_main(in: VertIn) -> VertOut {
    var out: VertOut;

    let cc = u.view_proj * vec4<f32>(in.world_pos, 1.0);

    // Convert point_size (radius in px) to clip-space offset.
    // NDC [-1,1] maps to viewport pixels, so 1px = 2/W or 2/H NDC units.
    // Multiply by w to convert from NDC to clip coordinates.
    let rx = in.point_size * 2.0 / u.viewport.x * cc.w;
    let ry = in.point_size * 2.0 / u.viewport.y * cc.w;

    out.clip_pos = vec4<f32>(
        cc.x + in.quad_offset.x * rx,
        cc.y + in.quad_offset.y * ry,
        cc.z,
        cc.w,
    );
    out.color = in.color;
    out.uv    = in.quad_offset;
    return out;
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    let d = length(in.uv);
    if d > 1.0 { discard; }

    // Smooth circular edge + bright specular core.
    let alpha      = 1.0 - smoothstep(0.55, 1.0, d);
    let brightness = 1.0 + 0.55 * max(0.0, 1.0 - d * 2.5);
    return vec4<f32>(in.color.rgb * brightness, in.color.a * alpha);
}
"#;

// ─── wgpu render pipeline wrapper ────────────────────────────────────────────

struct GpuPipeline {
    pipeline:     wgpu::RenderPipeline,
    uniform_buf:  wgpu::Buffer,
    vertex_buf:   wgpu::Buffer,
    bind_group:   wgpu::BindGroup,
    vertex_count: u32,
}

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
                visibility: wgpu::ShaderStages::VERTEX,
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
                        blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology:  wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample:   wgpu::MultisampleState::default(),
                multiview:     None,
            });

        Self {
            pipeline,
            uniform_buf,
            vertex_buf,
            bind_group,
            vertex_count: 0,
        }
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
            depth_stencil_attachment: None,
            timestamp_writes:         None,
            occlusion_query_set:      None,
        });

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
        _target_size: Size<u32>,
        scale_factor: f32,
        storage:      &mut shader::Storage,
    ) {
        if !storage.has::<GpuPipeline>() {
            storage.store(GpuPipeline::new(device, format));
        }
        let gpu = storage.get_mut::<GpuPipeline>().unwrap();

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
#[derive(Debug, Clone, Default)]
pub struct InteractionState {
    pub camera: ArcballCamera,
    drag:       Option<DragState>,
}

#[derive(Debug, Clone)]
struct DragState {
    button:   mouse::Button,
    /// Last cursor position in *window* coordinates (absolute).
    last_pos: Point,
}

impl shader::Program<()> for PointCloudProgram {
    type State     = InteractionState;
    type Primitive = PointCloudPrimitive;

    fn draw(
        &self,
        state:   &Self::State,
        _cursor: mouse::Cursor,
        _bounds: Rectangle,
    ) -> Self::Primitive {
        PointCloudPrimitive {
            vertices: expand_to_vertices(&self.cloud.points),
            camera:   state.camera.clone(),
        }
    }

    fn update(
        &self,
        state:  &mut Self::State,
        event:  shader::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
        shell:  &mut iced_core::Shell<'_, ()>,
    ) -> (Status, Option<()>) {
        let shader::Event::Mouse(mouse_event) = event else {
            return (Status::Ignored, None);
        };

        match mouse_event {
            // ── Start drag ──
            mouse::Event::ButtonPressed(btn) => {
                if let Some(abs) = cursor.position() {
                    if cursor.is_over(bounds) {
                        state.drag = Some(DragState { button: btn, last_pos: abs });
                        shell.request_redraw(window::RedrawRequest::NextFrame);
                        return (Status::Captured, None);
                    }
                }
            }

            // ── End drag ──
            mouse::Event::ButtonReleased(_) => {
                if state.drag.take().is_some() {
                    return (Status::Captured, None);
                }
            }

            // ── Drag motion → rotate or pan ──
            mouse::Event::CursorMoved { position } => {
                if let Some(ref mut drag) = state.drag {
                    let dx = position.x - drag.last_pos.x;
                    let dy = position.y - drag.last_pos.y;
                    drag.last_pos = position;

                    match drag.button {
                        mouse::Button::Left => {
                            state.camera.rotate(dx * 0.005, dy * 0.005);
                        }
                        mouse::Button::Right | mouse::Button::Middle => {
                            let scale = state.camera.distance * 0.002;
                            state.camera.pan(dx * scale, -dy * scale);
                        }
                        _ => {}
                    }

                    shell.request_redraw(window::RedrawRequest::NextFrame);
                    return (Status::Captured, None);
                }
            }

            // ── Scroll → zoom ──
            mouse::Event::WheelScrolled { delta } => {
                if cursor.is_over(bounds) {
                    let scroll = match delta {
                        mouse::ScrollDelta::Lines  { y, .. } => y,
                        mouse::ScrollDelta::Pixels { y, .. } => y * 0.02,
                    };
                    state.camera.zoom(-scroll * 0.35);
                    shell.request_redraw(window::RedrawRequest::NextFrame);
                    return (Status::Captured, None);
                }
            }

            _ => {}
        }

        (Status::Ignored, None)
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
