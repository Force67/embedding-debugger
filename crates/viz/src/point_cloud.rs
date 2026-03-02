use embedding_core::projection::ProjectedPoint;
use iced::event::Status;
use iced::widget::shader;
use iced::{Color, Rectangle, Size, mouse};
use nalgebra::Matrix4;

use crate::camera::ArcballCamera;

/// GPU-rendered 3D point cloud using iced's shader widget.
///
/// This is the core visualization component that displays embedding points
/// in 3D space. It integrates with iced's `Shader` primitive for custom
/// wgpu rendering, enabling smooth interaction even with thousands of points.
#[derive(Debug, Clone)]
pub struct PointCloud {
    /// The projected 3D points to render.
    pub points: Vec<PointData>,
    /// The camera controlling the view.
    pub camera: ArcballCamera,
}

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

impl Default for PointCloud {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            camera: ArcballCamera::default(),
        }
    }
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
                PointData::from_projected(p, color, 4.0)
            })
            .collect();
    }
}

// ---- iced Shader integration ----

/// The primitive sent to the GPU each frame.
#[derive(Debug, Clone)]
pub struct PointCloudPrimitive {
    /// MVP matrix as column-major f32 array.
    pub view_projection: [f32; 16],
    /// All points to draw this frame.
    pub points: Vec<PointData>,
}

impl PointCloudPrimitive {
    pub fn new(camera: &ArcballCamera, points: &[PointData]) -> Self {
        let vp: Matrix4<f32> = camera.view_projection();
        let mut vp_array = [0.0f32; 16];
        vp_array.copy_from_slice(vp.as_slice());

        Self {
            view_projection: vp_array,
            points: points.to_vec(),
        }
    }
}

impl shader::Primitive for PointCloudPrimitive {
    fn prepare(
        &self,
        _format: iced::widget::shader::wgpu::TextureFormat,
        _device: &iced::widget::shader::wgpu::Device,
        _queue: &iced::widget::shader::wgpu::Queue,
        _bounds: Rectangle,
        _viewport: Size<u32>,
        _scale_factor: f32,
        _storage: &mut shader::Storage,
    ) {
        // TODO: Upload point data to GPU buffers.
        // This will create/update:
        // - A vertex buffer with point positions + colors
        // - A uniform buffer with the view-projection matrix
        // - The render pipeline (on first call)
    }

    fn render(
        &self,
        _storage: &shader::Storage,
        _target: &iced::widget::shader::wgpu::TextureView,
        _target_size: Size<u32>,
        _clip_bounds: Rectangle<u32>,
        _encoder: &mut iced::widget::shader::wgpu::CommandEncoder,
    ) {
        // TODO: Record a render pass that draws point sprites.
        // - Bind the pipeline
        // - Set vertex buffer
        // - Set uniform bind group
        // - Draw points
    }
}

/// iced `Shader` program that drives the point cloud rendering.
#[derive(Debug, Clone)]
pub struct PointCloudProgram {
    pub cloud: PointCloud,
}

impl shader::Program<()> for PointCloudProgram {
    type State = ();
    type Primitive = PointCloudPrimitive;

    fn draw(
        &self,
        _state: &Self::State,
        _cursor: mouse::Cursor,
        _bounds: Rectangle,
    ) -> Self::Primitive {
        PointCloudPrimitive::new(&self.cloud.camera, &self.cloud.points)
    }

    fn update(
        &self,
        _state: &mut Self::State,
        _event: shader::Event,
        _bounds: Rectangle,
        _cursor: mouse::Cursor,
        _shell: &mut iced_core::Shell<'_, ()>,
    ) -> (Status, Option<()>) {
        // TODO: Handle mouse drag for rotation, scroll for zoom, middle-click for pan.
        (Status::Ignored, None)
    }

    fn mouse_interaction(
        &self,
        _state: &Self::State,
        _bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> mouse::Interaction {
        mouse::Interaction::default()
    }
}
