use nalgebra::{Matrix4, Point3, Vector3};

/// An arcball camera for orbiting around a 3D scene.
///
/// Supports smooth rotation, zoom, and panning — essential for
/// exploring a point cloud without frame drops.
#[derive(Debug, Clone)]
pub struct ArcballCamera {
    /// Where the camera is looking at (center of rotation).
    pub target: Point3<f32>,
    /// Distance from the target.
    pub distance: f32,
    /// Rotation around the vertical axis (radians).
    pub yaw: f32,
    /// Rotation around the horizontal axis (radians).
    pub pitch: f32,
    /// Field of view in degrees.
    pub fov: f32,
    /// Aspect ratio (width / height).
    pub aspect: f32,
    /// Near clipping plane.
    pub near: f32,
    /// Far clipping plane.
    pub far: f32,
}

impl Default for ArcballCamera {
    fn default() -> Self {
        Self {
            target: Point3::origin(),
            distance: 3.0,
            yaw: 0.0,
            pitch: 0.3,
            fov: 60.0,
            aspect: 16.0 / 9.0,
            near: 0.01,
            far: 100.0,
        }
    }
}

impl ArcballCamera {
    /// Compute the camera's eye position in world space.
    pub fn eye(&self) -> Point3<f32> {
        let x = self.distance * self.pitch.cos() * self.yaw.sin();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.pitch.cos() * self.yaw.cos();
        Point3::new(self.target.x + x, self.target.y + y, self.target.z + z)
    }

    /// Camera-local right vector (world space).
    pub fn right(&self) -> Vector3<f32> {
        let forward = (self.target - self.eye()).normalize();
        forward.cross(&Vector3::y()).normalize()
    }

    /// Camera-local up vector (world space).
    pub fn up(&self) -> Vector3<f32> {
        let forward = (self.target - self.eye()).normalize();
        let right = forward.cross(&Vector3::y()).normalize();
        right.cross(&forward).normalize()
    }

    /// Compute the view matrix (world → camera space).
    pub fn view_matrix(&self) -> Matrix4<f32> {
        let eye = self.eye();
        let up = Vector3::y();
        Matrix4::look_at_rh(&eye, &self.target, &up)
    }

    /// Compute the projection matrix (camera → clip space).
    pub fn projection_matrix(&self) -> Matrix4<f32> {
        let fov_rad = self.fov.to_radians();
        Matrix4::new_perspective(self.aspect, fov_rad, self.near, self.far)
    }

    /// Combined view-projection matrix.
    pub fn view_projection(&self) -> Matrix4<f32> {
        self.projection_matrix() * self.view_matrix()
    }

    /// Rotate the camera by a delta in yaw and pitch.
    pub fn rotate(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );
    }

    /// Logarithmic zoom — consistent feel at all distances.
    /// Positive `delta` zooms in.
    pub fn zoom(&mut self, delta: f32) {
        let factor = (-delta * 0.15).exp();
        self.distance = (self.distance * factor).clamp(0.05, 100.0);
    }

    /// Pan the camera target in the camera's local XY plane.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let right = self.right();
        let up = self.up();
        self.target += right * dx + up * dy;
    }

    /// Reset the camera to its default state.
    pub fn reset(&mut self) {
        *self = Self {
            aspect: self.aspect,
            ..Self::default()
        };
    }

    /// Project a world-space axis direction into view space.
    /// Returns `[right, up, depth]` where right/up are screen directions
    /// and depth is the toward/away component (positive = toward viewer).
    pub fn project_axis(&self, axis: [f32; 3]) -> [f32; 3] {
        let view = self.view_matrix();
        let v = view.transform_vector(&Vector3::new(axis[0], axis[1], axis[2]));
        // In RH view space: +x = right, +y = up, +z = toward viewer (behind camera).
        [v.x, v.y, v.z]
    }
}
