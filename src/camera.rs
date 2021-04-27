use crate::linear::Vec3;
use crate::Size;

#[derive(Debug, Copy, Clone)]
pub(crate) struct Camera {
    pub position: Vec3,
    pub direction: Vec3,
    pub fov: f32,
}

impl Camera {
    pub fn axis(&self) -> [Vec3; 3] {
        let forward = self.direction.norm();
        let right = Vec3::new(0.0, 1.0, 0.0).cross(forward).norm();
        let up = forward.cross(right);
        [right, up, forward]
    }

    pub fn axis_scaled(&self, size: Size) -> [Vec3; 3] {
        let [right, up, forward] = self.axis();

        let fov_scale = (self.fov / 2.0).tan();
        let w = size.width as f32;
        let h = size.height as f32;
        let forward_ray = (-w / 2.0) * right + (h / 2.0) * up + (h / 2.0) / fov_scale * forward;

        [right, up, forward_ray]
    }
}
