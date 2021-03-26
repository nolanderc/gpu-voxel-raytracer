use crate::linear::Vec3;
use crate::{Ray, Size};

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

    pub fn cast_rays(
        &self,
        size: Size,
    ) -> impl rayon::iter::IndexedParallelIterator<Item = (Ray, (u32, u32))> {
        use rayon::prelude::*;

        let fov_scale = (self.fov / 2.0).tan();

        let forward = self.direction.norm();
        let right = Vec3::new(0.0, 1.0, 0.0).cross(forward).norm();
        let up = forward.cross(right);

        let w = size.width as f32;
        let h = size.height as f32;
        let forward_ray = (-w / 2.0) * right + (h / 2.0) * up + (h / 2.0) / fov_scale * forward;

        let origin = self.position;

        (0..size.width * size.height)
            .into_par_iter()
            .map(move |index| {
                let row = index / size.width;
                let col = index % size.width;

                let screen_dir = col as f32 * right - row as f32 * up + forward_ray;
                let coord = (col, row);
                let ray = Ray {
                    origin,
                    direction: screen_dir.norm(),
                };
                (ray, coord)
            })
    }

    fn cast_ray_single(&self, x: u32, y: u32, size: Size) -> Ray {
        let fov_scale = (self.fov / 2.0).tan();

        let forward = self.direction.norm();
        let right = Vec3::new(0.0, 1.0, 0.0).cross(forward).norm();
        let up = forward.cross(right);

        let w = size.width as f32;
        let h = size.height as f32;
        let forward_ray = (-w / 2.0) * right + (h / 2.0) * up + (h / 2.0) / fov_scale * forward;

        let screen_dir = x as f32 * right - y as f32 * up + forward_ray;
        Ray {
            origin: self.position,
            direction: screen_dir.norm(),
        }
    }
}
