use crate::linear::Vec3;
use crate::{Coord, Ray};

#[derive(Debug)]
pub(crate) struct Octree<T> {
    root: OctreeSlot<T>,
    depth: u8,
}

type OctreeSlot<T> = Option<OctreeNode<T>>;

#[derive(Debug)]
enum OctreeNode<T> {
    Leaf(T),
    Branch(Box<[OctreeSlot<T>; 8]>),
}

#[derive(Debug, Copy, Clone)]
struct RayExt {
    origin: Vec3,
    direction: Vec3,
    inv_direction: Vec3,
}

impl From<Ray> for RayExt {
    fn from(ray: Ray) -> Self {
        RayExt {
            origin: ray.origin,
            direction: ray.direction,
            inv_direction: ray.direction.map(f32::recip),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct RayHit<'a, T> {
    pub value: &'a T,
    pub normal: Vec3,
    pub time: f32,
}

#[derive(PartialEq)]
struct Ordered(f32);

impl Eq for Ordered {}

impl PartialOrd for Ordered {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for Ordered {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Less)
    }
}

#[derive(Copy, Clone)]
struct OctantIntersection {
    count: u8,
    octants: [u8; 4],
    planes: [u8; 4],
    times: [f32; 5],
}

pub const MAX_DEPTH: usize = 15;

impl<T: std::fmt::Debug> Octree<T> {
    pub fn with_depth(depth: u8) -> Self {
        assert!(depth as usize <= MAX_DEPTH);
        Octree { root: None, depth }
    }

    pub fn insert(&mut self, coord: Coord, value: T) {
        let mut center = self.root_center();
        let mut current = &mut self.root;
        let mut size = 1 << self.depth;
        while size > 1 {
            let (dx, dy, dz) = Self::octant_offset(center, coord);
            let octant = Self::octant_from_offset(dx, dy, dz);
            current = Self::get_or_insert_octant(current, octant);

            size /= 2;

            center = Self::octant_center(octant, center, size);
        }

        // at max depth, insert the value
        *current = Some(OctreeNode::Leaf(value));
    }

    fn get_or_insert_octant(slot: &mut OctreeSlot<T>, octant: usize) -> &mut OctreeSlot<T> {
        match slot {
            Some(OctreeNode::Leaf(_old)) => panic!("insert into leaf node"),
            Some(OctreeNode::Branch(children)) => &mut children[octant],
            None => {
                *slot = Some(OctreeNode::empty_branch());
                match slot {
                    Some(OctreeNode::Branch(children)) => &mut children[octant],
                    _ => unreachable!(),
                }
            }
        }
    }

    pub fn cast_ray(&self, ray: Ray) -> Option<RayHit<T>> {
        let center_coord = self.root_center();
        let center = Vec3 {
            x: center_coord.x as f32,
            y: center_coord.y as f32,
            z: center_coord.z as f32,
        };

        let ray = RayExt::from(ray);
        let size = (1 << self.depth) as f32;

        // Did we intersect with the node?
        let (entry, exit, entry_plane) = ray_cube_intersection(ray, center, size)?;

        match self.root.as_ref()? {
            OctreeNode::Leaf(value) => Some(RayHit {
                value,
                normal: Self::plane_normal(entry_plane, ray.direction),
                time: entry,
            }),
            OctreeNode::Branch(children) => Self::cast_ray_children_iterative(
                children,
                &ray,
                center,
                size,
                entry,
                exit,
                entry_plane,
            ),
        }
    }

    fn cast_ray_children_iterative<'a>(
        children: &'a [OctreeSlot<T>; 8],
        ray: &RayExt,
        center: Vec3,
        size: f32,
        entry: f32,
        exit: f32,
        entry_plane: usize,
    ) -> Option<RayHit<'a, T>> {
        let initial_intersections =
            Self::octant_intersections(ray, center, entry, exit, entry_plane);
        let mut stack = [None; MAX_DEPTH];
        stack[0] = Some((0usize, center, size, children, initial_intersections));
        let mut stack_top = 0;

        while let Some((step, center, size, children, intersections)) = stack[stack_top].as_mut() {
            let i = *step;

            if i >= intersections.count as usize {
                if stack_top == 0 {
                    break;
                } else {
                    stack_top -= 1;
                    continue;
                }
            }

            *step += 1;

            let octant = intersections.octants[i] as usize;
            match &children[octant] {
                None => {}
                Some(OctreeNode::Branch(children)) => {
                    let plane = intersections.planes[i] as usize;
                    let entry = intersections.times[i];
                    let exit = intersections.times[i + 1];

                    let child_size = *size / 2.0;
                    let child_center = Self::child_octant_center(*center, child_size, octant);

                    let child_intersections =
                        Self::octant_intersections(ray, child_center, entry, exit, plane);
                    stack_top += 1;
                    stack[stack_top] =
                        Some((0, child_center, child_size, children, child_intersections));
                }
                Some(OctreeNode::Leaf(value)) => {
                    return Some(RayHit {
                        value,
                        normal: Self::plane_normal(intersections.planes[i] as usize, ray.direction),
                        time: intersections.times[i],
                    });
                }
            }
        }

        None
    }

    fn cast_ray_children<'a>(
        children: &'a [OctreeSlot<T>; 8],
        ray: &RayExt,
        center: Vec3,
        size: f32,
        entry: f32,
        exit: f32,
        entry_plane: usize,
    ) -> Option<RayHit<'a, T>> {
        let intersections = Self::octant_intersections(ray, center, entry, exit, entry_plane);

        for i in 0..intersections.count as usize {
            let octant = intersections.octants[i] as usize;
            match &children[octant] {
                None => {}
                Some(OctreeNode::Branch(children)) => {
                    let plane = intersections.planes[i] as usize;
                    let entry = intersections.times[i];
                    let exit = intersections.times[i + 1];

                    let child_size = size / 2.0;
                    let child_center = Self::child_octant_center(center, child_size, octant);
                    if let Some(hit) = Self::cast_ray_children(
                        children,
                        ray,
                        child_center,
                        child_size,
                        entry,
                        exit,
                        plane,
                    ) {
                        return Some(hit);
                    }
                }
                Some(OctreeNode::Leaf(value)) => {
                    return Some(RayHit {
                        value,
                        normal: Self::plane_normal(intersections.planes[i] as usize, ray.direction),
                        time: intersections.times[i],
                    });
                }
            }
        }

        None
    }

    fn octant_intersections(
        ray: &RayExt,
        center: Vec3,
        entry: f32,
        exit: f32,
        entry_plane: usize,
    ) -> OctantIntersection {
        let delta = center - ray.origin;
        let plane_entry = delta.elementwise_product(ray.inv_direction);
        let mut order = [0, 1, 2];

        // Insertion sort by entry time:
        if plane_entry[order[0]] > plane_entry[order[1]] {
            order.swap(0, 1);
        }
        if plane_entry[order[1]] > plane_entry[order[2]] {
            order.swap(1, 2);
            if plane_entry[order[0]] > plane_entry[order[1]] {
                order.swap(0, 1);
            }
        }

        // Index of the current plane
        let mut i = 0;

        // Skip planes while behind the ray
        while i < 3 && plane_entry[order[i]] < 0.0 {
            i += 1
        }

        // Get the octant of they ray, assuming it is inside the node
        let mut octant =
            Self::octant_from_offset(delta.x < 0.0, delta.y < 0.0, delta.z < 0.0) as u8;

        // Skip planes while before intersecting the node
        while i < 3 && plane_entry[order[i]] < entry {
            // Advance to the next octant (since we assumed we were inside)
            octant ^= 4 >> order[i];
            i += 1
        }

        let mut octants = [0; 4];
        let mut planes = [0; 4];
        let mut times = [0.0; 5];

        // store the first intersection (the entry)
        let mut octant_count = 1;
        octants[0] = octant;
        planes[0] = entry_plane as u8;
        times[0] = entry;

        // While we haven't exited the node...
        while i < 3 && plane_entry[order[i]] < exit {
            // advance to next octant
            octant ^= 4 >> order[i];
            i += 1;

            // Visit octant
            octants[octant_count] = octant;
            planes[octant_count] = order[i - 1] as u8;
            times[octant_count] = plane_entry[order[i - 1]];
            octant_count += 1;
        }

        // Store the time of the exit
        times[octant_count] = exit;

        OctantIntersection {
            count: octant_count as u8,
            octants,
            planes,
            times,
        }
    }

    fn plane_normal(plane: usize, ray_direction: Vec3) -> Vec3 {
        let mut normal = Vec3::zero();
        normal[plane] = -f32::copysign(1.0, ray_direction[plane]);
        normal
    }

    fn child_octant_center(parent_center: Vec3, child_size: f32, octant: usize) -> Vec3 {
        let bool_sign = |b| if b { -0.5 } else { 0.5 };

        let add_or_sub = Vec3::new(
            bool_sign(octant & 4 == 0),
            bool_sign(octant & 2 == 0),
            bool_sign(octant & 1 == 0),
        );

        parent_center + child_size * add_or_sub
    }

    fn root_center(&self) -> Coord {
        let size = 1 << self.depth;
        Coord {
            x: size / 2,
            y: size / 2,
            z: size / 2,
        }
    }

    fn octant_offset(center: Coord, coord: Coord) -> (bool, bool, bool) {
        let dx = coord.x >= center.x;
        let dy = coord.y >= center.y;
        let dz = coord.z >= center.z;
        (dx, dy, dz)
    }

    fn octant_from_offset(dx: bool, dy: bool, dz: bool) -> usize {
        4 * dx as usize + 2 * dy as usize + dz as usize
    }

    fn octant_center(octant: usize, parent_center: Coord, size: u16) -> Coord {
        fn add_else_sub(cond: bool, value: &mut u16, amount: u16) {
            if cond {
                *value += amount;
            } else {
                *value -= amount
            }
        }

        let mut center = parent_center;
        add_else_sub(octant & 0x4 != 0, &mut center.x, size / 2);
        add_else_sub(octant & 0x2 != 0, &mut center.y, size / 2);
        add_else_sub(octant & 0x1 != 0, &mut center.z, size / 2);
        center
    }
}

impl<T> OctreeNode<T> {
    const SLOT_EMPTY: OctreeSlot<T> = None;

    pub fn empty_branch() -> Self {
        OctreeNode::Branch(Box::new([Self::SLOT_EMPTY; 8]))
    }
}

fn ray_cube_intersection(ray: RayExt, center: Vec3, size: f32) -> Option<(f32, f32, usize)> {
    let half_size = size / 2.0;
    let delta = center - ray.origin;

    let entry_exit = |plane: usize| {
        use std::f32::INFINITY;
        let dir = ray.direction[plane];
        let dist = delta[plane];
        if dir == 0.0 {
            if dist.abs() <= half_size {
                (-INFINITY, INFINITY)
            } else {
                (INFINITY, -INFINITY)
            }
        } else {
            let inv_dir = ray.inv_direction[plane];
            let entry = (dist - half_size.copysign(inv_dir)) * inv_dir;
            let exit = (dist + half_size.copysign(inv_dir)) * inv_dir;
            (entry, exit)
        }
    };

    let (entry_x, exit_x) = entry_exit(0);
    let (entry_y, exit_y) = entry_exit(1);
    let (entry_z, exit_z) = entry_exit(2);

    let entries = [entry_x, entry_y, entry_z];
    let mut max_entry = 0;
    if entries[1] > entries[max_entry] {
        max_entry = 1;
    }
    if entries[2] > entries[max_entry] {
        max_entry = 2;
    }

    let min_exit = exit_x.min(exit_y).min(exit_z);

    if min_exit > 0.0 && entries[max_entry] < min_exit {
        Some((entries[max_entry], min_exit, max_entry))
    } else {
        None
    }
}
