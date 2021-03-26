use crate::context::GpuContext;

pub(crate) struct Buffer<T> {
    raw: wgpu::Buffer,
    count: usize,
    marker: std::marker::PhantomData<T>,
}

impl<T: bytemuck::Pod> Buffer<T> {
    pub fn new(gpu: &GpuContext, usage: wgpu::BufferUsage, data: &[T]) -> Buffer<T> {
        use wgpu::util::DeviceExt;

        let raw = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage,
            });

        Buffer {
            raw,
            count: data.len(),
            marker: Default::default(),
        }
    }

    pub fn write(&self, gpu: &GpuContext, offset: usize, data: &[T]) {
        gpu.queue
            .write_buffer(&self.raw, offset as u64, bytemuck::cast_slice(data));
    }

    pub fn size(&self) -> Option<wgpu::BufferSize> {
        let bytes = (self.count * std::mem::size_of::<T>()) as u64;
        wgpu::BufferSize::new(bytes)
    }
}

impl<T> std::ops::Deref for Buffer<T> {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}
