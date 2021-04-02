use super::GpuContext;

pub(crate) fn copy_entire_texture_to_texture(
    source: &wgpu::Texture,
    destination: &wgpu::Texture,
    size: crate::Size,
    encoder: &mut wgpu::CommandEncoder,
) {
    encoder.copy_texture_to_texture(
        wgpu::TextureCopyView {
            texture: source,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        wgpu::TextureCopyView {
            texture: destination,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth: 1,
        },
    );
}

pub(crate) fn create_texture(
    size: crate::Size,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsage,
    gpu: &GpuContext,
) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
    })
}

pub fn view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}
