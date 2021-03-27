use crate::context::GpuContext;
use anyhow::{anyhow, Context as _};
use std::path::Path;

#[instrument(skip(gpu))]
pub(crate) fn create_shader_module(
    gpu: &GpuContext,
    source_path: impl AsRef<Path> + std::fmt::Debug,
) -> anyhow::Result<wgpu::ShaderModule> {
    let source_path = source_path.as_ref();
    let spirv = load_shader_binary(source_path)
        .with_context(|| anyhow!("failed to load shader `{}`", source_path.display()))?;

    Ok(gpu
        .device
        .create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some(&source_path.to_string_lossy()),
            source: wgpu::ShaderSource::SpirV(spirv.into()),
            flags: wgpu::ShaderFlags::default(),
        }))
}

fn load_shader_binary(source_path: &Path) -> anyhow::Result<Vec<u32>> {
    if !source_path.exists() {
        return Err(anyhow!("file does not exist"));
    }

    let mut file_name = source_path
        .file_name()
        .ok_or_else(|| anyhow!("shader path must point to file"))?
        .to_os_string();
    file_name.push(".spv");
    let binary_path = source_path.with_file_name(&file_name);
    let binary_path = binary_path.as_path();

    if should_rebuild(source_path, binary_path) {
        info!(shader = %source_path.display(), "rebuilding shader");
        compile_shader(source_path, binary_path).context("failed to rebuild shader from source")?;
    }

    let bytes: Vec<u8> = std::fs::read(binary_path)?;
    let spirv: Vec<u32> = bytemuck::pod_collect_to_vec(&bytes);
    Ok(spirv)
}

fn should_rebuild(source_path: &Path, binary_path: &Path) -> bool {
    let last_modification =
        |path: &Path| path.metadata().ok().and_then(|meta| meta.modified().ok());

    match (
        last_modification(source_path),
        last_modification(binary_path),
    ) {
        // skip rebuild if the source hasn't changed since the last time the binary was built
        (Some(source_time), Some(binary_time)) if source_time <= binary_time => false,
        _ => true,
    }
}

fn compile_shader(source_path: &Path, binary_path: &Path) -> anyhow::Result<()> {
    use std::ffi::OsStr;

    let args = [
        // Set the input file
        source_path.as_os_str(),
        // Set the output file
        OsStr::new("-o"),
        binary_path.as_os_str(),
        // Enable Vulkan semantics, outputs SPIR-V
        OsStr::new("-V"),
    ];

    let output = std::process::Command::new("glslangValidator")
        .args(&args)
        .output()
        .context("failed to invoke GLSL compiler `glslangValidator`")?;

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(anyhow!("{}", stdout));
    }

    Ok(())
}
