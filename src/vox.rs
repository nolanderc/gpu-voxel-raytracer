// Parser for the VOX format: https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox.txt

use anyhow::{anyhow, Context as _};
use std::collections::HashMap;

pub fn load(path: impl AsRef<std::path::Path>) -> anyhow::Result<Vox> {
    let bytes = std::fs::read(path).context("failed to read file")?;
    parse(&bytes)
}

fn parse(bytes: &[u8]) -> anyhow::Result<Vox> {
    let mut bytes = bytes
        .strip_prefix(b"VOX ")
        .ok_or_else(|| anyhow!("invalid magic number"))?;

    let version = read_i32(&mut bytes)?;
    if version != 150 {
        return Err(anyhow!("unsupported VOX-format: version {}", version));
    }

    let main = peek_chunk(*b"MAIN", &mut bytes)?.ok_or_else(|| anyhow!("missing MAIN chunk"))?;
    parse_main_chunk(main)
}

fn parse_main_chunk(main: Chunk) -> anyhow::Result<Vox> {
    let mut bytes = main.content;

    let mut model_count = 1;
    if let Some(mut pack) = peek_chunk(*b"PACK", &mut bytes)? {
        model_count = read_u32(&mut pack.content)?;
    }

    let mut models = Vec::with_capacity(model_count as usize);

    for _ in 0..model_count {
        let mut size = expect_chunk(*b"SIZE", &mut bytes)?;
        let mut xyzi = expect_chunk(*b"XYZI", &mut bytes)?;
        models.push(Model {
            size: parse_size(&mut size.content)?,
            voxels: parse_xyzi(&mut xyzi.content)?,
        });
    }

    let mut palette = Box::new(DEFAULT_PALETTE);
    let mut materials = HashMap::with_capacity(256);

    while !bytes.is_empty() {
        let mut chunk = read_chunk(&mut bytes)?;
        match &chunk.id {
            b"RGBA" => {
                for i in 1..256 {
                    let rgba = read_u32(&mut chunk.content)?;
                    palette[i] = rgba;
                }
            }
            b"MATL" => {
                let id = read_u32(&mut chunk.content)?;
                let material = parse_material(&mut chunk.content)?;
                materials.insert(id, material);
            }
            _ => eprintln!("unknown chunk {}", bytes_to_string(&chunk.id)),
        }
    }

    Ok(Vox {
        models,
        palette,
        materials,
    })
}

fn parse_material(bytes: &mut &[u8]) -> anyhow::Result<Material> {
    let entries = read_dict(bytes)?;

    let mut material = Material {
        kind: MaterialKind::Diffuse,
        flux: 0.0,
    };

    for (key, value) in entries {
        match key {
            b"_type" => match value {
                b"_emit" => material.kind = MaterialKind::Emit,
                b"_diffuse" => material.kind = MaterialKind::Diffuse,
                typ => {
                    return Err(anyhow!(
                            "unsupported material type: {}",
                            bytes_to_string(typ)
                    ))
                }
            },
            b"_flux" => {
                material.flux = parse_bytes(value)
                    .context("failed to parse value of material key `_flux`")?
            }
            _ => {}
        }
    }

    Ok(material)
}

const DEFAULT_PALETTE: [u32; 256] = [
    0x00000000, 0xffffffff, 0xffccffff, 0xff99ffff, 0xff66ffff, 0xff33ffff, 0xff00ffff, 0xffffccff,
    0xffccccff, 0xff99ccff, 0xff66ccff, 0xff33ccff, 0xff00ccff, 0xffff99ff, 0xffcc99ff, 0xff9999ff,
    0xff6699ff, 0xff3399ff, 0xff0099ff, 0xffff66ff, 0xffcc66ff, 0xff9966ff, 0xff6666ff, 0xff3366ff,
    0xff0066ff, 0xffff33ff, 0xffcc33ff, 0xff9933ff, 0xff6633ff, 0xff3333ff, 0xff0033ff, 0xffff00ff,
    0xffcc00ff, 0xff9900ff, 0xff6600ff, 0xff3300ff, 0xff0000ff, 0xffffffcc, 0xffccffcc, 0xff99ffcc,
    0xff66ffcc, 0xff33ffcc, 0xff00ffcc, 0xffffcccc, 0xffcccccc, 0xff99cccc, 0xff66cccc, 0xff33cccc,
    0xff00cccc, 0xffff99cc, 0xffcc99cc, 0xff9999cc, 0xff6699cc, 0xff3399cc, 0xff0099cc, 0xffff66cc,
    0xffcc66cc, 0xff9966cc, 0xff6666cc, 0xff3366cc, 0xff0066cc, 0xffff33cc, 0xffcc33cc, 0xff9933cc,
    0xff6633cc, 0xff3333cc, 0xff0033cc, 0xffff00cc, 0xffcc00cc, 0xff9900cc, 0xff6600cc, 0xff3300cc,
    0xff0000cc, 0xffffff99, 0xffccff99, 0xff99ff99, 0xff66ff99, 0xff33ff99, 0xff00ff99, 0xffffcc99,
    0xffcccc99, 0xff99cc99, 0xff66cc99, 0xff33cc99, 0xff00cc99, 0xffff9999, 0xffcc9999, 0xff999999,
    0xff669999, 0xff339999, 0xff009999, 0xffff6699, 0xffcc6699, 0xff996699, 0xff666699, 0xff336699,
    0xff006699, 0xffff3399, 0xffcc3399, 0xff993399, 0xff663399, 0xff333399, 0xff003399, 0xffff0099,
    0xffcc0099, 0xff990099, 0xff660099, 0xff330099, 0xff000099, 0xffffff66, 0xffccff66, 0xff99ff66,
    0xff66ff66, 0xff33ff66, 0xff00ff66, 0xffffcc66, 0xffcccc66, 0xff99cc66, 0xff66cc66, 0xff33cc66,
    0xff00cc66, 0xffff9966, 0xffcc9966, 0xff999966, 0xff669966, 0xff339966, 0xff009966, 0xffff6666,
    0xffcc6666, 0xff996666, 0xff666666, 0xff336666, 0xff006666, 0xffff3366, 0xffcc3366, 0xff993366,
    0xff663366, 0xff333366, 0xff003366, 0xffff0066, 0xffcc0066, 0xff990066, 0xff660066, 0xff330066,
    0xff000066, 0xffffff33, 0xffccff33, 0xff99ff33, 0xff66ff33, 0xff33ff33, 0xff00ff33, 0xffffcc33,
    0xffcccc33, 0xff99cc33, 0xff66cc33, 0xff33cc33, 0xff00cc33, 0xffff9933, 0xffcc9933, 0xff999933,
    0xff669933, 0xff339933, 0xff009933, 0xffff6633, 0xffcc6633, 0xff996633, 0xff666633, 0xff336633,
    0xff006633, 0xffff3333, 0xffcc3333, 0xff993333, 0xff663333, 0xff333333, 0xff003333, 0xffff0033,
    0xffcc0033, 0xff990033, 0xff660033, 0xff330033, 0xff000033, 0xffffff00, 0xffccff00, 0xff99ff00,
    0xff66ff00, 0xff33ff00, 0xff00ff00, 0xffffcc00, 0xffcccc00, 0xff99cc00, 0xff66cc00, 0xff33cc00,
    0xff00cc00, 0xffff9900, 0xffcc9900, 0xff999900, 0xff669900, 0xff339900, 0xff009900, 0xffff6600,
    0xffcc6600, 0xff996600, 0xff666600, 0xff336600, 0xff006600, 0xffff3300, 0xffcc3300, 0xff993300,
    0xff663300, 0xff333300, 0xff003300, 0xffff0000, 0xffcc0000, 0xff990000, 0xff660000, 0xff330000,
    0xff0000ee, 0xff0000dd, 0xff0000bb, 0xff0000aa, 0xff000088, 0xff000077, 0xff000055, 0xff000044,
    0xff000022, 0xff000011, 0xff00ee00, 0xff00dd00, 0xff00bb00, 0xff00aa00, 0xff008800, 0xff007700,
    0xff005500, 0xff004400, 0xff002200, 0xff001100, 0xffee0000, 0xffdd0000, 0xffbb0000, 0xffaa0000,
    0xff880000, 0xff770000, 0xff550000, 0xff440000, 0xff220000, 0xff110000, 0xffeeeeee, 0xffdddddd,
    0xffbbbbbb, 0xffaaaaaa, 0xff888888, 0xff777777, 0xff555555, 0xff444444, 0xff222222, 0xff111111,
];

struct Chunk<'a> {
    id: [u8; 4],
    content: &'a [u8],
}

#[derive(Debug, Clone)]
pub struct Vox {
    pub models: Vec<Model>,
    pub palette: Box<[u32; 256]>,
    pub materials: HashMap<u32, Material>,
}

#[derive(Debug, Copy, Clone)]
pub struct Material {
    pub kind: MaterialKind,
    pub flux: f32,
}

#[derive(Debug, Copy, Clone)]
pub enum MaterialKind {
    Diffuse,
    Emit,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub size: Size,
    pub voxels: Vec<Voxel>,
}

#[derive(Debug, Copy, Clone)]
pub struct Size {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Debug, Copy, Clone)]
pub struct Voxel {
    pub x: u8,
    pub y: u8,
    pub z: u8,
    pub color: u8,
}

impl Vox {
    pub fn get_color_rgb(&self, index: u8) -> [u8; 3] {
        let color = self.palette[index as usize];
        let r = color & 0xff;
        let g = (color >> 8) & 0xff;
        let b = (color >> 16) & 0xff;
        [r as u8, g as u8, b as u8]
    }
}

fn parse_size(bytes: &mut &[u8]) -> anyhow::Result<Size> {
    Ok(Size {
        x: read_u32(bytes)?,
        y: read_u32(bytes)?,
        z: read_u32(bytes)?,
    })
}

fn parse_xyzi(bytes: &mut &[u8]) -> anyhow::Result<Vec<Voxel>> {
    let count = read_u32(bytes)? as usize;
    let mut voxels = Vec::with_capacity(count);

    for _ in 0..count {
        voxels.push(Voxel {
            x: read_u8(bytes)?,
            y: read_u8(bytes)?,
            z: read_u8(bytes)?,
            color: read_u8(bytes)?,
        });
    }

    Ok(voxels)
}

fn bytes_to_string<'a>(id: &'a [u8]) -> std::borrow::Cow<'a, str> {
    String::from_utf8_lossy(id)
}

fn expect_chunk<'a>(name: [u8; 4], bytes: &mut &'a [u8]) -> anyhow::Result<Chunk<'a>> {
    let chunk = read_chunk(bytes)?;
    if chunk.id == name {
        Ok(chunk)
    } else {
        Err(anyhow!(
            "expected chunk {}, found chunk {}",
            bytes_to_string(&name),
            bytes_to_string(&chunk.id)
        ))
    }
}

fn peek_chunk<'a>(name: [u8; 4], bytes: &mut &'a [u8]) -> anyhow::Result<Option<Chunk<'a>>> {
    if !bytes.starts_with(&name) {
        Ok(None)
    } else {
        read_chunk(bytes).map(Some)
    }
}

fn read_chunk<'a>(bytes: &mut &'a [u8]) -> anyhow::Result<Chunk<'a>> {
    let id = read::<4>(bytes)?;
    let content_size = read_u32(bytes)?;
    let children_size = read_u32(bytes)?;
    let chunk_size = content_size + children_size;
    let content = split(chunk_size as usize, bytes)?;

    Ok(Chunk { id, content })
}

fn split<'a>(count: usize, bytes: &mut &'a [u8]) -> anyhow::Result<&'a [u8]> {
    if bytes.len() < count {
        return Err(anyhow!("unexpected end of file"));
    }

    let (before, after) = bytes.split_at(count);
    *bytes = after;
    Ok(before)
}

fn read<const N: usize>(bytes: &mut &[u8]) -> anyhow::Result<[u8; N]> {
    if bytes.len() < N {
        return Err(anyhow!("unexpected end of file"));
    }

    let mut buffer = [0; N];
    buffer.copy_from_slice(&bytes[..N]);
    *bytes = &bytes[N..];
    Ok(buffer)
}

fn read_i32(bytes: &mut &[u8]) -> anyhow::Result<i32> {
    let buffer = read(bytes)?;
    Ok(i32::from_le_bytes(buffer))
}

fn read_u32(bytes: &mut &[u8]) -> anyhow::Result<u32> {
    let buffer = read(bytes)?;
    Ok(u32::from_le_bytes(buffer))
}

fn read_u8(bytes: &mut &[u8]) -> anyhow::Result<u8> {
    Ok(read::<1>(bytes)?[0])
}

fn read_str<'a>(bytes: &mut &'a [u8]) -> anyhow::Result<&'a [u8]> {
    let length = read_u32(bytes)?;
    split(length as usize, bytes)
}

fn read_dict<'a>(bytes: &mut &'a [u8]) -> anyhow::Result<Vec<(&'a [u8], &'a [u8])>> {
    let count = read_u32(bytes)?;
    let mut entries = Vec::with_capacity(count as usize);

    for _ in 0..count {
        let key = read_str(bytes)?;
        let value = read_str(bytes)?;
        entries.push((key, value));
    }

    Ok(entries)
}

fn parse_bytes<T>(bytes: &[u8]) -> anyhow::Result<T>
where
    T: std::str::FromStr,
    T::Err: Into<anyhow::Error>,
{
    let text = std::str::from_utf8(bytes).context("value did not contain valid UTF-8")?;
    text.parse::<T>().map_err(Into::into)
}
