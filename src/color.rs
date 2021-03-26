
#[derive(Copy, Clone, PartialEq)]
pub(crate) struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl From<[u8; 3]> for Color {
    fn from([r, g, b]: [u8; 3]) -> Self {
        Color {
            r,
            g,
            b,
        }
    }
}

impl Color {
    pub const BLACK: Color = Color::gray(0);

    pub const fn gray(gray: u8) -> Color {
        Color::new(gray, gray, gray)
    }

    pub const fn new(r: u8, g: u8, b: u8) -> Color {
        Color { r, g, b }
    }
}

impl std::fmt::Debug for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Color { r, g, b } = *self;
        write!(
            f,
            "\x1b[38;2;{};{};{}m██\x1b[0m Color({}, {}, {})",
            r, g, b, r, g, b
        )
    }
}

