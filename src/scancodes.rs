use std::convert::TryFrom;

macro_rules! scancodes {
    (
        $vis:vis enum $ident:ident {
            $($key:ident = $value:expr),+
            $(,)?
        }
    ) => {
        #[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
        #[repr(u32)]
        $vis enum $ident {
            $(
                $key = $value,
            )+
        }

        impl TryFrom<u32> for $ident {
            type Error = anyhow::Error;

            fn try_from(code: u32) -> Result<$ident, Self::Error> {
                match code {
                    $(
                        $value => Ok($ident::$key),
                    )+
                    _ => Err(anyhow::anyhow!("unknown scancode: {}", code))
                }
            }
        }
    }
}

#[cfg(target_os = "linux")]
scancodes! {
    pub enum Scancode {
        Q = 16,
        W = 17,
        E = 18,
        R = 19,
        T = 20,
        Y = 21,
        I = 22,
        O = 23,
        P = 24,

        A = 30,
        S = 31,
        D = 32,
        F = 33,
        G = 34,
        H = 35,
        J = 36,
        K = 37,
        L = 38,


        Z = 44,
        X = 45,
        C = 46,
        V = 47,
        B = 48,
        N = 49,
        M = 50,

        Tab = 15,
        LeftControl = 29,
        LeftShift = 42,
        Escape = 58,
        Back = 14,
        Enter = 28,
    }
}

