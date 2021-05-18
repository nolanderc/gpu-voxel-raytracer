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

#[cfg(target_os = "macos")]
scancodes! {
    pub enum Scancode {
        Q = 12,
        W = 13,
        E = 14,
        R = 15,
        T = 17,
        Y = 16,
        U = 32,
        I = 34,
        O = 31,
        P = 35,

        A = 0,
        S = 1,
        D = 2,
        F = 3,
        G = 5,
        H = 4,
        J = 38,
        K = 40,
        L = 37,

        Z = 6,
        X = 7,
        C = 8,
        V = 9,
        B = 11,
        N = 45,
        M = 46,

        Tab = 48,
        Escape = 53,
        LeftShift = 56,
        LeftControl = 59,
        Space = 49,
        Return = 36,
        Back = 51,

        Left = 123,
        Right = 124,
        Up = 126,
        Down = 125,

        Key1 = 18,
        Key2 = 19,
        Key3 = 20,
        Key4 = 21,
        Key5 = 23,
        Key6 = 22,
        Key7 = 26,
        Key8 = 28,
        Key9 = 25,
        Key0 = 29,
    }
}

#[cfg(target_os = "windows")]
scancodes! {
    pub enum Scancode {
        Q = 16,
        W = 17,
        E = 18,
        R = 19,
        T = 20,
        Y = 21,
        U = 22,
        I = 23,
        O = 24,
        P = 25,

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

        Escape = 1,

        LeftShift = 42,
        RightShift = 54,

        LeftControl = 29,
        RightControl = 57373,

        Space = 57,
        Tab = 15,

        Return = 28,
        Back = 14,

        Left = 57419,
        Up = 57416,
        Down = 57424,
        Right = 57421,
        Home = 57415,
        End = 57423,

        Key1 = 2,
        Key2 = 3,
        Key3 = 4,
        Key4 = 5,
        Key5 = 6,
        Key6 = 7,
        Key7 = 8,
        Key8 = 9,
        Key9 = 10,
        Key0 = 11,
    }
}
