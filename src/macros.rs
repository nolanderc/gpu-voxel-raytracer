
macro_rules! count {
    () => { 0usize };
    ($($a:tt, $b:tt),*) => { count!($($a),*) * 2usize };
    ($tt:tt $(, $a:tt, $b:tt)*) => { count!($($a),*) * 2usize + 1usize };
}

macro_rules! sequence_binary {
    ($op:path, [$head:expr, $($values:expr),*]) => {
        $op($head, sequence_binary!($op, [$($values),*]))
    };
    ($op:path, [$head:expr]) => {
        $head
    };
}
