use forward_ref_generic::forward_ref_binop;
use libm;
use std::ops::{Add, BitXor, Div, Mul, Sub};

pub trait Sqrt {
    fn square(self) -> f64;
}

impl Sqrt for f32 {
    fn square(self) -> f64 {
        f64::sqrt(self as f64)
    }
}
impl Sqrt for f64 {
    fn square(self) -> f64 {
        f64::sqrt(self as f64)
    }
}
impl Sqrt for usize {
    fn square(self) -> f64 {
        f64::sqrt(self as f64)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Vec3<T: Copy> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Copy + Add<T, Output = T> + Mul<T, Output = T> + Sqrt + Div> Vec3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).square()
    }

    pub fn scale_by(&self, a: T) -> Self {
        Self {
            x: a * self.x,
            y: a * self.y,
            z: a * self.z,
        }
    }
}

impl<T> Add for Vec3<T>
where
    T: Copy + Add<T, Output = T>,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

forward_ref_binop! {
    [T]
    impl Add for Vec3<T>
    where T: Copy + Add<T,Output = T>
}

impl<T> Mul for Vec3<T>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T>,
{
    type Output = T;
    fn mul(self, rhs: Self) -> T {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
}

forward_ref_binop! {
    [T]
    impl Mul for Vec3<T>
    where T: Copy + Add<T, Output = T> + Mul<T, Output = T>
}

impl<T> Sub for Vec3<T>
where
    T: Copy + Sub<T, Output = T>,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

forward_ref_binop! {
    [T]
    impl Sub for Vec3<T>
    where T: Copy + Sub<T, Output = T>
}

impl<T> BitXor for Vec3<T>
where
    T: Copy + Sub<T, Output = T> + Mul<T, Output = T>,
{
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }
}

impl<T> BitXor<Vec3<T>> for &Vec3<T>
where
    T: Copy + Sub<T, Output = T> + Mul<T, Output = T>,
{
    type Output = <Vec3<T> as BitXor<Vec3<T>>>::Output;
    fn bitxor(self, rhs: Vec3<T>) -> Self::Output {
        <Vec3<T>>::bitxor(*self, rhs)
    }
}
impl<T> BitXor<&Vec3<T>> for Vec3<T>
where
    T: Copy + Sub<T, Output = T> + Mul<T, Output = T>,
{
    type Output = <Vec3<T> as BitXor<Vec3<T>>>::Output;
    fn bitxor(self, rhs: &Vec3<T>) -> Self::Output {
        <Vec3<T>>::bitxor(self, *rhs)
    }
}
impl<T> BitXor<&Vec3<T>> for &Vec3<T>
where
    T: Copy + Sub<T, Output = T> + Mul<T, Output = T>,
{
    type Output = <Vec3<T> as BitXor<Vec3<T>>>::Output;
    fn bitxor(self, rhs: &Vec3<T>) -> Self::Output {
        <Vec3<T>>::bitxor(*self, *rhs)
    }
}

struct Interval {
    start: f32,
    end: f32,
}

impl Interval {
    fn new(a: f32, b: f32) -> Self {
        Self { start: a, end: b }
    }

    fn transform(&self, x: f32, J: &Interval) -> Result<f32, String> {
        if !(x > self.end || x < self.start) {
            let y = (J.end - J.start) * (x - self.start) / (self.end - self.start) + J.start;
            Ok(y)
        } else {
            Err(format!(
                "point {}, is not in the interval ({},{})",
                x, self.start, self.end
            ))
        }
    }
}

#[derive(Debug)]
pub struct Matrix<const N: usize, const M: usize> {
    data: [[f64; M]; N],
    pub rows: usize,
    pub cols: usize,
}

impl<const N: usize, const M: usize> Matrix<N, M> {
    pub fn identity(dim: usize) -> Self {
        if M != N {
            println!("not a square");
        }
        let mut coefficients = [[0. as f64; M]; N];
        for i in 0..N {
            //'column: for j in 0..M {
            //if i == j {
            coefficients[i][i] = 1.;
            //    break 'column
            //}
            //}
        }
        Self {
            data: coefficients,
            rows: N,
            cols: M,
        }
    }
}
