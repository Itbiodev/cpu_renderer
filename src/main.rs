use rand::Rng;
use std::cmp::{max, min};
use std::mem;
use std::ops::{Add, BitXor, Div, Mul, Sub};
use tgaimage::*;
use wavefront::Obj;

const WIDTH: usize = 800;
const HEIGHT: usize = 800;

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

#[derive(Debug, Clone)]
struct Point<T> {
    x: T,
    y: T,
}

impl<T> Point<T>
where
    T: Copy,
{
    fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

/*impl<T> Add for Point<T>
where
    T: Add + Copy,
    T::Output: Add + Copy,
{
    type Output = Point<T::Output>;

    fn add(self, rhs: Point<T>) -> Point<T::Output> {
        Point {
            x: self.x + rhs.y,
            y: self.y + rhs.y,
        }
    }
}*/

type TGAPoint = Point<usize>;
fn line(p0: &Point<usize>, p1: &Point<usize>, image: &mut TGAImage, color: &TGAColor) {
    let mut x0 = p0.x;
    let mut y0 = p0.y;
    let mut x1 = p1.x;
    let mut y1 = p1.y;

    let mut steep = false;

    if isize::abs(x1 as isize - x0 as isize) < isize::abs(y1 as isize - y0 as isize) {
        mem::swap(&mut x0, &mut y0);
        mem::swap(&mut x1, &mut y1);
        steep = true;
    }

    if x0 > x1 {
        mem::swap(&mut x0, &mut x1);
        mem::swap(&mut y0, &mut y1);
    }

    let dx = x1 as isize - x0 as isize;
    let dy = y1 as isize - y0 as isize;
    let derror2 = isize::abs(dy) * 2;
    let mut error2 = 0;
    let mut y = y0 as isize;

    for x in x0..x1 {
        if steep {
            image.set(y as usize, x, &color);
        } else {
            image.set(x, y as usize, &color);
        }
        error2 += derror2;
        if error2 > dx {
            let eps: isize = if y1 > y0 { 1 } else { -1 };
            y = y + eps;
            error2 -= dx * 2;
        }
    }

    /*for t in (0..100).map(|s| s as f32 * 0.01) {
        // let t = (t as f32) * 0.01;
        let x = x0 + (t * ((x1 - x0) as f32)) as usize;
        let y = y1 + (t * ((y1 - y0) as f32)) as usize;
        image.set(x, y, &color);
    }*/
}

fn draw_triangle(
    p0: &Point<usize>,
    p1: &Point<usize>,
    p2: &Point<usize>,
    mut image: &mut TGAImage,
    color: &TGAColor,
) {
    line(p0, p1, &mut image, color);
    line(p1, p2, &mut image, color);
    line(p2, p0, &mut image, color);
}

type Triangle<T> = Vec<Point<T>>; //[Point<T>; 3];

type BoxCorners<T> = [Point<T>; 2];
// TODO: Simplify this one
fn bounding_box(triangle: &Triangle<usize>) -> BoxCorners<usize> {
    // let lower_corner = Point { x: 0, y: 0 };
    let upper_corner = Point {
        x: WIDTH - 1,
        y: HEIGHT - 1,
    };

    let mut bboxmax = Point { x: 0, y: 0 };
    let mut bboxmin = upper_corner.clone();

    for vertex in triangle {
        bboxmin.x = max(0, min(bboxmin.x, vertex.x));
        bboxmin.y = max(0, min(bboxmin.y, vertex.y));
        // println!("Vertex: {:?}, bboxmin: {:?}", vertex, bboxmin);
        bboxmax.x = min(upper_corner.x, max(bboxmax.x, vertex.x));
        bboxmax.y = min(upper_corner.y, max(bboxmax.y, vertex.y));
        // println!("Vertex: {:?}, bboxmax: {:?}", vertex, bboxmax);
    }
    [bboxmin, bboxmax]
}

fn barycentric_coords(triangle: &Triangle<usize>, point: &Point<usize>) -> [f32; 3] {
    let (a, b, c) = (&triangle[0], &triangle[1], &triangle[2]);

    let (x, y) = (point.x as f32, point.y as f32);

    let (x1, y1) = (a.x as f32, a.y as f32);
    let (x2, y2) = (b.x as f32, b.y as f32);
    let (x3, y3) = (c.x as f32, c.y as f32);

    let det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2);
    if f32::abs(det) < 1. {
        return [-1., 1., 1.];
    }
    let u = (x3 * y1 - x1 * y3) * 1. + (y3 - y1) * x + (x1 - x3) * y; // / det;
    let v = (x1 * y2 - x2 * y1) * 1. + (y1 - y2) * x + (x2 - x1) * y; // / det;
    [1. - (u + v) / det, u / det, v / det]
}

fn draw_triangle_barycentric(triangle: &Triangle<usize>, image: &mut TGAImage, color: &TGAColor) {
    let [bboxmin, bboxmax] = bounding_box(&triangle);
    for pixel_x in bboxmin.x..=bboxmax.x {
        for pixel_y in bboxmin.y..=bboxmax.y {
            let [w, u, v] = barycentric_coords(&triangle, &Point::new(pixel_x, pixel_y));
            if w < 0. || u < 0. || v < 0. {
                continue;
            }
            image.set(pixel_x, pixel_y, &color);
        }
    }
}

// TODO: Improve ASAP
type Triangle3d = Vec<Vec3f>;
fn barycentric3d(triangle: &Triangle3d, point: &Vec3f) -> [f32; 3] {
    let ab = triangle[0].clone() - triangle[1].clone();
    let ac = triangle[0].clone() - triangle[2].clone();
    let ap = triangle[0].clone() - point.clone();
    let u = ab.clone() ^ ac.clone();
    let k = Vec3f::new(0., 0., 1.);
    let m_b = (ap.clone() ^ ac.clone()) * k.clone();
    let m_c = (ab.clone() ^ ap.clone()) * k.clone();
    let den = u.clone() * k.clone();
    if f32::abs(den) > 0.001 {
        [1. - (m_b + m_c) / den, m_b / den, m_c / den]
    } else {
        [-1., 1., 1.]
    }
}

fn barycentric(triangle: &Triangle3d, point: &Vec3f) -> [f32; 3] {
    let mut s = [[0. as f32;3];2]; 
    
    s[1][0] = triangle[2].y - triangle[0].y; 
    s[1][1] = triangle[1].y - triangle[0].y;
    s[1][2] = triangle[0].y - point.y;

    s[0][0] = triangle[2].x - triangle[0].x; 
    s[0][1] = triangle[1].x - triangle[0].x;
    s[0][2] = triangle[0].x - point.x;
    
    let u = Vec3f::new(s[0][0], s[0][1], s[0][2]);
    let v = Vec3f::new(s[1][0], s[1][1], s[1][2]);

    let w = u^v;
    if f32::abs(w.z) > 0.01 {
        [ 1.-(w.x+w.y)/w.z, w.y/w.z, w.x/w.z ]
    } else {
        [-1., 1., 1.]
    }
}

fn bounding_box3d(triangle: &Triangle3d) -> [[f32; 2]; 2] {
    let mut bboxmin = [f32::MAX, f32::MAX];
    let mut bboxmax = [-f32::MAX, -f32::MAX];
    let clamp = [WIDTH as f32 - 1., HEIGHT as f32 - 1.];
    for vertex in triangle {
        bboxmin[0] = f32::max(0., f32::min(bboxmin[0], vertex.x));
        bboxmin[1] = f32::max(0., f32::min(bboxmin[1], vertex.y));
        bboxmax[0] = f32::min(clamp[0], f32::max(bboxmax[0], vertex.x));
        bboxmax[1] = f32::min(clamp[1], f32::max(bboxmax[1], vertex.y));
    }
    [bboxmin, bboxmax]
}

fn draw_triangle3d(
    triangle: &Triangle3d,
    z_buffer: &mut [f32],
    image: &mut TGAImage,
    color: &TGAColor,
) {
    let [bboxmin, bboxmax] = bounding_box3d(&triangle);
    for pixel_x in bboxmin[0] as usize..=bboxmax[0] as usize {
        for pixel_y in bboxmin[1] as usize..=bboxmax[1] as usize {
            let mut point = Vec3f::new(pixel_x as f32, pixel_y as f32, 0.);
            let [w, u, v] = barycentric3d(&triangle, &point);
            if w < 0. || u < 0. || v < 0. {
                continue;
            }
            for (i, coord) in [w, u, v].into_iter().enumerate() {
                point.z += coord * triangle[i].z;
            }
            if  z_buffer[pixel_x + pixel_y*WIDTH] < point.z {
                z_buffer[pixel_x + pixel_y*WIDTH] = point.z;
                image.set(pixel_x, pixel_y, &color);
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Vec3f {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3f {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn norm(&self) -> f32 {
        f32::sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    }

    fn scale_by(&self, a: f32) -> Self {
        Self {
            x: a * self.x,
            y: a * self.y,
            z: a * self.z,
        }
    }

    fn normalize(&self) -> Self {
        let norm = self.norm();
        Self {
            x: self.x / norm,
            y: self.y / norm,
            z: self.z / norm,
        }
    }
}

impl BitXor for Vec3f {
    type Output = Self;
    fn bitxor(self, rhs: Vec3f) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: -(self.x * rhs.z - self.z * rhs.x),
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }
}

impl Mul for Vec3f {
    type Output = f32;
    fn mul(self, rhs: Vec3f) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
}

impl Sub for Vec3f {
    type Output = Vec3f;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Add for Vec3f {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

fn main() {
    let mut image = TGAImage::new(WIDTH, HEIGHT, 3);
    // let red = TGAColor::rgb(255, 0, 0);
    let white = TGAColor::rgb(255, 255, 255);

    let model = Obj::from_file("./african_head.obj").unwrap();
    let mut z_buffer = [-f32::MAX; WIDTH * HEIGHT];
    for triangle in model.triangles() {
        let light_dir = Vec3f::new(0., 0., -1.);

        // https://users.rust-lang.org/t/what-is-a-more-efficient-way-to-clear-a-vec/40190
        let mut screen_triangle = Vec::with_capacity(3); // What's the correct way
        let mut model_triangle = Vec::with_capacity(3); // What's the correct way

        for vertex in triangle {
            let x = (vertex.position()[0] + 1.) * (WIDTH as f32) / 2.;
            let y = (vertex.position()[1] + 1.) * (HEIGHT as f32) / 2.;
            screen_triangle.push(Vec3f::new(x , y, 0.));
            model_triangle.push(Vec3f::new(
                vertex.position()[0],
                vertex.position()[1],
                vertex.position()[2],
            ));
        }

        let n = (model_triangle[2].clone() - model_triangle[0].clone())
            ^ (model_triangle[1].clone() - model_triangle[0].clone());
        let n = n.normalize();
        let intensity = n * light_dir;
        if intensity > 0. {
            draw_triangle3d(
                &screen_triangle,
                &mut z_buffer,
                &mut image,
                &TGAColor::rgb(
                    (intensity * 255.) as u8,
                    (intensity * 255.) as u8,
                    (intensity * 255.) as u8,
                ),
            );
        }
    }
    /*
    let p0 = Point::new(10, 10);
    let p1 = Point::new(100, 30);
    let p2 = Point::new(190, 160);
    draw_triangle_barycentric(&[p0, p1, p2], &mut image, &white);
    */
    /*
    let p0 = Point::new(100, 100);
    let p1 = Point::new(500, 200);
    let p2 = Point::new(200, 500);
    draw_triangle(&p0, &p1, &p2, &mut image, &white);
    let square = bounding_box(&[p0, p1, p2]);
    //
    line(
        &square[0],
        &Point::new(square[0].x, square[1].y),
        &mut image,
        &white,
    ); // |
    line(
        &Point::new(square[1].x, square[0].y),
        &square[1],
        &mut image,
        &white,
    ); // |
    line(
        &square[0],
        &Point::new(square[1].x, square[0].y),
        &mut image,
        &white,
    ); // _
    line(
        &square[1],
        &Point::new(square[0].x, square[1].y),
        &mut image,
        &white,
    ); // -
    */
    image.flip_vertically();
    image.write_tga_file("./test.tga", true);
}
