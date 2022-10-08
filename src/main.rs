mod geometry;
mod shaders;

use libm::fabs;
use nalgebra::*;
use std::cmp::{max, min};
use std::mem;
use tgaimage::*;
use wavefront::{Obj, Vertex};

const WIDTH: usize = 800;
const HEIGHT: usize = 800;
const DEPTH: usize = 255;

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

fn draw_triangle_with_lines(
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

type Triangle3d = Vec<Vector3<f64>>;
fn barycentric3d(triangle: &Triangle3d, point: &Vector3<f64>) -> [f64; 3] {
    let ab = &triangle[0] - &triangle[1];
    let ac = &triangle[0] - &triangle[2];
    let ap = &triangle[0] - point;
    let u = ab.cross(&ac);
    let k = Vector3::<f64>::new(0., 0., 1.);
    let m_b = ap.cross(&ac).dot(&k);
    let m_c = ab.cross(&ap).dot(&k);
    let den = u.dot(&k);
    if libm::fabs(den) > 0.01 {
        [1. - (m_b + m_c) / den, m_b / den, m_c / den]
    } else {
        [-1., 1., 1.]
    }
}

fn barycentric(triangle: &Triangle3d, point: &Vector3<f64>) -> [f64; 3] {
    let mut s = [[0. as f64; 3]; 2];

    s[1][0] = triangle[2].y - triangle[0].y;
    s[1][1] = triangle[1].y - triangle[0].y;
    s[1][2] = triangle[0].y - point.y;

    s[0][0] = triangle[2].x - triangle[0].x;
    s[0][1] = triangle[1].x - triangle[0].x;
    s[0][2] = triangle[0].x - point.x;

    let u = Vector3::new(s[0][0], s[0][1], s[0][2]);
    let v = Vector3::new(s[1][0], s[1][1], s[1][2]);

    let w = u.cross(&v);
    if libm::fabs(w.z) > 0.01 {
        [1. - (w.x + w.y) / w.z, w.y / w.z, w.x / w.z]
    } else {
        [-1., 1., 1.]
    }
}

fn bounding_box3d(triangle: &Triangle3d) -> [Vector2<usize>; 2] {
    let mut bboxmin = [f64::MAX, f64::MAX];
    let mut bboxmax = [-f64::MAX, -f64::MAX];
    let clamp = [WIDTH as f64 - 1., HEIGHT as f64 - 1.];
    for vertex in triangle {
        bboxmin[0] = f64::max(0., f64::min(bboxmin[0], vertex.x));
        bboxmin[1] = f64::max(0., f64::min(bboxmin[1], vertex.y));
        bboxmax[0] = f64::min(clamp[0], f64::max(bboxmax[0], vertex.x));
        bboxmax[1] = f64::min(clamp[1], f64::max(bboxmax[1], vertex.y));
    }

    [
        Vector2::new(bboxmin[0] as usize, bboxmin[1] as usize),
        Vector2::new(bboxmax[0] as usize, bboxmax[1] as usize),
    ]
}

fn draw_triangle3d(
    triangle: &Triangle3d,
    z_buffer: &mut [f64],
    image: &mut TGAImage,
    color: &TGAColor,
) {
    let [bboxmin, bboxmax] = bounding_box3d(&triangle);
    for pixel_x in bboxmin.x..=bboxmax.x {
        for pixel_y in bboxmin.y..=bboxmax.y {
            let mut point = Vector3::new(pixel_x as f64, pixel_y as f64, 0.);
            let [w, u, v] = barycentric3d(&triangle, &point);
            if w < 0. || u < 0. || v < 0. {
                continue;
            }
            for (i, coord) in [w, u, v].into_iter().enumerate() {
                point.z += coord * triangle[i].z;
            }
            if z_buffer[pixel_x + pixel_y * WIDTH] < point.z {
                z_buffer[pixel_x + pixel_y * WIDTH] = point.z;
                image.set(pixel_x, pixel_y, &color);
            }
        }
    }
}

fn draw_triangle3d_with_shader(
    triangle: &Triangle3d,
    z_buffer: &mut [f64],
    image: &mut TGAImage,
    color: &TGAColor,
    vertex_shader: &fn(&Vertex),
    fragment_shader: &fn(&Vector3<f64>,&TGAColor)
) {
    let [bboxmin, bboxmax] = bounding_box3d(&triangle);
    for pixel_x in bboxmin.x..=bboxmax.x {
        for pixel_y in bboxmin.y..=bboxmax.y {
            let mut point = Vector3::new(pixel_x as f64, pixel_y as f64, 0.);
            let [w, u, v] = barycentric3d(&triangle, &point);
            if w < 0. || u < 0. || v < 0. {
                continue;
            }
            for (i, coord) in [w, u, v].into_iter().enumerate() {
                point.z += coord * triangle[i].z;
            }
            if z_buffer[pixel_x + pixel_y * WIDTH] < point.z {
                z_buffer[pixel_x + pixel_y * WIDTH] = point.z;
                image.set(pixel_x, pixel_y, &color);
            }
        }
    }
}

fn viewport(x: f64, y: f64, w: f64, h: f64, d: f64) -> Matrix4<f64> {
    let viewport = Matrix4::<f64>::from_rows(&[
        RowVector4::new(w as f64 / 2., 0., 0., (x + w as f64) / 2.),
        RowVector4::new(0., h as f64 / 2., 0., (y + h as f64) / 2.),
        RowVector4::new(0., 0., d as f64 / 2., (d as f64) / 2.),
        RowVector4::new(0., 0., 0., 1.),
    ]);
    viewport
}

fn projection(camera: &Vector3<f64>) -> Matrix4<f64> {
    let mut proj = Matrix4::<f64>::identity();
    proj[(3, 2)] = 1. / camera.norm(); //-1. / camera.z
    proj
}

fn lookat(eye: &Vector3<f64>, center: &Vector3<f64>, up: &Vector3<f64>) -> Matrix4<f64> {
    let z = (eye - center).normalize();
    let x = up.cross(&z).normalize();
    let y = z.cross(&x).normalize();
    let mut res = Matrix4::<f64>::identity();
    for i in 0..3 {
        res[(0, i)] = x[i];
        res[(1, i)] = y[i];
        res[(2, i)] = z[i];
        res[(i, 3)] = -center[i];
    }
    res
}

fn render_model(mut image: &mut TGAImage) {
    let model = Obj::from_file("./african_head.obj").unwrap();
    let mut z_buffer = [-f64::MAX; WIDTH * HEIGHT];
    let light_dir = Vector3::new(-1., -1., -1.).normalize();
    let eye = Vector3::new(10., 0., 3.);
    let center = Vector3::new(0., 0., 0.);
    let camera = eye - center;
    let up = Vector3::new(0., 1., 0.);
    let viewport = viewport(
        WIDTH as f64 / 8.,
        HEIGHT as f64 / 8.,
        WIDTH as f64 * 3. / 4.,
        HEIGHT as f64 * 3. / 4.,
        DEPTH as f64,
    );
    let projection = projection(&camera);
    let modelview = lookat(&eye, &center, &up);
    println!("{}", projection);
    // TODO: Improve
    for triangle in model.triangles() {
        let mut screen_triangle = Vec::with_capacity(3);
        let mut model_triangle = Vec::with_capacity(3);
        for vertex in triangle {
            // let x = (vertex.position()[0] + 1.) * (WIDTH as f32) / 2.;
            // let y = (vertex.position()[1] + 1.) * (HEIGHT as f32) / 2.;
            let new_triangle = Vector4::<f64>::new(
                vertex.position()[0] as f64,
                vertex.position()[1] as f64,
                vertex.position()[2] as f64,
                1.,
            );
            let perspective = viewport * projection * modelview * new_triangle;
            // screen_triangle.push(Vector3::<f64>::new(x as f64, y as f64, 0.));
            screen_triangle.push(Vector3::<f64>::new(
                perspective.x,
                perspective.y,
                perspective.z,
            ));
            model_triangle.push(Vector3::<f64>::new(
                vertex.position()[0] as f64,
                vertex.position()[1] as f64,
                vertex.position()[2] as f64,
            ));
        }
        let n = (&model_triangle[2] - &model_triangle[0])
            .cross(&(model_triangle[1] - model_triangle[0]));
        let n = n.normalize();
        let intensity = n.dot(&light_dir);
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
}

fn main() {
    let mut image = TGAImage::new(WIDTH, HEIGHT, 3);
    let diffuse = TGAImage::from_tga_file("./african_head_diffuse.tga");

    /*let model = Obj::from_file("./cube.obj").unwrap();

    for (i, vertex) in model.triangles().enumerate() {
        println!("{}:{:?}", i, vertex);
    }*/
    /*
        let e1 = Vector3::new(1., 0., 0.);
        let e2 = Vector3::new(0., 1., 0.);
        let e3 = Vector3::new(0., 0., 1.);
        println!("{:?}", &e3 * 2.);
        println!("Difference: {:?}", &e2 - &e3);
        println!("{:?}", &e2.cross(&e1));
        println!("{:?}", Matrix4::<f64>::identity().try_inverse().unwrap());
        println!("{:?}", Matrix4::<f64>::identity().transpose());
    */
    render_model(&mut image);
    image.flip_vertically();
    image.write_tga_file("./test.tga", true);
}
