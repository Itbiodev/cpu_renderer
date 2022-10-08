use libm;
use nalgebra::Vector3;
use tgaimage::TGAColor;
use wavefront::{Obj, Vertex};

pub fn vertex(vertex: &Vertex) -> Vector3<f64> {
    Vector3::<f64>::z()
}

pub trait FragmentShader {
    fn fragment(&self, barcoords: &[f64; 3], color: &TGAColor) -> bool;
}

struct GouraudShader<'a> {
    face: &'a Vertex<'a>,
    light_dir: Vector3<f64>,
}
