use nalgebra::{Vector3, Vector4};
use wavefront::Vertex;
use tgaimage::TGAColor;
pub trait Shader {
    fn vertex_shader(&mut self, vertex: &Vertex) -> Vector4<f64>;
    fn fragment_shader(self, bar_coords: &Vector3<f64>, color: &mut TGAColor) -> bool;
}

pub struct GouradShader {
    pub varying_intensity: Option<Vector3<f64>>
}

/*
impl Shader for GouradShader {
    fn vertex_shader(&mut self, vertex: &Vertex) -> Vector4<f64> {
       let normal = vertex.normal();
       self.varying_intensity = Some(f64::max(0., normal.dot(&light_dir)));
       let vec_vertex = vertex.position().to_vec();
       viewport*projection*modelview*vec_vertex
    }
    fn fragment_shader(self, bar_coords: &Vector3<f64>, color: &mut TGAColor) -> bool {
        if let Some(intensity) = self.varying_intensity {
            color.r = 255*intensity as u8;
            color.g = 255*intensity as u8;
            color.b = 255*intensity as u8;    
        }
        false
    }
}
*/
