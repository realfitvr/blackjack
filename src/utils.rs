use crate::prelude::*;

pub trait IteratorUtils: Iterator {
    fn collect_svec(self) -> SVec<Self::Item>
    where
        Self: Sized,
    {
        self.collect()
    }
}

impl<T: ?Sized> IteratorUtils for T where T: Iterator {}

pub trait SliceUtils<T> {
    /// Same as .iter().copied(), but doesn't trigger rustfmt line breaks
    fn iter_cpy(&self) -> std::iter::Copied<std::slice::Iter<'_, T>>;
}

impl<T: Copy> SliceUtils<T> for [T] {
    fn iter_cpy(&self) -> std::iter::Copied<std::slice::Iter<'_, T>> {
        self.iter().copied()
    }
}
use std::{fs::File, io::{BufReader, copy, stdout}, path::PathBuf, sync::{atomic::{AtomicI32, Ordering}, Arc}};
use exr::{self, image::read::levels::ReadLargestLevel, image::read::specific_channels::ReadSpecificChannel, image::write::channels::WritableChannels, math::Vec2, meta::MetaData, prelude::{ReadChannels, ReadLayers}};
use na::{Point2, Point, Matrix3x4, RowVector4};
use serde_json;
use serde::{Deserialize};
extern crate nalgebra as na;
use anyhow::{Context, Result};
use std::path::Path;


///       |-----|
///       |  2  |
/// |-----|-----|-----|-----|
/// |  1  |  4  |  0  |  5  |  v
/// |_____|_____|_____|_____|  |__ u
///       |     |
///       |  3  |
///       -------
///
/// 0 = +x
/// 1 = -x
/// 2 = +y
/// 3 = -y
/// 4 = +z
/// 5 = -z
///
fn convert_xyz_to_cube_uv(x: f32, y: f32, z: f32) -> (usize, f32, f32)
{
    let abs_x = x.abs();
    let abs_y = y.abs();
    let abs_z = z.abs();

    let is_x_positive = x > 0.0;
    let is_y_positive = y > 0.0;
    let is_z_positive = z > 0.0;

    let mut max_axis: f32 = 0.0;
    let mut uc: f32 = 0.0;
    let mut vc: f32 = 0.0;

    let mut index: usize = 0;

    // POSITIVE X
    if is_x_positive && abs_x >= abs_y && abs_x >= abs_z {
        // u (0 to 1) goes from +z to -z
        // v (0 to 1) goes from +y to -y
        max_axis = abs_x;
        uc = -z;
        vc = -y;
        index = 0;
    }
    // NEGATIVE X
    if !is_x_positive && abs_x >= abs_y && abs_x >= abs_z {
        // u (0 to 1) goes from -z to +z
        // v (0 to 1) goes from +y to -y
        max_axis = abs_x;
        uc = z;
        vc = -y;
        index = 1;
    }
    // POSITIVE Y
    if is_y_positive && abs_y >= abs_x && abs_y >= abs_z {
        // u (0 to 1) goes from -x to +x
        // v (0 to 1) goes from -z to +z
        max_axis = abs_y;
        uc = x;
        vc = z;
        index = 2;
    }
    // NEGATIVE Y
    if !is_y_positive && abs_y >= abs_x && abs_y >= abs_z {
        // u (0 to 1) goes from -x to +x
        // v (0 to 1) goes from +z to -z
        max_axis = abs_y;
        uc = x;
        vc = -z;
        index = 3;
    }
    // POSITIVE Z
    if is_z_positive && abs_z >= abs_x && abs_z >= abs_y {
        // u (0 to 1) goes from -x to +x
        // v (0 to 1) goes from +y to -y
        max_axis = abs_z;
        uc = x;
        vc = -y;
        index = 4;
    }
    // NEGATIVE Z
    if !is_z_positive && abs_z >= abs_x && abs_z >= abs_y {
        // u (0 to 1) goes from +x to -x
        // v (0 to 1) goes from +y to -y
        max_axis = abs_z;
        uc = -x;
        vc = -y;
        index = 5;
    }

    // Convert range from -1 to 1 to 0 to 1
    let u = 0.5f32 * (uc / max_axis + 1.0f32);
    let v = 0.5f32 * (vc / max_axis + 1.0f32);

    (index, u, v)
}

#[derive(Deserialize)]
//#[serde(rename_all = "snake_case")]
pub struct ProjectorInfo {
    pub n_faces: i32,
    pub n_vertices: i32,
    pub camera_height: f32
}

pub struct ProjectorContext {
    projector: ProjectorInfo,
    depth_dir_path: PathBuf,
    //out_dir_path: PathBuf,
    vertices: Vec<f32>,
    triangle_indices: Vec<u32>,

    //frame_transforms: Vec<Matrix3x4<f32>>,
}

pub fn project_frame_from_depth(ctx: &ProjectorContext, file_path: PathBuf) -> Result<Mesh> {

    use exr::prelude::*;

    let default_depth = 64.0f32;

    //let file = File::open(file_path).unwrap();
    let depth_img = exr::image::read::read()
        .no_deep_data() // (currently required)
        .largest_resolution_level() // or `all_resolution_levels()`
        .specific_channels()
        .optional("ViewLayer.Depth.left.Z", default_depth)
        .optional("ViewLayer.Depth.right.Z", default_depth)
        .required("ViewLayer.Depth.front.Z")
        .optional("ViewLayer.Depth.back.Z", default_depth)
        .optional("ViewLayer.Depth.up.Z", default_depth)
        .optional("ViewLayer.Depth.down.Z", default_depth)
        //.all_channels() // or `rgba_channels(constructor, setter)`
        .collect_pixels(
            |resolution, (_left, _right, _front, _back, _up, _down)| {

                //println!("Down Z channel type = {:?}", down_z.sample_type);
                //println!("image contains stereoscopic luma channel? {}", y_right_channel.is_some());
                //println!("the type of luma samples is {:?}", y_channel.sample_type);

                vec![vec![[default_depth; 6]; resolution.width()]; resolution.height()]
            },

            // all samples will be converted to f32 (you can also use the enum `Sample` instead of `f32` here to retain the original data type from the file)
            |vec, position: Vec2<usize>, (left, right, front, back, up, down): (f32, f32, f32, f32, f32, f32)| {
                vec[position.y()][position.x()] = [right, left, up, down, front, back];
            }
        )
        .all_layers() // or `first_valid_layer()`
        .all_attributes() // (currently required)
        //.on_progress(|progress| println!("progress: {:.1}", progress*100.0)) // optional
        .from_file(file_path.clone())?;

    let meta_data = MetaData::read_from_file(
        file_path,
        false // do not throw an error for invalid or missing attributes, skipping them instead
    ).unwrap();

    for (layer_index, image_layer) in meta_data.headers.iter().enumerate() {
        println!(
            "custom meta data of layer #{}:\n{:#?}",
            layer_index, image_layer.own_attributes
        );
    }

    let layer = depth_img.layer_data.first().unwrap();

    //for layer in depth_img.layer_data {
        println!("layer name = {}", layer.attributes.layer_name.as_ref().unwrap_or(&Text::default()));
        println!("view name = {}", layer.attributes.view_name.as_ref().unwrap_or(&Text::default()));
        println!("size = {:?}", layer.size);

        //let channels = layer.channel_data.infer_channel_list();
        let channels = &layer.channel_data.channels;
        println!("channels: {:?}", channels);

        let pixels = layer.channel_data.pixels[512][512];
        println!("pixel = {:?}", pixels);
        //for channel in layer.channel_data {
        //    println!("> channel name = {}", channel.attributes.name);
       // }
    //}

    let pos = &ctx.vertices;
    let x_res = layer.size.x();
    let y_res = layer.size.y();
    assert_eq!(x_res, y_res); // mapping z values to camera distance assumes square cube maps

    let mut depths = vec![];

    let mut projected_verts = vec![];

    // XXX: the 'depth' in the cube map depth buffers is the distance along the
    // forward axis of the camera, _not_ the distance as the crow flies from the
    // camera (which is what we need)
    //
    // We calculate the distance in pixel units and scale that into meters
    //
    // Since we can assume as have a 90 degree FOV for each cube face
    // our focal length in pixel units is simply the resolution / 2...
    //
    let fx_90 = x_res as f64 / 2.0;
    let fy_90 = y_res as f64 / 2.0;

    for i in 0..ctx.projector.n_vertices {
        let i = (i * 3) as usize;

        let mut point = na::Vector3::new(pos[i], pos[i + 1], pos[i + 2]); // ref: rust-analyzer issue #8654
        point.normalize_mut();

        let (face, u, v) = convert_xyz_to_cube_uv(point.x, point.y, point.z);


        let buffer_x = ((u * x_res as f32) as usize).clamp(0, x_res - 1);
        let x_center = (buffer_x as f64 + 0.5) - (x_res as f64 / 2.0);
        let buffer_y = ((v * y_res as f32) as usize).clamp(0, y_res - 1);
        let y_center = (buffer_y  as f64 + 0.5) - (y_res as f64 / 2.0);

        let pixel = layer.channel_data.pixels[buffer_y][buffer_x];
        let depth = pixel[face];
        //let depth = 10.0f32 + (face as f32 * 10.0f32);
        //let depth = 10.0f32;

        let tanx = x_center / fx_90;
        let tany = y_center / fy_90;

        let x_meters = tanx * depth as f64;
        let y_meters = tany * depth as f64;

        let dist = (x_meters * x_meters + y_meters * y_meters + depth as f64 * depth as f64).sqrt();

        //println!("buf x = {buffer_x}, x_res = {x_res}, x center = {x_center}, fx_90 = {fx_90}, tanx = {tanx}, tanx 0 = {tanx_0}, x_meters_0 = {x_meters_0}, dist = {dist}");
        //println!("buf y = {buffer_y}, y_res = {y_res}, y center = {y_center}, fy_90 = {fy_90}, y_meters_0 = {y_meters_0}, dist = {dist}");

        //let pix_dist_x = (u*u) + fx_squared;
        //let pix_dist_y = (v*v) + fy_squared;


        // The u and v coordinates

        //let pt0 = Point2::new(0.5f32, 0.5f32);
        //let pt1 = Point2::new(u, v);
        //let dist = na::distance(&pt0, &pt1);
        //let depth = 10.0f32 + v * 30.0;
        //println!("f={face}, x={f_x}, y={f_y} => depth = {depth}");


        depths.push(dist.min(65.0));
        let projected: na::Vector3<f32> = point.scale(depth);
        projected_verts.push(projected);

        //depths.push(1.0f32);
    }

    Ok(Mesh {
        position: projected_verts,
        triangle_mask: None
    })
}

pub fn clamp_point_distance(ctx: &ProjectorContext, mesh: &Mesh, max: f32) -> Result<Mesh> {

    let mut projected_verts = Vec::<na::Vector3<f32>>::with_capacity(mesh.position.len());

    for v in &mesh.position {
        let depth = v.magnitude();
        let clamped = v.normalize().scale(depth.min(max));
        projected_verts.push(clamped);
    }

    Ok(Mesh {
        position: projected_verts,
        triangle_mask: None
    })
}

/*
pub fn write_projector_depth_map(ctx: &ProjectorContext, frame_no: i32, verts: &Vec<na::Vector3<f32>>) -> Result<()> {
    let mut depths = vec![];
    for v in verts {
        let depth = v.magnitude();
        depths.push(depth.min(65.0));
    }

    let file_path = ctx.out_dir_path.join(format!("{frame_no:04}_ProjectorDepthMap.json"));
    serde_json::to_writer(&File::create(file_path)?, &depths)?;

    let tri_mask = vec![1; (ctx.projector.n_faces) as usize];
    let file_path = ctx.out_dir_path.join(format!("{frame_no:04}_ProjectorTriangleMask.json"));
    serde_json::to_writer(&File::create(file_path)?, &tri_mask)?;

    Ok(())
}
*/

pub struct Mesh {
    //indices: Vec<i32>,
    position: Vec<na::Vector3<f32>>,
    triangle_mask: Option<Vec<bool>>
}

pub fn load_projector_frame(projector_file: PathBuf, projector_indices_file: PathBuf, projector_positions_file: PathBuf, depth_frames_dir: PathBuf, frame_no: i32) -> Result<(Vec<Vec3>, Vec<[u32; 3]>)> {

    //let data_dir_path = Path::new(data_dir);

    let file = File::open(projector_file)?;
    let reader = BufReader::new(file);
    // Read the JSON contents of the file as an instance of `User`.
    let projector: ProjectorInfo = serde_json::from_reader(reader).unwrap();
    let n_faces = projector.n_faces;
    let n_vertices = projector.n_vertices;
    let cam_height = projector.camera_height;
    println!("Num faces = {n_faces}");
    println!("Num vertices = {n_vertices}");
    println!("Camera height = {cam_height}");

    //let file_path = data_dir_path.join(format!("{fn_prefix}ProjectorTriangleIndices.json"));
    let file = File::open(projector_indices_file)?;
    let reader = BufReader::new(file);
    // Read the JSON contents of the file as an instance of `User`.
    let indices: Vec<u32> = serde_json::from_reader(reader).unwrap();

    //let file_path = data_dir_path.join(format!("{fn_prefix}ProjectorTrianglesPos.json"));
    let file = File::open(projector_positions_file)?;
    let reader = BufReader::new(file);
    let pos: Vec<f32> = serde_json::from_reader(reader).unwrap();

    //let depth_dir_path = Path::new(depth_dir);
    //let out_dir_path = Path::new(out_dir);
    let mut polygons: Vec<[u32; 3]> = vec![];
    for i in 0..projector.n_faces as usize {
        polygons.push([indices[i*3], indices[i*3 + 1], indices[i*3 + 2]]);
    }

    let ctx = Arc::new(ProjectorContext {
        depth_dir_path: depth_frames_dir.to_path_buf(),
        //out_dir_path: out_dir_path.to_path_buf(),
        projector,
        vertices: pos,
        triangle_indices: indices,
        //frame_transforms: transforms
        //next_frame: AtomicI32::new(0)
    });

    let file_path = ctx.depth_dir_path.join(format!("{frame_no:04}.exr"));
    if !file_path.exists() {
        return Err(anyhow!("Couldn't find {file_path:?}"));
    }
    let mesh = project_frame_from_depth(&ctx, file_path)?;
    let mesh = clamp_point_distance(&ctx, &mesh, 65.)?;

    //write_projector_depth_map(&ctx, f, &mesh.position)?;

    // Convert into a layout that can _then_ be converted into a halfedge representation...


    let mut positions = vec![];
    for i in 0..mesh.position.len() {
        positions.push(Vec3::new(mesh.position[i].x, mesh.position[i].y, mesh.position[i].z));
    }

    Ok((positions, polygons))
}