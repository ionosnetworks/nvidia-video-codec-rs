use super::{ffi, CudaResult};

mod chroma;
mod codec;
mod decoder;
mod encoder;
mod gpu_frame;
mod surface;

pub use self::chroma::VideoChromaFormat;
pub use self::codec::Codec;
pub use self::decoder::Decoder;
pub use self::encoder::Encoder;
pub use self::gpu_frame::GpuFrame;
pub use self::surface::VideoSurfaceFormat;

#[derive(Copy, Clone, Debug)]
pub enum Bitrate {
    CQP,
    CBR(u32),
    VBR(u32),
}
