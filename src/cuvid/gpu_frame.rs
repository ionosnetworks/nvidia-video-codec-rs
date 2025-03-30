use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use super::{ffi, CudaResult};

pub use ffi::cuvid::{CUdeviceptr, CUvideodecoder};

pub struct GpuFrame {
    pub width: u32,
    pub height: u32,
    pub ptr: CUdeviceptr,
    pub pitch: u32,
    pub timestamp: i64,
    pub(crate) frame_in_use: Arc<AtomicU64>,
    pub(crate) idx: i32,
    pub(crate) decoder: CUvideodecoder,
}

impl Drop for GpuFrame {
    fn drop(&mut self) {
        unsafe {
            if !ffi::cuvid::cuvidUnmapVideoFrame64(self.decoder, self.ptr).ok() {
                tracing::error!("Failed to unmap current frame.");
            }

            if !ffi::cuda::cuCtxPopCurrent_v2(std::ptr::null_mut()).ok() {
                tracing::error!("Failed to pop current context.");
            }

            let v = !(1 << self.idx);
            self.frame_in_use
                .fetch_and(v, std::sync::atomic::Ordering::SeqCst);
        }
    }
}
