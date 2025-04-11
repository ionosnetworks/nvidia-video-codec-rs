use std::marker::PhantomData;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Condvar, Mutex};

use super::{ffi, CudaResult};

use ffi::cuda::CUcontext;
pub use ffi::cuvid::{CUdeviceptr, CUvideodecoder};

pub struct GpuFrame {
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub timestamp: i64,
    pub has_concealed_error: Option<bool>,
    pub(crate) ptr: CUdeviceptr,
    pub(crate) frame_in_use: Arc<AtomicU64>,
    pub(crate) idx: i32,
    pub(crate) decoder: CUvideodecoder,
    pub(crate) context: CUcontext,
    pub(crate) frames_in_flight: Arc<(Mutex<usize>, Condvar)>,
}

impl GpuFrame {
    pub fn ptr<'a>(&'a self) -> GpuFramePtr<'a> {
        unsafe {
            if !ffi::cuda::cuCtxPushCurrent_v2(self.context).ok() {
                tracing::error!("Failed to push current context.");
            }
        }
        GpuFramePtr {
            ptr: self.ptr,
            _lifetime: Default::default(),
            _unsend: Default::default(),
        }
    }
}

impl Drop for GpuFrame {
    fn drop(&mut self) {
        unsafe {
            if !ffi::cuda::cuCtxPushCurrent_v2(self.context).ok() {
                tracing::error!("Failed to push current context.");
            }
            if !ffi::cuvid::cuvidUnmapVideoFrame64(self.decoder, self.ptr).ok() {
                tracing::error!("Failed to unmap current frame.");
            }
            if !ffi::cuda::cuCtxPopCurrent_v2(std::ptr::null_mut()).ok() {
                tracing::error!("Failed to pop current context.");
            }
            {
                let (lock, cvar) = &*self.frames_in_flight;
                let mut count = lock.lock().unwrap();
                *count = *count - 1;
                cvar.notify_one();
            }

            let v = !(1 << self.idx);
            self.frame_in_use
                .fetch_and(v, std::sync::atomic::Ordering::SeqCst);
        }
    }
}

pub struct GpuFramePtr<'a> {
    pub ptr: CUdeviceptr,
    _lifetime: PhantomData<&'a ()>,
    _unsend: PhantomData<std::sync::MutexGuard<'static, ()>>,
}

impl<'a> std::ops::Deref for GpuFramePtr<'a> {
    type Target = CUdeviceptr;

    fn deref(&self) -> &CUdeviceptr {
        &self.ptr
    }
}

impl<'a> Drop for GpuFramePtr<'a> {
    fn drop(&mut self) {
        unsafe {
            if !ffi::cuda::cuCtxPopCurrent_v2(std::ptr::null_mut()).ok() {
                tracing::error!("Failed to pop current context.");
            }
        }
    }
}
