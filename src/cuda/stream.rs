use std::os::raw::c_char;
use std::os::raw::c_int;

use super::CudaResult;
use ffi::cuda::*;

pub struct CuStream {
    pub(crate) stream: CUstream,
}

impl CuStream {
    pub fn with_context(
        ctx: super::context::CuContext,
        non_blocking: bool,
    ) -> Result<Self, ffi::cuda::CUresult> {
        let mut stream = CuStream {
            stream: std::ptr::null_mut(),
        };
        let flags = if non_blocking {
            ffi::cuda::CUstream_flags_enum_CU_STREAM_NON_BLOCKING
        } else {
            ffi::cuda::CUstream_flags_enum_CU_STREAM_DEFAULT
        };
        unsafe { ffi::cuda::cuCtxPushCurrent_v2(ctx.context).err()? };
        let res = unsafe { ffi::cuda::cuStreamCreate(&mut stream.stream, flags) };
        unsafe { ffi::cuda::cuCtxPopCurrent_v2(std::ptr::null_mut()).err()? };

        wrap!(stream, res)
    }
}

impl Drop for CuStream {
    fn drop(&mut self) {
        unsafe {
            ffi::cuda::cuStreamDestroy_v2(self.stream);
        }
    }
}
