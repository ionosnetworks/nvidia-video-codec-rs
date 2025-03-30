use ffi::cuvid::CUresult;

pub struct CudaPtr {
    context: Option<ffi::cuda::CUcontext>,
    inner: ffi::cuda::CUdeviceptr,
    pitch: u64,
}

pub fn malloc_pitch_ctx(
    context: ffi::cuda::CUcontext,
    width_in_bytes: u64,
    height: u64,
    element_size_bytes: u32,
) -> Result<CudaPtr, CUresult> {
    let mut pitch: std::ffi::c_ulong = 0;
    let mut ptr: ffi::cuda::CUdeviceptr = 0;
    unsafe {
        ffi::cuda::cuCtxPushCurrent_v2(context);
    }
    let res = unsafe {
        ffi::cuda::cuMemAllocPitch_v2(
            &mut ptr,
            &mut pitch,
            width_in_bytes,
            height,
            element_size_bytes,
        )
    };
    unsafe {
        ffi::cuda::cuCtxPopCurrent_v2(std::ptr::null_mut());
    }
    let val = CudaPtr {
        context: Some(context),
        inner: ptr,
        pitch,
    };
    wrap!(val, res)
}

impl CudaPtr {
    pub fn as_ptr(&self) -> ffi::cuda::CUdeviceptr {
        self.inner
    }

    pub fn pitch(&self) -> u64 {
        self.pitch
    }

    pub fn copy_from_device_2d(
        &self,
        other: ffi::cuda::CUdeviceptr,
        pitch: u64,
        width: u64,
        height: u64,
    ) -> Result<(), CUresult> {
        let mut m: ffi::cuda::CUDA_MEMCPY2D_v2 = unsafe { std::mem::zeroed() };
        m.srcMemoryType = ffi::cuda::CUmemorytype_enum_CU_MEMORYTYPE_DEVICE;
        m.srcDevice = other;
        m.srcPitch = pitch;

        m.dstMemoryType = ffi::cuda::CUmemorytype_enum_CU_MEMORYTYPE_DEVICE;
        m.dstDevice = self.inner;
        m.dstPitch = self.pitch;

        m.WidthInBytes = width;
        m.Height = height;

        if let Some(ctx) = self.context {
            unsafe {
                ffi::cuda::cuCtxPushCurrent_v2(ctx);
            }
        }
        let res = unsafe { ffi::cuda::cuMemcpy2D_v2(&m) };
        if self.context.is_some() {
            unsafe {
                ffi::cuda::cuCtxPopCurrent_v2(std::ptr::null_mut());
            }
        }
        let val = ();
        wrap!(val, res)
    }
}

impl Drop for CudaPtr {
    fn drop(&mut self) {
        unsafe {
            if let Some(ctx) = self.context {
                ffi::cuda::cuCtxPushCurrent_v2(ctx);
            }
            ffi::cuda::cuMemFree_v2(self.inner);
            if self.context.is_some() {
                ffi::cuda::cuCtxPopCurrent_v2(std::ptr::null_mut());
            }
        }
    }
}
