use cuda::device::CuDevice;

pub struct CuContext {
    pub(crate) context: ffi::cuda::CUcontext,
}

unsafe impl Send for CuContext {}
unsafe impl Sync for CuContext {}

impl CuContext {
    pub fn new(dev: CuDevice, flags: u32) -> Result<CuContext, ffi::cuda::CUresult> {
        let mut ctx = CuContext {
            context: std::ptr::null_mut(),
        };
        let res = unsafe { ffi::cuda::cuCtxCreate_v2(&mut ctx.context, flags, dev.device) };

        wrap!(ctx, res)
    }

    pub fn get_api_version(&self) -> Result<u32, ffi::cuda::CUresult> {
        let mut ver = 0;
        let res = unsafe { ffi::cuda::cuCtxGetApiVersion(self.context, &mut ver as *mut u32) };

        wrap!(ver, res)
    }
}

impl Drop for CuContext {
    fn drop(&mut self) {
        unsafe {
            ffi::cuda::cuCtxDestroy_v2(self.context);
        }
    }
}
