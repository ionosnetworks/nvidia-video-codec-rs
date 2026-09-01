use std::cell::RefCell;
use std::ffi::CStr;
use std::mem::MaybeUninit;
use std::os::raw::c_char;

#[cfg(feature = "async")]
pub extern crate futures;
pub extern crate nvidia_video_codec_sys as ffi;

#[macro_use]
mod macros;

pub mod cuda;
pub mod cuvid;

thread_local! {
    static INIT: RefCell<Option<()>> = RefCell::new(None);
}

pub fn init() {
    INIT.with(|init| {
        let mut status = init.borrow_mut();
        if status.is_none() {
            let _ = unsafe { ffi::cuvid::cuInit(0) };
            status.replace(());
        }
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaError(pub u32);

impl CudaError {
    pub fn name(&self) -> String {
        let mut ptr: *const c_char = std::ptr::null();
        let res = unsafe { ffi::cuvid::cuGetErrorName(self.0, &mut ptr) };
        if res.ok() && !ptr.is_null() {
            unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
        } else {
            format!("CUDA_ERROR({})", self.0)
        }
    }

    pub fn string(&self) -> String {
        let mut ptr: *const c_char = std::ptr::null();
        let res = unsafe { ffi::cuvid::cuGetErrorString(self.0, &mut ptr) };
        if res.ok() && !ptr.is_null() {
            unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
        } else {
            format!("unrecognized error code ({})", self.0)
        }
    }
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.name(), self.string())
    }
}

impl std::error::Error for CudaError {}

pub trait CudaResult {
    fn ok(&self) -> bool;
    fn err(&self) -> Result<(), CudaError>;
}

impl CudaResult for ffi::cuda::CUresult {
    fn ok(&self) -> bool {
        return *self == ffi::cuda::cudaError_enum_CUDA_SUCCESS;
    }

    fn err(&self) -> Result<(), CudaError> {
        if *self == ffi::cuda::cudaError_enum_CUDA_SUCCESS {
            Ok(())
        } else {
            Err(CudaError(*self))
        }
    }
}

pub trait NppResult {
    fn ok(&self) -> bool;
    fn err(&self) -> Result<(), CudaError>;
}

impl NppResult for ffi::npp::NppStatus {
    fn ok(&self) -> bool {
        return *self == ffi::npp::NppStatus_NPP_SUCCESS;
    }

    fn err(&self) -> Result<(), CudaError> {
        if *self == ffi::npp::NppStatus_NPP_SUCCESS {
            Ok(())
        } else {
            Err(CudaError(*self as u32))
        }
    }
}

pub fn nv12_to_rgb24(
    ptr: ffi::cuvid::CUdeviceptr,
    width: u32,
    height: u32,
    pitch: i32,
    dest_ptr: *mut std::os::raw::c_void,
    dest_pitch: i32,
    stream: Option<&cuda::stream::CuStream>,
) -> Result<(), CudaError> {
    let src: [*const ffi::npp::Npp8u; 2] = unsafe {
        [
            (ptr as *const ffi::npp::Npp8u),
            (ptr as *const ffi::npp::Npp8u).offset((pitch * (height as i32)) as isize),
        ]
    };
    let size_roi = ffi::npp::NppiSize {
        width: width as _,
        height: height as _,
    };

    if let Some(stream) = stream {
        unsafe {
            if ffi::npp::nppGetStream() != (stream.stream as _) {
                ffi::npp::nppSetStream(stream.stream as _);
            }
        }
    }

    let stream_ctx = unsafe {
        let mut ctx: MaybeUninit<ffi::npp::NppStreamContext> = MaybeUninit::uninit();
        ffi::npp::nppGetStreamContext(ctx.as_mut_ptr()).err()?;
        ctx.assume_init()
    };

    unsafe {
        ffi::npp::nppiNV12ToRGB_8u_P2C3R_Ctx(
            src.as_ptr(),
            pitch,
            dest_ptr as _,
            dest_pitch,
            size_roi,
            stream_ctx,
        )
        .err()?;
    }

    Ok(())
}

pub fn nv12_to_bgr24(
    ptr: ffi::cuvid::CUdeviceptr,
    width: u32,
    height: u32,
    pitch: i32,
    dest_ptr: *mut std::os::raw::c_void,
    dest_pitch: i32,
    stream: Option<&cuda::stream::CuStream>,
) -> Result<(), CudaError> {
    let src: [*const ffi::npp::Npp8u; 2] = unsafe {
        [
            (ptr as *const ffi::npp::Npp8u),
            (ptr as *const ffi::npp::Npp8u).offset((pitch * (height as i32)) as isize),
        ]
    };
    let size_roi = ffi::npp::NppiSize {
        width: width as _,
        height: height as _,
    };

    if let Some(stream) = stream {
        unsafe {
            if ffi::npp::nppGetStream() != (stream.stream as _) {
                ffi::npp::nppSetStream(stream.stream as _);
            }
        }
    }

    let stream_ctx = unsafe {
        let mut ctx: MaybeUninit<ffi::npp::NppStreamContext> = MaybeUninit::uninit();
        ffi::npp::nppGetStreamContext(ctx.as_mut_ptr()).err()?;
        ctx.assume_init()
    };

    unsafe {
        ffi::npp::nppiNV12ToBGR_8u_P2C3R_Ctx(
            src.as_ptr(),
            pitch,
            dest_ptr as _,
            dest_pitch,
            size_roi,
            stream_ctx,
        )
        .err()?;
    }

    Ok(())
}
