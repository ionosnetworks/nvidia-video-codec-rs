use std::cell::RefCell;
use std::mem::MaybeUninit;

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

pub trait CudaResult {
    fn ok(&self) -> bool;
    fn err(&self) -> Result<(), Self>
    where
        Self: Sized;
}

impl CudaResult for ffi::cuda::CUresult {
    fn ok(&self) -> bool {
        return *self == ffi::cuda::cudaError_enum_CUDA_SUCCESS;
    }

    fn err(&self) -> Result<(), Self> {
        if *self == ffi::cuda::cudaError_enum_CUDA_SUCCESS {
            Ok(())
        } else {
            Err(*self)
        }
    }
}

pub trait NppResult {
    fn ok(&self) -> bool;
    fn err(&self) -> Result<(), Self>
    where
        Self: Sized;
}

impl NppResult for ffi::npp::NppStatus {
    fn ok(&self) -> bool {
        return *self == ffi::npp::NppStatus_NPP_SUCCESS;
    }

    fn err(&self) -> Result<(), Self> {
        if *self == ffi::npp::NppStatus_NPP_SUCCESS {
            Ok(())
        } else {
            Err(*self)
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
) -> Result<(), ffi::npp::NppStatus> {
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
) -> Result<(), ffi::npp::NppStatus> {
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
