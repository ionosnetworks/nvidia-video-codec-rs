use super::ffi;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum VideoSurfaceFormat {
    NV12 = ffi::cuvid::cudaVideoSurfaceFormat_enum_cudaVideoSurfaceFormat_NV12,
    P016 = ffi::cuvid::cudaVideoSurfaceFormat_enum_cudaVideoSurfaceFormat_P016,
    YUV444 = ffi::cuvid::cudaVideoSurfaceFormat_enum_cudaVideoSurfaceFormat_YUV444,
    YUV444_16 = ffi::cuvid::cudaVideoSurfaceFormat_enum_cudaVideoSurfaceFormat_YUV444_16Bit,
}

impl Into<ffi::cuvid::cudaVideoSurfaceFormat> for VideoSurfaceFormat {
    fn into(self) -> ffi::cuvid::cudaVideoSurfaceFormat {
        self as ffi::cuvid::cudaVideoSurfaceFormat
    }
}

impl From<ffi::cuvid::cudaVideoSurfaceFormat> for VideoSurfaceFormat {
    fn from(format: ffi::cuvid::cudaVideoSurfaceFormat) -> Self {
        match format {
            ffi::cuvid::cudaVideoSurfaceFormat_enum_cudaVideoSurfaceFormat_NV12 => {
                VideoSurfaceFormat::NV12
            }
            ffi::cuvid::cudaVideoSurfaceFormat_enum_cudaVideoSurfaceFormat_P016 => {
                VideoSurfaceFormat::P016
            }
            ffi::cuvid::cudaVideoSurfaceFormat_enum_cudaVideoSurfaceFormat_YUV444 => {
                VideoSurfaceFormat::YUV444
            }
            ffi::cuvid::cudaVideoSurfaceFormat_enum_cudaVideoSurfaceFormat_YUV444_16Bit => {
                VideoSurfaceFormat::YUV444_16
            }
            _ => panic!("Invalid cuda video surface formate"),
        }
    }
}
