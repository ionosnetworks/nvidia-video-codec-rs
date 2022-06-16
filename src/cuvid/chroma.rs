use super::ffi;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum VideoChromaFormat {
    Monochrome = ffi::cuvid::cudaVideoChromaFormat_enum_cudaVideoChromaFormat_Monochrome,
    YUV420 = ffi::cuvid::cudaVideoChromaFormat_enum_cudaVideoChromaFormat_420,
    YUV422 = ffi::cuvid::cudaVideoChromaFormat_enum_cudaVideoChromaFormat_422,
    YUV444 = ffi::cuvid::cudaVideoChromaFormat_enum_cudaVideoChromaFormat_444,
}

impl Into<ffi::cuvid::cudaVideoChromaFormat> for VideoChromaFormat {
    fn into(self) -> ffi::cuvid::cudaVideoChromaFormat {
        self as ffi::cuvid::cudaVideoChromaFormat
    }
}

impl From<ffi::cuvid::cudaVideoChromaFormat> for VideoChromaFormat {
    fn from(format: ffi::cuvid::cudaVideoChromaFormat) -> Self {
        match format {
            ffi::cuvid::cudaVideoChromaFormat_enum_cudaVideoChromaFormat_Monochrome => {
                VideoChromaFormat::Monochrome
            }
            ffi::cuvid::cudaVideoChromaFormat_enum_cudaVideoChromaFormat_420 => {
                VideoChromaFormat::YUV420
            }
            ffi::cuvid::cudaVideoChromaFormat_enum_cudaVideoChromaFormat_422 => {
                VideoChromaFormat::YUV422
            }
            ffi::cuvid::cudaVideoChromaFormat_enum_cudaVideoChromaFormat_444 => {
                VideoChromaFormat::YUV444
            }
            _ => panic!("Invalid cuda video chrome format"),
        }
    }
}
