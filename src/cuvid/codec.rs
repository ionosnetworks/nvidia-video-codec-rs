use super::ffi;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Codec {
    MPEG1 = ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_MPEG1,
    MPEG2 = ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_MPEG2,
    VC1 = ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_VC1,
    H264 = ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_H264,
    JPEG = ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_JPEG,
    H264Svc = ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_H264_SVC,
    H264Mvc = ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_H264_MVC,
    HEVC = ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_HEVC,
    VP8 = ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_VP8,
    VP9 = ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_VP9,
    AV1 = ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_AV1,

    YUV420 = ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_YUV420,
    YV12 = ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_YV12,
    NV12 = ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_NV12,
    YUYV = ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_YUYV,
    UYVY = ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_UYVY,
}

impl Into<ffi::cuvid::cudaVideoCodec> for Codec {
    fn into(self) -> ffi::cuvid::cudaVideoCodec {
        self as ffi::cuvid::cudaVideoCodec
    }
}

impl From<ffi::cuvid::cudaVideoCodec> for Codec {
    fn from(codec: ffi::cuvid::cudaVideoCodec) -> Self {
        match codec {
            ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_MPEG1 => Codec::MPEG1,
            ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_MPEG2 => Codec::MPEG2,
            ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_VC1 => Codec::VC1,
            ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_H264 => Codec::H264,
            ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_JPEG => Codec::JPEG,
            ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_H264_SVC => Codec::H264Svc,
            ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_H264_MVC => Codec::H264Mvc,
            ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_HEVC => Codec::HEVC,
            ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_VP8 => Codec::VP8,
            ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_VP9 => Codec::VP9,
            ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_AV1 => Codec::AV1,

            ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_YUV420 => Codec::YUV420,
            ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_YV12 => Codec::YV12,
            ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_NV12 => Codec::NV12,
            ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_YUYV => Codec::YUYV,
            ffi::cuvid::cudaVideoCodec_enum_cudaVideoCodec_UYVY => Codec::UYVY,
            _ => panic!("Invalid cuda video codec"),
        }
    }
}
