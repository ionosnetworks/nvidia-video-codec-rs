use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::sync::{Arc, Condvar, Mutex, Weak};

use crate::cuda::mem::CudaPtr;

use super::codec::Codec;
use super::GpuFrame;

pub struct Encoder {
    inner: Box<Inner>,
}

unsafe impl Send for Encoder {}
unsafe impl Sync for Encoder {}

#[allow(non_snake_case)]
pub const fn NVENCAPI_STRUCT_VERSION(ver: u32) -> u32 {
    ffi::cuvid::NVENCAPI_VERSION | (ver << 16) | (0x7 << 28)
}

struct FunctionList(ffi::cuvid::NV_ENCODE_API_FUNCTION_LIST);

impl std::ops::Deref for FunctionList {
    type Target = ffi::cuvid::NV_ENCODE_API_FUNCTION_LIST;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

unsafe impl Send for FunctionList {}
unsafe impl Sync for FunctionList {}

static NVENC_LIB: once_cell::sync::Lazy<FunctionList> = once_cell::sync::Lazy::new(|| {
    let mut function_list: ffi::cuvid::NV_ENCODE_API_FUNCTION_LIST = unsafe { std::mem::zeroed() };
    function_list.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    let res = unsafe { ffi::cuvid::NvEncodeAPICreateInstance(&mut function_list) };
    if res != ffi::cuvid::_NVENCSTATUS_NV_ENC_SUCCESS {
        panic!("Failed to create NvEncode API Instance {}", res);
    }
    FunctionList(function_list)
});

// pub const NV_ENC_CAPS_PARAM_VER: u32 = NVENCAPI_STRUCT_VERSION(1);
// pub const NV_ENC_RESTORE_ENCODER_STATE_PARAMS_VER: u32 = NVENCAPI_STRUCT_VERSION(1);
// pub const NV_ENC_OUTPUT_STATS_BLOCK_VER: u32 = NVENCAPI_STRUCT_VERSION(1);
// pub const NV_ENC_OUTPUT_STATS_ROW_VER: u32 = NVENCAPI_STRUCT_VERSION(1);
// pub const NV_ENC_ENCODE_OUT_PARAMS_VER: u32 = NVENCAPI_STRUCT_VERSION(1);
// pub const NV_ENC_LOOKAHEAD_PIC_PARAMS_VER: u32 = NVENCAPI_STRUCT_VERSION(1);
// pub const NV_ENC_CREATE_INPUT_BUFFER_VER: u32 = NVENCAPI_STRUCT_VERSION(1);
pub const NV_ENC_CREATE_BITSTREAM_BUFFER_VER: u32 = NVENCAPI_STRUCT_VERSION(1);
// pub const NV_ENC_CREATE_MV_BUFFER_VER: u32 = NVENCAPI_STRUCT_VERSION(1);
pub const NV_ENC_RC_PARAMS_VER: u32 = NVENCAPI_STRUCT_VERSION(1);
pub const NV_ENC_CONFIG_VER: u32 = NVENCAPI_STRUCT_VERSION(8) | (1 << 31);
pub const NV_ENC_INITIALIZE_PARAMS_VER: u32 = NVENCAPI_STRUCT_VERSION(6) | (1 << 31);
// pub const NV_ENC_RECONFIGURE_PARAMS_VER: u32 = NVENCAPI_STRUCT_VERSION(1) | (1 << 31);
// pub const NV_ENC_PRESET_CONFIG_VER: u32 = NVENCAPI_STRUCT_VERSION(4) | (1 << 31);
// pub const NV_ENC_PIC_PARAMS_MVC_VER: u32 = NVENCAPI_STRUCT_VERSION(1);
pub const NV_ENC_PIC_PARAMS_VER: u32 = NVENCAPI_STRUCT_VERSION(6) | (1 << 31);
// pub const NV_ENC_MEONLY_PARAMS_VER: u32 = NVENCAPI_STRUCT_VERSION(3);
pub const NV_ENC_LOCK_BITSTREAM_VER: u32 = NVENCAPI_STRUCT_VERSION(1) | (1 << 31);
// pub const NV_ENC_LOCK_INPUT_BUFFER_VER: u32 = NVENCAPI_STRUCT_VERSION(1);
pub const NV_ENC_MAP_INPUT_RESOURCE_VER: u32 = NVENCAPI_STRUCT_VERSION(4);
// pub const NV_ENC_FENCE_POINT_D3D12_VER: u32 = NVENCAPI_STRUCT_VERSION(1);
// pub const NV_ENC_INPUT_RESOURCE_D3D12_VER: u32 = NVENCAPI_STRUCT_VERSION(1);
// pub const NV_ENC_OUTPUT_RESOURCE_D3D12_VER: u32 = NVENCAPI_STRUCT_VERSION(1);
pub const NV_ENC_REGISTER_RESOURCE_VER: u32 = NVENCAPI_STRUCT_VERSION(4);
// pub const NV_ENC_STAT_VER: u32 = NVENCAPI_STRUCT_VERSION(1);
// pub const NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER: u32 = NVENCAPI_STRUCT_VERSION(1);
// pub const NV_ENC_EVENT_PARAMS_VER: u32 = NVENCAPI_STRUCT_VERSION(1);
pub const NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER: u32 = NVENCAPI_STRUCT_VERSION(1);
pub const NV_ENCODE_API_FUNCTION_LIST_VER: u32 = NVENCAPI_STRUCT_VERSION(2);

pub const NV_ENC_CODEC_H264_GUID: ffi::cuvid::GUID = ffi::cuvid::GUID {
    Data1: 0x6bc82762,
    Data2: 0x4e63,
    Data3: 0x4ca4,
    Data4: [0xaa, 0x85, 0x1e, 0x50, 0xf3, 0x21, 0xf6, 0xbf],
};

pub const NV_ENC_CODEC_HEVC_GUID: ffi::cuvid::GUID = ffi::cuvid::GUID {
    Data1: 0x790cdc88,
    Data2: 0x4522,
    Data3: 0x4d7b,
    Data4: [0x94, 0x25, 0xbd, 0xa9, 0x97, 0x5f, 0x76, 0x3],
};

pub const NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID: ffi::cuvid::GUID = ffi::cuvid::GUID {
    Data1: 0xd5bfb716,
    Data2: 0xc604,
    Data3: 0x44e7,
    Data4: [0x9b, 0xb8, 0xde, 0xa5, 0x51, 0xf, 0xc3, 0xac],
};

#[allow(dead_code)]
pub const NV_ENC_H264_PROFILE_MAIN_GUID: ffi::cuvid::GUID = ffi::cuvid::GUID {
    Data1: 0x60b5c1d4,
    Data2: 0x67fe,
    Data3: 0x4790,
    Data4: [0x94, 0xd5, 0xc4, 0x72, 0x6d, 0x7b, 0x6e, 0x6d],
};

// {E7CBC309-4F7A-4b89-AF2A-D537C92BE310}
pub const NV_ENC_H264_PROFILE_HIGH_GUID: ffi::cuvid::GUID = ffi::cuvid::GUID {
    Data1: 0xe7cbc309,
    Data2: 0x4f7a,
    Data3: 0x4b89,
    Data4: [0xaf, 0x2a, 0xd5, 0x37, 0xc9, 0x2b, 0xe3, 0x10],
};

pub const NV_ENC_HEVC_PROFILE_MAIN_GUID: ffi::cuvid::GUID = ffi::cuvid::GUID {
    Data1: 0xb514c39a,
    Data2: 0xb55b,
    Data3: 0x40fa,
    Data4: [0x87, 0x8f, 0xf1, 0x25, 0x3b, 0x4d, 0xfd, 0xec],
};

struct Inner {
    gpu_context: ffi::cuda::CUcontext,
    encoder: EncoderContext,
    input_format: ffi::cuvid::NV_ENC_BUFFER_FORMAT,
    pool: ResourcePool,
    bitstream: Arc<Vec<BitStream>>,
    receiver: flume::Receiver<MappedInputResource>,

    input: Vec<Option<Arc<CudaPtr>>>,
    pending: Vec<MappedInputResource>,
    sender: Option<flume::Sender<MappedInputResource>>,
}

impl Encoder {
    pub fn create(
        gpu_id: usize,
        context: Option<&'static super::super::cuda::context::CuContext>,
        codec: Codec,
        bitrate: u32,
        output_size: (u32, u32),
        framerate: (u32, u32),
        surfaces: NonZeroUsize,
    ) -> Result<Self, ffi::cuda::CUresult> {
        let context = match context {
            Some(context) => super::super::cuda::context::CuContextRef::Borrowed(context),
            None => {
                let device = super::super::cuda::device::CuDevice::new(gpu_id as _)?;
                let context = super::super::cuda::context::CuContext::new(device, 0)?;
                super::super::cuda::context::CuContextRef::Owned(context)
            }
        };

        let encoder = EncoderContext::new(&context)?;
        // let mut encoder: std::ptr::NonNull<std::os::raw::c_void> = std::ptr::NonNull::dangling();
        /*
        let mut params: ffi::cuvid::CUVIDPARSERPARAMS = unsafe { std::mem::zeroed() };
        params.CodecType = codec.into();
        params.ulMaxNumDecodeSurfaces = decode_surfaces.unwrap_or(1) as _;
        params.ulClockRate = 10000000;
        params.ulErrorThreshold = 100;
        params.ulMaxDisplayDelay = if low_latency { 0 } else { 1 };
        params.pfnSequenceCallback = Some(handle_video_sequence_proc);
        params.pfnDecodePicture = Some(handle_picture_decode_proc);
        params.pfnDisplayPicture = Some(handle_picture_display_proc);
        params.pfnGetOperatingPoint = Some(handle_operating_point_proc);
        params.pUserData = (&mut *inner as *mut Inner) as *mut std::os::raw::c_void;
        */

        let guids = unsafe {
            let guid_count = {
                let mut guid_count = 0u32;
                let res =
                    NVENC_LIB.nvEncGetEncodeGUIDCount.unwrap()(encoder.as_ptr(), &mut guid_count);
                wrap!(res, res)?;
                guid_count
            };

            let mut guids: Vec<ffi::cuvid::GUID> = Vec::with_capacity(guid_count as _);
            let mut supported_guid_count = 0u32;
            let res = NVENC_LIB.nvEncGetEncodeGUIDs.unwrap()(
                encoder.as_ptr(),
                guids.as_mut_ptr(),
                guid_count,
                &mut supported_guid_count,
            );
            wrap!(res, res)?;
            guids.set_len(guid_count as _);
            guids
        };

        let selected_codec = {
            match codec {
                Codec::HEVC => guids.iter().find(|&&g| g == NV_ENC_CODEC_HEVC_GUID),
                Codec::H264 | Codec::H264Mvc | Codec::H264Svc => {
                    guids.iter().find(|&&g| g == NV_ENC_CODEC_H264_GUID)
                }
                _ => None,
            }
        };

        let selected_codec = match selected_codec {
            Some(codec) => *codec,
            None => return Err(ffi::cuda::cudaError_enum_CUDA_ERROR_UNKNOWN + 2),
        };

        let _presets = unsafe {
            let preset_count = {
                let mut preset_count = 0u32;
                let res = NVENC_LIB.nvEncGetEncodePresetCount.unwrap()(
                    encoder.as_ptr(),
                    selected_codec,
                    &mut preset_count,
                );
                wrap!(res, res)?;
                preset_count
            };

            let mut presets: Vec<ffi::cuvid::GUID> = Vec::with_capacity(preset_count as _);
            let mut supported_preset_count = 0u32;
            let res = NVENC_LIB.nvEncGetEncodePresetGUIDs.unwrap()(
                encoder.as_ptr(),
                selected_codec,
                presets.as_mut_ptr(),
                preset_count,
                &mut supported_preset_count,
            );
            wrap!(res, res)?;
            presets.set_len(preset_count as _);
            presets
        };

        let selected_preset = NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID;

        let profiles = unsafe {
            let profile_count = {
                let mut profile_count = 0u32;
                let res = NVENC_LIB.nvEncGetEncodeProfileGUIDCount.unwrap()(
                    encoder.as_ptr(),
                    selected_codec,
                    &mut profile_count,
                );
                wrap!(res, res)?;
                profile_count
            };

            let mut profiles: Vec<ffi::cuvid::GUID> = Vec::with_capacity(profile_count as _);
            let mut supported_profile_count = 0u32;
            let res = NVENC_LIB.nvEncGetEncodeProfileGUIDs.unwrap()(
                encoder.as_ptr(),
                selected_codec,
                profiles.as_mut_ptr(),
                profile_count,
                &mut supported_profile_count,
            );
            wrap!(res, res)?;
            profiles.set_len(profile_count as _);
            profiles
        };

        let selected_profile = {
            match codec {
                Codec::HEVC => profiles
                    .iter()
                    .find(|&&g| g == NV_ENC_HEVC_PROFILE_MAIN_GUID),
                Codec::H264 | Codec::H264Mvc | Codec::H264Svc => profiles
                    .iter()
                    .find(|&&g| g == NV_ENC_H264_PROFILE_HIGH_GUID),
                _ => None,
            }
        };

        let selected_profile = match selected_profile {
            Some(profile) => *profile,
            None => return Err(ffi::cuda::cudaError_enum_CUDA_ERROR_UNKNOWN + 3),
        };

        let input_formats = unsafe {
            let input_format_count = {
                let mut input_format_count = 0u32;
                let res = NVENC_LIB.nvEncGetInputFormatCount.unwrap()(
                    encoder.as_ptr(),
                    selected_codec,
                    &mut input_format_count,
                );
                wrap!(res, res)?;
                input_format_count
            };

            let mut input_formats: Vec<ffi::cuvid::NV_ENC_BUFFER_FORMAT> =
                Vec::with_capacity(input_format_count as _);
            let mut supported_input_format_count = 0u32;
            let res = NVENC_LIB.nvEncGetInputFormats.unwrap()(
                encoder.as_ptr(),
                selected_codec,
                input_formats.as_mut_ptr(),
                input_format_count,
                &mut supported_input_format_count,
            );
            wrap!(res, res)?;
            input_formats.set_len(input_format_count as _);
            input_formats
        };

        let selected_input_format = match input_formats
            .iter()
            .find(|&&f| f == ffi::cuvid::_NV_ENC_BUFFER_FORMAT_NV_ENC_BUFFER_FORMAT_NV12)
        {
            Some(format) => *format,
            None => return Err(ffi::cuda::cudaError_enum_CUDA_ERROR_UNKNOWN + 4),
        };

        let mut encode_config: ffi::cuvid::NV_ENC_CONFIG = unsafe { std::mem::zeroed() };
        encode_config.version = NV_ENC_CONFIG_VER;
        encode_config.profileGUID = selected_profile;
        encode_config.gopLength = 50; // 2 seconds
        encode_config.frameIntervalP = 0;
        match codec {
            Codec::HEVC => {}
            Codec::H264 | Codec::H264Mvc | Codec::H264Svc => {}
            _ => (),
        };
        encode_config.rcParams.version = NV_ENC_RC_PARAMS_VER;
        encode_config.rcParams.rateControlMode =
            ffi::cuvid::_NV_ENC_PARAMS_RC_MODE_NV_ENC_PARAMS_RC_VBR;
        encode_config.rcParams.averageBitRate = bitrate;

        let mut params: ffi::cuvid::NV_ENC_INITIALIZE_PARAMS = unsafe { std::mem::zeroed() };
        params.version = NV_ENC_INITIALIZE_PARAMS_VER;
        params.encodeGUID = selected_codec;
        params.presetGUID = selected_preset;
        params.encodeWidth = output_size.0;
        params.encodeHeight = output_size.1;
        //params.darWidth = output_size.0;
        //params.darHeight = output_size.1;
        //params.encodeConfig = &mut encode_config;
        params.bufferFormat = selected_input_format;
        params.frameRateNum = framerate.0;
        params.frameRateDen = framerate.1;
        params.enablePTD = 1;

        unsafe {
            let res = NVENC_LIB.nvEncInitializeEncoder.unwrap()(encoder.as_ptr(), &mut params);
            wrap!(res, res)?;
        }

        let (sender, receiver) = flume::bounded(surfaces.get());

        let inner = Box::new(Inner {
            gpu_context: context.context,
            input_format: selected_input_format,
            pool: ResourcePool::with_capacity(surfaces),
            bitstream: Arc::new(
                (0..surfaces.get())
                    .map(|_| BitStream::new(&encoder))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            pending: Vec::new(),
            input: (0..surfaces.get()).map(|_| None).collect::<Vec<_>>(),
            encoder,
            sender: Some(sender),
            receiver,
        });

        Ok(Self { inner })
    }

    pub fn queue_gpu_frame(
        &mut self,
        frame: GpuFrame,
        copy: bool,
    ) -> Result<bool, ffi::cuda::CUresult> {
        if self.inner.sender.is_none() {
            panic!("Encoder::queue was called, but eos has already been sent.");
        }

        let permit = self.inner.pool.get();

        if self.inner.input[permit.0].is_none() && copy {
            self.inner.input[permit.0] = Some(Arc::new(
                crate::cuda::mem::malloc_pitch_ctx(
                    self.inner.gpu_context,
                    frame.width as _,
                    (frame.height * 3 / 2) as _,
                    16,
                )
                .unwrap(),
            ));
        }
        let (resource, width, height, pitch) = if copy {
            let mem = self.inner.input[permit.0].as_ref().unwrap();

            mem.copy_from_device_2d(
                frame.ptr,
                frame.pitch as _,
                frame.width as _,
                (frame.height * 3 / 2) as _,
            )
            .expect("Failed to copy from gpu frame");
            let (width, height) = (frame.width, frame.height);
            drop(frame);
            (Ok(mem.clone()), width, height, mem.pitch())
        } else {
            let (width, height, pitch) = (frame.width, frame.height, frame.pitch);
            (Err(frame), width, height, pitch as u64)
        };

        let mut pic_params: ffi::cuvid::NV_ENC_PIC_PARAMS = unsafe { std::mem::zeroed() };
        pic_params.version = NV_ENC_PIC_PARAMS_VER;
        pic_params.inputWidth = width;
        pic_params.inputHeight = height;
        pic_params.inputPitch = pitch as _;
        pic_params.encodePicFlags = 0;
        pic_params.frameIdx = 0;

        let resource = MappedInputResource::new(
            &self.inner.encoder,
            permit,
            width,
            height,
            pitch as _,
            self.inner.input_format,
            resource,
        )?;

        // pic_params.inputTimeStamp = NV_ENC_PIC_PARAMS_VER;
        // pic_params.inputDuration = NV_ENC_PIC_PARAMS_VER;
        pic_params.inputBuffer = resource.as_ptr();
        pic_params.outputBitstream = self.inner.bitstream[permit.0].as_ptr();
        pic_params.bufferFmt = self.inner.input_format;
        pic_params.pictureStruct = ffi::cuvid::_NV_ENC_PIC_STRUCT_NV_ENC_PIC_STRUCT_FRAME;
        pic_params.pictureType = 0;
        //pic_params.codecPicParams = 0;

        // drop(frame);

        let res = unsafe {
            let res =
                NVENC_LIB.nvEncEncodePicture.unwrap()(self.inner.encoder.as_ptr(), &mut pic_params);
            wrap!(res, res)
        };

        match res {
            Ok(_) => {
                let sender = self.inner.sender.as_ref().unwrap();
                for resource in self.inner.pending.drain(..) {
                    sender.send(resource).unwrap();
                }
                sender.send(resource).unwrap();
                Ok(true)
            }
            Err(ffi::cuvid::_NVENCSTATUS_NV_ENC_ERR_NEED_MORE_INPUT) => {
                self.inner.pending.push(resource);
                Ok(false)
            }
            Err(err) => Err(err),
        }
    }

    pub fn send_eos(&mut self) -> Result<(), ffi::cuda::CUresult> {
        let mut pic_params: ffi::cuvid::NV_ENC_PIC_PARAMS = unsafe { std::mem::zeroed() };
        pic_params.version = NV_ENC_PIC_PARAMS_VER;
        pic_params.encodePicFlags = ffi::cuvid::_NV_ENC_PIC_FLAGS_NV_ENC_PIC_FLAG_EOS;
        pic_params.frameIdx = !0;

        unsafe {
            let res =
                NVENC_LIB.nvEncEncodePicture.unwrap()(self.inner.encoder.as_ptr(), &mut pic_params);
            wrap!(res, res)?;
        }
        let _ = self.inner.sender.take();
        Ok(())
    }

    pub fn frames(&self) -> FramesIter {
        FramesIter {
            bitstream: Arc::downgrade(&self.inner.bitstream),
            pool: self.inner.pool.clone(),
            receiver: self.inner.receiver.clone(),
        }
    }
}

struct EncoderContext {
    inner: std::ptr::NonNull<std::os::raw::c_void>,
}

impl EncoderContext {
    fn new(context: &crate::cuda::context::CuContextRef<'_>) -> Result<Self, ffi::cuda::CUresult> {
        let mut params: ffi::cuvid::NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS =
            unsafe { std::mem::zeroed() };
        params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
        params.deviceType = ffi::cuvid::_NV_ENC_DEVICE_TYPE_NV_ENC_DEVICE_TYPE_CUDA;
        params.device = context.context as _;
        params.apiVersion = ffi::cuvid::NVENCAPI_VERSION;

        // let encoder: std::ptr::NonNull<std::os::raw::c_void> = std::ptr::NonNull::dangling();
        let mut encoder: *mut std::os::raw::c_void = std::ptr::null_mut();
        unsafe {
            let res = NVENC_LIB.nvEncOpenEncodeSessionEx.unwrap()(&mut params, &mut encoder);
            wrap!(res, res)?;
        }
        Ok(Self {
            inner: std::ptr::NonNull::new(encoder).unwrap(),
        })
    }

    fn as_ptr(&self) -> *mut std::os::raw::c_void {
        self.inner.as_ptr()
    }
}

impl Drop for EncoderContext {
    fn drop(&mut self) {
        unsafe {
            NVENC_LIB.nvEncDestroyEncoder.unwrap()(self.inner.as_ptr());
        }
    }
}

struct MappedInputResource {
    permit: Permit,
    resource: std::ptr::NonNull<std::os::raw::c_void>,
    mapped: std::ptr::NonNull<std::os::raw::c_void>,
    encoder: std::ptr::NonNull<std::os::raw::c_void>,
    _frame: Option<GpuFrame>,
}

impl MappedInputResource {
    pub fn new(
        encoder: &EncoderContext,
        permit: Permit,
        width: u32,
        height: u32,
        pitch: u32,
        input_format: ffi::cuvid::NV_ENC_BUFFER_FORMAT,
        resource: Result<Arc<CudaPtr>, GpuFrame>,
    ) -> Result<Self, ffi::cuda::CUresult> {
        let mut register_resource: ffi::cuvid::NV_ENC_REGISTER_RESOURCE =
            unsafe { std::mem::zeroed() };
        register_resource.version = NV_ENC_REGISTER_RESOURCE_VER;
        register_resource.resourceType =
            ffi::cuvid::_NV_ENC_INPUT_RESOURCE_TYPE_NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
        register_resource.width = width;
        register_resource.height = height;
        register_resource.pitch = pitch;
        register_resource.bufferFormat = input_format;
        register_resource.bufferUsage = ffi::cuvid::_NV_ENC_BUFFER_USAGE_NV_ENC_INPUT_IMAGE;
        register_resource.resourceToRegister = match &resource {
            Ok(resource) => resource.as_ptr() as _,
            Err(err) => err.ptr as _,
        };

        unsafe {
            let res =
                NVENC_LIB.nvEncRegisterResource.unwrap()(encoder.as_ptr(), &mut register_resource);
            wrap!(res, res)?;
        }

        let mut input_resource: ffi::cuvid::NV_ENC_MAP_INPUT_RESOURCE =
            unsafe { std::mem::zeroed() };
        input_resource.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
        input_resource.registeredResource = register_resource.registeredResource;
        unsafe {
            let res =
                NVENC_LIB.nvEncMapInputResource.unwrap()(encoder.as_ptr(), &mut input_resource);
            wrap!(res, res)?;
        }

        Ok(Self {
            permit,
            resource: std::ptr::NonNull::new(register_resource.registeredResource).unwrap(),
            mapped: std::ptr::NonNull::new(input_resource.mappedResource).unwrap(),
            encoder: encoder.inner,
            _frame: match resource {
                Ok(_) => None,
                Err(err) => Some(err),
            },
        })
    }

    fn as_ptr(&self) -> *mut std::os::raw::c_void {
        self.mapped.as_ptr()
    }
}

impl Drop for MappedInputResource {
    fn drop(&mut self) {
        unsafe {
            NVENC_LIB.nvEncUnmapInputResource.unwrap()(self.encoder.as_ptr(), self.mapped.as_ptr());
        }
        unsafe {
            NVENC_LIB.nvEncUnregisterResource.unwrap()(
                self.encoder.as_ptr(),
                self.resource.as_ptr(),
            );
        }
    }
}

struct BitStream {
    inner: std::ptr::NonNull<std::os::raw::c_void>,
    encoder: std::ptr::NonNull<std::os::raw::c_void>,
}
impl BitStream {
    pub fn new(encoder: &EncoderContext) -> Result<Self, ffi::cuda::CUresult> {
        let mut params: ffi::cuvid::NV_ENC_CREATE_BITSTREAM_BUFFER = unsafe { std::mem::zeroed() };
        params.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;

        unsafe {
            let res = NVENC_LIB.nvEncCreateBitstreamBuffer.unwrap()(encoder.as_ptr(), &mut params);
            wrap!(res, res)?;
        }

        Ok(Self {
            inner: std::ptr::NonNull::new(params.bitstreamBuffer).unwrap(),
            encoder: encoder.inner,
        })
    }

    pub fn to_vec(&self) -> Result<(Vec<u8>, bool, u64, u64), ffi::cuda::CUresult> {
        let mut params: ffi::cuvid::NV_ENC_LOCK_BITSTREAM = unsafe { std::mem::zeroed() };
        params.version = NV_ENC_LOCK_BITSTREAM_VER;
        params.outputBitstream = self.inner.as_ptr();

        unsafe {
            let res = NVENC_LIB.nvEncLockBitstream.unwrap()(self.encoder.as_ptr(), &mut params);
            wrap!(res, res)?;
        };
        let data = unsafe {
            std::slice::from_raw_parts(
                params.bitstreamBufferPtr as *const u8,
                params.bitstreamSizeInBytes as usize,
            )
        };
        let data = data.to_vec();
        // let frame_idx = params.frameIdx;

        unsafe {
            let res =
                NVENC_LIB.nvEncUnlockBitstream.unwrap()(self.encoder.as_ptr(), self.inner.as_ptr());
            wrap!(res, res)?;
        };
        let is_idr = params.pictureType == ffi::cuvid::_NV_ENC_PIC_TYPE_NV_ENC_PIC_TYPE_IDR;

        Ok((data, is_idr, params.outputDuration, params.outputTimeStamp))
    }

    fn as_ptr(&self) -> *mut std::os::raw::c_void {
        self.inner.as_ptr()
    }
}

impl Drop for BitStream {
    fn drop(&mut self) {
        unsafe {
            NVENC_LIB.nvEncDestroyBitstreamBuffer.unwrap()(
                self.encoder.as_ptr(),
                self.inner.as_ptr(),
            );
        }
    }
}

pub struct FramesIter {
    bitstream: Weak<Vec<BitStream>>,
    receiver: flume::Receiver<MappedInputResource>,
    pool: ResourcePool,
}

impl Iterator for FramesIter {
    type Item = (Vec<u8>, bool, u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        let input_resource = self.receiver.recv().ok()?;
        let bitstreams = self.bitstream.upgrade()?;
        let bitstream = &bitstreams[input_resource.permit.0];
        let vec = match bitstream.to_vec() {
            Ok(vec) => Some(vec),
            Err(_) => None,
        };
        self.pool.put(input_resource.permit);
        vec
    }
}

#[derive(Clone)]
struct ResourcePool {
    pool: Arc<(Mutex<VecDeque<usize>>, Condvar)>,
}

#[derive(Clone, Copy, Debug)]
struct Permit(usize);

impl ResourcePool {
    pub fn with_capacity(size: NonZeroUsize) -> Self {
        let v = (0..size.get()).collect::<VecDeque<_>>();
        Self {
            pool: Arc::new((Mutex::new(v), Condvar::new())),
        }
    }

    pub fn get(&self) -> Permit {
        let (lock, cvar) = &*self.pool;
        let mut pool = lock.lock().unwrap();
        while pool.len() == 0 {
            pool = cvar.wait(pool).unwrap();
        }
        Permit(pool.pop_front().unwrap())
    }

    pub fn put(&self, permit: Permit) {
        let (lock, cvar) = &*self.pool;
        let mut pool = lock.lock().unwrap();
        pool.push_back(permit.0);
        cvar.notify_one();
    }
}
