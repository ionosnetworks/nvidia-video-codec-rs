use super::{ffi, CudaResult};

pub use ffi::cuvid::CUdeviceptr;

mod chroma;
mod codec;
mod surface;

pub use self::chroma::VideoChromaFormat;
pub use self::codec::Codec;
pub use self::surface::VideoSurfaceFormat;

pub struct Decoder {
    inner: Box<Inner>,
}

unsafe impl Send for Decoder {}
unsafe impl Sync for Decoder {}

struct Inner {
    parser: ffi::cuvid::CUvideoparser,
    lock: ffi::cuvid::CUvideoctxlock,
    context: super::cuda::context::CuContext,
    decoder: ffi::cuvid::CUvideodecoder,
    keyframe_only: bool,
    requested_size: (u32, u32),

    video_fmt: Option<ffi::cuvid::CUVIDEOFORMAT>,
    codec: Codec,
    chroma_format: VideoChromaFormat,
    bit_depth_minus8: u8,
    bpp: u8,
    output_format: VideoSurfaceFormat,
    out_size: (u32, u32),
    coded_size: (u32, u32),
    sender: Option<flume::Sender<PreparedFrame>>,
    receiver: flume::Receiver<PreparedFrame>,
}

#[derive(Debug)]
struct PreparedFrame {
    timestamp: i64,
    index: i32,
    parameters: ffi::cuvid::CUVIDPROCPARAMS,
}

impl PreparedFrame {
    fn timestamp(&self) -> i64 {
        self.timestamp
    }
}

pub struct GpuFrame {
    pub width: u32,
    pub height: u32,
    pub ptr: CUdeviceptr,
    pub pitch: u32,
    pub timestamp: i64,
    decoder: ffi::cuvid::CUvideodecoder,
}

impl Drop for GpuFrame {
    fn drop(&mut self) {
        unsafe {
            if !ffi::cuda::cuCtxPopCurrent_v2(std::ptr::null_mut()).ok() {
                tracing::error!("Failed to pop current context.");
            }

            if !ffi::cuvid::cuvidUnmapVideoFrame64(self.decoder, self.ptr).ok() {
                tracing::error!("Failed to unmap current frame.");
            }
        }
    }
}

impl Decoder {
    pub fn create(
        gpu_id: usize,
        codec: Codec,
        keyframe_only: bool,
        low_latency: bool,
        output_size: (u32, u32),
    ) -> Result<Self, ffi::cuda::CUresult> {
        let device = super::cuda::device::CuDevice::new(gpu_id as _)?;
        let context = super::cuda::context::CuContext::new(device, 0)?;

        let mut parser: ffi::cuvid::CUvideoparser = std::ptr::null_mut();
        let mut ctx_lock: ffi::cuvid::CUvideoctxlock = std::ptr::null_mut();

        unsafe {
            let res = ffi::cuvid::cuvidCtxLockCreate(&mut ctx_lock, context.context as _);
            wrap!(res, res)?;
        }
        let (sender, receiver) = flume::unbounded();

        let mut inner = Box::new(Inner {
            parser,
            context,
            codec,
            lock: ctx_lock,
            chroma_format: VideoChromaFormat::Monochrome,
            decoder: std::ptr::null_mut(),
            keyframe_only,
            video_fmt: None,
            bit_depth_minus8: 0,
            bpp: 0,
            output_format: VideoSurfaceFormat::NV12,
            out_size: (0, 0),
            coded_size: (0, 0),
            requested_size: output_size,
            receiver,
            sender: Some(sender),
        });

        let mut params: ffi::cuvid::CUVIDPARSERPARAMS = unsafe { std::mem::zeroed() };
        params.CodecType = codec.into();
        params.ulMaxNumDecodeSurfaces = 1;
        params.ulClockRate = 10000000;
        params.ulErrorThreshold = 100;
        params.ulMaxDisplayDelay = if low_latency { 0 } else { 1 };
        params.pfnSequenceCallback = Some(handle_video_sequence_proc);
        params.pfnDecodePicture = Some(handle_picture_decode_proc);
        params.pfnDisplayPicture = Some(handle_picture_display_proc);
        params.pfnGetOperatingPoint = Some(handle_operating_point_proc);
        params.pUserData = (&mut *inner as *mut Inner) as *mut std::os::raw::c_void;

        unsafe {
            let res = ffi::cuvid::cuvidCreateVideoParser(&mut parser, &mut params);
            wrap!(res, res)?;
        }
        inner.parser = parser;

        Ok(Self { inner })
    }

    pub fn queue(&self, data: &[u8], timestamp: i64) -> Result<(), ffi::cuda::CUresult> {
        let mut packet = ffi::cuvid::CUVIDSOURCEDATAPACKET {
            flags: ffi::cuvid::CUvideopacketflags_CUVID_PKT_TIMESTAMP as _,
            payload_size: data.len() as u64,
            payload: data.as_ptr(),
            timestamp: timestamp,
        };

        unsafe {
            let res = ffi::cuvid::cuvidParseVideoData(self.inner.parser, &mut packet);
            wrap!(res, res)?;
        }

        Ok(())
    }

    pub fn send_eos(&self) -> Result<(), ffi::cuda::CUresult> {
        let mut packet: ffi::cuvid::CUVIDSOURCEDATAPACKET = unsafe { std::mem::zeroed() };
        packet.flags = (ffi::cuvid::CUvideopacketflags_CUVID_PKT_ENDOFSTREAM
            | ffi::cuvid::CUvideopacketflags_CUVID_PKT_NOTIFY_EOS) as _;

        unsafe {
            let res = ffi::cuvid::cuvidParseVideoData(self.inner.parser, &mut packet);
            wrap!(res, res)?;
        }

        Ok(())
    }

    pub fn frames<'a, 'b>(
        &'a self,
        context: Option<&'b super::cuda::context::CuContext>,
    ) -> FramesIter<'a, 'b> {
        FramesIter {
            inner: &self.inner,
            context,
        }
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe {
            ffi::cuvid::cuvidDestroyVideoParser(self.inner.parser);

            if !self.inner.decoder.is_null() {
                ffi::cuda::cuCtxPushCurrent_v2(self.inner.context.context);
                ffi::cuvid::cuvidDestroyDecoder(self.inner.decoder);
            }
            ffi::cuda::cuCtxPopCurrent_v2(std::ptr::null_mut());
            ffi::cuvid::cuvidCtxLockDestroy(self.inner.lock);
        }
    }
}

impl Inner {
    fn sequence_cb(&mut self, video_fmt: *mut ffi::cuvid::CUVIDEOFORMAT) -> i32 {
        let fmt = unsafe { &*video_fmt };

        tracing::debug!(
            "Video Input Information

            Codec: {}
            Frame Rate : {}/{}
            Sequence: {}
            Coded Size {}x{}
            Display Area: {}x{}x{}x{}
            Chroma :{}
            Bit Depth: {}
            Minimum Surfaces: {}",
            fmt.codec,
            fmt.frame_rate.numerator,
            fmt.frame_rate.denominator,
            fmt.progressive_sequence,
            fmt.coded_width,
            fmt.coded_height,
            fmt.display_area.top,
            fmt.display_area.left,
            fmt.display_area.bottom,
            fmt.display_area.right,
            fmt.chroma_format,
            fmt.bit_depth_chroma_minus8,
            fmt.min_num_decode_surfaces,
        );

        let min_surfaces = fmt.min_num_decode_surfaces;

        let mut decode_caps: ffi::cuvid::CUVIDDECODECAPS = unsafe { std::mem::zeroed() };
        decode_caps.eCodecType = fmt.codec;
        decode_caps.eChromaFormat = fmt.chroma_format;
        decode_caps.nBitDepthMinus8 = fmt.bit_depth_chroma_minus8 as _;

        unsafe {
            if !ffi::cuda::cuCtxPushCurrent_v2(self.context.context).ok() {
                return min_surfaces as _;
            }
            if !ffi::cuvid::cuvidGetDecoderCaps(&mut decode_caps).ok() {
                return min_surfaces as _;
            }
            if !ffi::cuda::cuCtxPopCurrent_v2(std::ptr::null_mut()).ok() {
                return min_surfaces as _;
            }
        }

        if decode_caps.bIsSupported == 0 {
            tracing::error!("Codec not supported on this GPU");
            return min_surfaces as _;
        }

        if (fmt.coded_width > decode_caps.nMaxWidth) || (fmt.coded_height > decode_caps.nMaxHeight)
        {
            tracing::error!(
                "Resolution {}x{} if greater than max resolution {}x{} for the GPU",
                fmt.coded_width,
                fmt.coded_height,
                decode_caps.nMaxWidth,
                decode_caps.nMaxHeight
            );
            return min_surfaces as _;
        }
        if (fmt.coded_width >> 4) * (fmt.coded_height >> 4) > decode_caps.nMaxMBCount {
            tracing::error!(
                "bitrate {} if greater than max bitrate {} for the GPU",
                (fmt.coded_width >> 4) * (fmt.coded_height >> 4),
                decode_caps.nMaxMBCount
            );

            return min_surfaces as _;
        }

        self.codec = fmt.codec.into();
        self.chroma_format = fmt.chroma_format.into();
        self.bit_depth_minus8 = fmt.bit_depth_luma_minus8;
        self.bpp = if fmt.bit_depth_luma_minus8 > 0 { 2 } else { 1 };

        if false {
            if self.chroma_format == VideoChromaFormat::YUV420
                || self.chroma_format == VideoChromaFormat::Monochrome
            {
                self.output_format = if self.bit_depth_minus8 != 0 {
                    VideoSurfaceFormat::P016
                } else {
                    VideoSurfaceFormat::NV12
                }
            } else if self.chroma_format == VideoChromaFormat::YUV444 {
                self.output_format = if self.bit_depth_minus8 != 0 {
                    VideoSurfaceFormat::YUV444_16
                } else {
                    VideoSurfaceFormat::YUV444
                }
            } else if self.chroma_format == VideoChromaFormat::YUV422 {
                self.output_format = VideoSurfaceFormat::NV12
            }

            // Check if output format supported. If not, check falback options
            if (decode_caps.nOutputFormatMask & (1 << (self.output_format as u16))) == 0 {
                if decode_caps.nOutputFormatMask & (1 << (VideoSurfaceFormat::NV12 as u16)) != 0 {
                    self.output_format = VideoSurfaceFormat::NV12;
                } else if decode_caps.nOutputFormatMask & (1 << (VideoSurfaceFormat::P016 as u16))
                    != 0
                {
                    self.output_format = VideoSurfaceFormat::P016;
                } else if decode_caps.nOutputFormatMask & (1 << (VideoSurfaceFormat::YUV444 as u16))
                    != 0
                {
                    self.output_format = VideoSurfaceFormat::YUV444;
                } else if decode_caps.nOutputFormatMask
                    & (1 << (VideoSurfaceFormat::YUV444_16 as u16))
                    != 0
                {
                    self.output_format = VideoSurfaceFormat::YUV444_16;
                } else {
                    panic!("No supported output format found");
                }
            }
        } else {
            /*
                The above ouptut format selection was copied from NvDecoder.cpp
                in the Video Codec SDK samples; however OpenCV with usage
                of GPU Mat always selects NV12
            */

            self.output_format = VideoSurfaceFormat::NV12;
            if decode_caps.nOutputFormatMask & (1 << (VideoSurfaceFormat::NV12 as u16)) == 0 {
                tracing::error!("The output format NV12 is not supported by this decoder.");
                // should we blow up here?
                return 0;
            }
        }

        self.video_fmt = Some(*fmt);
        let video_fmt = self.video_fmt.as_ref().unwrap();
        let decode_surfaces = (min_surfaces as u64).max(12);

        let mut video_decode_create_info: ffi::cuvid::CUVIDDECODECREATEINFO =
            unsafe { std::mem::zeroed() };

        video_decode_create_info.CodecType = self.codec.into();
        video_decode_create_info.ChromaFormat = self.chroma_format.into();
        video_decode_create_info.OutputFormat = self.output_format.into();
        video_decode_create_info.bitDepthMinus8 = video_fmt.bit_depth_luma_minus8 as _;
        video_decode_create_info.DeinterlaceMode = if video_fmt.progressive_sequence != 0 {
            ffi::cuvid::cudaVideoDeinterlaceMode_enum_cudaVideoDeinterlaceMode_Weave
        } else {
            ffi::cuvid::cudaVideoDeinterlaceMode_enum_cudaVideoDeinterlaceMode_Adaptive
        };
        video_decode_create_info.ulNumOutputSurfaces = 3;
        video_decode_create_info.ulCreationFlags =
            ffi::cuvid::cudaVideoCreateFlags_enum_cudaVideoCreate_PreferCUVID as _;
        video_decode_create_info.ulNumDecodeSurfaces = decode_surfaces;
        video_decode_create_info.vidLock = self.lock;
        video_decode_create_info.ulWidth = video_fmt.coded_width as _;
        video_decode_create_info.ulHeight = video_fmt.coded_height as _;
        video_decode_create_info.ulMaxWidth = video_fmt.coded_width as _;
        video_decode_create_info.ulMaxHeight = video_fmt.coded_height as _;
        video_decode_create_info.ulTargetWidth = video_fmt.coded_width as _;
        video_decode_create_info.ulTargetHeight = video_fmt.coded_height as _;
        video_decode_create_info.ulIntraDecodeOnly = if self.keyframe_only { 1 } else { 0 };

        if self.requested_size.0 > 0 && self.requested_size.1 > 0 {
            video_decode_create_info.display_area.left = video_fmt.display_area.left as _;
            video_decode_create_info.display_area.top = video_fmt.display_area.top as _;
            video_decode_create_info.display_area.right = video_fmt.display_area.right as _;
            video_decode_create_info.display_area.bottom = video_fmt.display_area.bottom as _;

            self.out_size = self.requested_size;
            video_decode_create_info.ulTargetWidth = self.out_size.0 as _;
            video_decode_create_info.ulTargetHeight = self.out_size.1 as _;
            self.coded_size = self.out_size;
        } else {
            self.out_size.0 = (video_fmt.display_area.right - video_fmt.display_area.left) as _;
            self.out_size.1 = (video_fmt.display_area.bottom - video_fmt.display_area.top) as _;
            self.coded_size = (video_fmt.coded_width, video_fmt.coded_height);
        }
        unsafe {
            if !ffi::cuda::cuCtxPushCurrent_v2(self.context.context).ok() {
                return min_surfaces as _;
            }
            if !ffi::cuvid::cuvidCreateDecoder(&mut self.decoder, &mut video_decode_create_info)
                .ok()
            {
                return min_surfaces as _;
            }
            if !ffi::cuda::cuCtxPopCurrent_v2(std::ptr::null_mut()).ok() {
                return min_surfaces as _;
            }
        }

        return decode_surfaces as _;
    }

    fn picture_decode_cb(&self, pic_params: *mut ffi::cuvid::CUVIDPICPARAMS) -> i32 {
        if self.decoder.is_null() {
            tracing::error!("picture_decode_cb called but decoder is not initialized.");
            return 0;
        }
        unsafe {
            if !ffi::cuda::cuCtxPushCurrent_v2(self.context.context).ok() {
                return 0;
            }
            if !ffi::cuvid::cuvidDecodePicture(self.decoder, pic_params).ok() {
                return 0;
            }
            // low latency option
            if !ffi::cuda::cuCtxPopCurrent_v2(std::ptr::null_mut()).ok() {
                return 0;
            }
        }

        1
    }

    fn picture_display_cb(&mut self, display_info: *mut ffi::cuvid::CUVIDPARSERDISPINFO) -> i32 {
        if display_info.is_null() {
            drop(self.sender.take());
            return 1;
        }
        if self.sender.is_none() {
            return 1;
        }
        let display_info = unsafe { &*display_info };
        let video_processing_parameters = {
            let mut video_processing_parameters: ffi::cuvid::CUVIDPROCPARAMS =
                unsafe { std::mem::zeroed() };
            video_processing_parameters.progressive_frame = display_info.progressive_frame;
            video_processing_parameters.second_field = display_info.repeat_first_field + 1;
            video_processing_parameters.top_field_first = display_info.top_field_first;
            video_processing_parameters.unpaired_field =
                (display_info.repeat_first_field < 0) as i32;

            video_processing_parameters
        };

        let res = self.sender.as_ref().unwrap().send(PreparedFrame {
            index: display_info.picture_index,
            parameters: video_processing_parameters,
            timestamp: display_info.timestamp,
        });
        if let Err(_) = res {
            return 0;
        }
        return 1;
    }

    fn operating_point_cb(&self, _op_info: *mut ffi::cuvid::CUVIDOPERATINGPOINTINFO) -> i32 {
        0
    }
}

pub struct FramesIter<'a, 'b> {
    inner: &'a Inner,
    context: Option<&'b super::cuda::context::CuContext>,
}

impl<'a, 'b> Iterator for FramesIter<'a, 'b> {
    type Item = GpuFrame;

    fn next(&mut self) -> Option<Self::Item> {
        let mut frame = self.inner.receiver.recv().ok()?;

        let mut dp_src_frame: CUdeviceptr = 0;
        let mut n_src_pitch = 0u32;

        unsafe {
            if !ffi::cuda::cuCtxPushCurrent_v2(
                self.context
                    .map(|c| c.context)
                    .unwrap_or(self.inner.context.context),
            )
            .ok()
            {
                tracing::error!("Failed to push current context.");
                return None;
            }
            // tracing::info!("{}: {}", context.is_some(), frame.index);
            if let Err(err) = ffi::cuvid::cuvidMapVideoFrame64(
                self.inner.decoder,
                frame.index,
                &mut dp_src_frame,
                &mut n_src_pitch,
                &mut frame.parameters,
            )
            .err()
            {
                tracing::error!("Failed to map video frame: {}", err);
                return None;
            }

            let mut decode_status: ffi::cuvid::CUVIDGETDECODESTATUS = std::mem::zeroed();

            if ffi::cuvid::cuvidGetDecodeStatus(self.inner.decoder, frame.index, &mut decode_status)
                .ok()
            {
                if decode_status.decodeStatus
                    == ffi::cuvid::cuvidDecodeStatus_enum_cuvidDecodeStatus_Error
                    || decode_status.decodeStatus
                        == ffi::cuvid::cuvidDecodeStatus_enum_cuvidDecodeStatus_Error_Concealed
                {
                    tracing::error!("Decoding error occured");
                }
            }
        }

        let frame = GpuFrame {
            width: self.inner.out_size.0,
            height: self.inner.out_size.1,
            ptr: dp_src_frame,
            pitch: n_src_pitch,
            timestamp: frame.timestamp(),
            decoder: self.inner.decoder,
        };

        Some(frame)
    }
}

pub unsafe extern "C" fn handle_video_sequence_proc(
    user_data: *mut std::os::raw::c_void,
    video_format: *mut ffi::cuvid::CUVIDEOFORMAT,
) -> i32 {
    let decoder = user_data as *mut Inner;
    let decoder = &mut *decoder;

    decoder.sequence_cb(video_format)
}

pub unsafe extern "C" fn handle_picture_decode_proc(
    user_data: *mut std::os::raw::c_void,
    pic_params: *mut ffi::cuvid::CUVIDPICPARAMS,
) -> i32 {
    let decoder = user_data as *mut Inner;
    let decoder = &*decoder;

    decoder.picture_decode_cb(pic_params)
}

pub unsafe extern "C" fn handle_picture_display_proc(
    user_data: *mut std::os::raw::c_void,
    display_info: *mut ffi::cuvid::CUVIDPARSERDISPINFO,
) -> i32 {
    let decoder = user_data as *mut Inner;
    let decoder = &mut *decoder;

    decoder.picture_display_cb(display_info)
}

pub unsafe extern "C" fn handle_operating_point_proc(
    user_data: *mut std::os::raw::c_void,
    op_info: *mut ffi::cuvid::CUVIDOPERATINGPOINTINFO,
) -> i32 {
    let decoder = user_data as *mut Inner;
    let decoder = &*decoder;

    decoder.operating_point_cb(op_info)
}
