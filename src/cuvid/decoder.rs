use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use super::{ffi, CudaResult, GpuFrame};

pub use ffi::cuvid::CUdeviceptr;

pub use super::chroma::VideoChromaFormat;
pub use super::codec::Codec;
pub use super::surface::VideoSurfaceFormat;

pub const ADDITIONAL_DECODE_SURFACES: usize = 3;

pub struct Decoder {
    inner: Box<Inner>,
}

unsafe impl Send for Decoder {}
unsafe impl Sync for Decoder {}

struct Inner {
    parser: ffi::cuvid::CUvideoparser,
    lock: ffi::cuvid::CUvideoctxlock,
    context: super::super::cuda::context::CuContextRef<'static>,
    decoder: ffi::cuvid::CUvideodecoder,
    keyframe_only: bool,
    requested_size: (u32, u32),
    frame_in_use: Arc<AtomicU64>,
    frames_in_flight: Arc<(Mutex<usize>, Condvar)>,

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
    requested_output_surfaces: Option<usize>,
    requested_decode_surfaces: Option<usize>,
    current_output_surfaces: usize,
    frame_timeout: Option<Duration>,
    name: Option<String>,
}

#[derive(Debug)]
struct PreparedFrame {
    timestamp: i64,
    index: i32,
    parameters: ffi::cuvid::CUVIDPROCPARAMS,
}

unsafe impl Send for PreparedFrame {}
unsafe impl Sync for PreparedFrame {}

impl PreparedFrame {
    fn timestamp(&self) -> i64 {
        self.timestamp
    }
}

impl Decoder {
    pub fn create(
        gpu_id: usize,
        context: Option<&'static super::super::cuda::context::CuContext>,
        codec: Codec,
        keyframe_only: bool,
        low_latency: bool,
        output_size: (u32, u32),
        decode_surfaces: Option<usize>,
        output_surfaces: Option<usize>,
        frame_timeout: Option<Duration>,
        picture_buffer: Option<usize>,
    ) -> Result<Self, ffi::cuda::CUresult> {
        let context = match context {
            Some(context) => super::super::cuda::context::CuContextRef::Borrowed(context),
            None => {
                let device = super::super::cuda::device::CuDevice::new(gpu_id as _)?;
                let context = super::super::cuda::context::CuContext::new(device, 0)?;
                super::super::cuda::context::CuContextRef::Owned(context)
            }
        };

        let mut parser: ffi::cuvid::CUvideoparser = std::ptr::null_mut();
        let mut ctx_lock: ffi::cuvid::CUvideoctxlock = std::ptr::null_mut();

        unsafe {
            let res = ffi::cuvid::cuvidCtxLockCreate(&mut ctx_lock, context.context as _);
            wrap!(res, res)?;
        }
        let (sender, receiver) = match picture_buffer {
            Some(buf) => flume::bounded(buf),
            None => flume::unbounded(),
        };

        let mut inner = Box::new(Inner {
            parser,
            context,
            codec,
            lock: ctx_lock,
            chroma_format: VideoChromaFormat::Monochrome,
            decoder: std::ptr::null_mut(),
            frame_in_use: Default::default(),
            frames_in_flight: Arc::new((Mutex::new(0), Condvar::new())),
            keyframe_only,
            video_fmt: None,
            bit_depth_minus8: 0,
            bpp: 0,
            output_format: VideoSurfaceFormat::NV12,
            out_size: (0, 0),
            coded_size: (0, 0),
            requested_size: output_size,
            receiver,
            requested_output_surfaces: output_surfaces,
            requested_decode_surfaces: decode_surfaces,
            sender: Some(sender),
            frame_timeout,
            name: None,
            current_output_surfaces: 0,
        });

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

        unsafe {
            let res = ffi::cuvid::cuvidCreateVideoParser(&mut parser, &mut params);
            wrap!(res, res)?;
        }
        inner.parser = parser;

        Ok(Self { inner })
    }

    pub fn set_name<T: AsRef<str>>(&mut self, name: T) {
        self.inner.name = Some(String::from(name.as_ref()));
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
        context: Option<&'b super::super::cuda::context::CuContext>,
    ) -> FramesIter<'a, 'b> {
        FramesIter {
            inner: &self.inner,
            frame_timeout: self.inner.frame_timeout,
            context,
        }
    }

    #[cfg(feature = "async")]
    pub fn stream<'a, 'b>(
        &'a self,
        context: Option<&'b super::super::cuda::context::CuContext>,
    ) -> impl futures::Stream<Item = GpuFrame> + use<'a, 'b> {
        use futures::StreamExt;
        let frame_timeout = self.inner.frame_timeout;
        self.inner
            .receiver
            .stream()
            .map(move |frame| {
                let f = FramesIter {
                    inner: &self.inner,
                    frame_timeout,
                    context,
                };
                f.map_frame(frame)
            })
            .take_while(|f| futures::future::ready(f.is_some()))
            .map(|f| f.unwrap())
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        {
            let (lock, cvar) = &*self.inner.frames_in_flight;
            let mut count = lock.lock().unwrap();
            while *count > 0 {
                count = cvar.wait(count).unwrap();
            }
        }
        unsafe {
            if !self.inner.decoder.is_null() {
                ffi::cuda::cuCtxPushCurrent_v2(self.inner.context.context);
                ffi::cuvid::cuvidDestroyDecoder(self.inner.decoder);
                self.inner.decoder = std::ptr::null_mut();
                ffi::cuda::cuCtxPopCurrent_v2(std::ptr::null_mut());
            }
            ffi::cuvid::cuvidDestroyVideoParser(self.inner.parser);
            self.inner
                .frame_in_use
                .store(0, std::sync::atomic::Ordering::SeqCst);
            ffi::cuvid::cuvidCtxLockDestroy(self.inner.lock);
        }
    }
}

impl Inner {
    fn is_frame_in_use(&self, idx: usize) -> bool {
        let f = self.frame_in_use.load(std::sync::atomic::Ordering::SeqCst);
        f & (1 << idx) != 0
    }

    fn set_frame_status(&self, idx: usize, status: bool) {
        if status {
            let v = 1 << idx;
            self.frame_in_use
                .fetch_or(v, std::sync::atomic::Ordering::SeqCst);
        } else {
            let v = !(1 << idx);
            self.frame_in_use
                .fetch_and(v, std::sync::atomic::Ordering::SeqCst);
        }
    }

    fn sequence_cb(&mut self, video_fmt: *mut ffi::cuvid::CUVIDEOFORMAT) -> i32 {
        let fmt = unsafe { &*video_fmt };

        tracing::debug!(
            name = self.name.as_deref().unwrap_or_default(),
            "Video Input Information

            Status: {},
            Codec: {}
            Frame Rate : {}/{}
            Sequence: {}
            Coded Size {}x{}
            Display Area: {}x{}x{}x{}
            Chroma :{}
            Bit Depth: {}
            Minimum Surfaces: {}",
            if self.decoder.is_null() {
                "New"
            } else {
                "Reconfigure"
            },
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

        /*
           if you use exactly min_num_decode_surfaces, the moment the decoder
           needs a new surface but all are either locked for display or
           in flight, it stalls. By adding two or three extra surfaces,
           you give the pipeline enough headroom to keep decoding while some
           frames remain undisplayed or are still being transferred out for
           rendering. Without that extra buffer, it’s easy to see stuttering or
            stalls, since the GPU can’t proceed until one of the
            “minimum” surfaces has been freed.
        */
        let min_surfaces = fmt.min_num_decode_surfaces + (ADDITIONAL_DECODE_SURFACES as u8);

        let mut decode_caps: ffi::cuvid::CUVIDDECODECAPS = unsafe { std::mem::zeroed() };
        decode_caps.eCodecType = fmt.codec;
        decode_caps.eChromaFormat = fmt.chroma_format;
        decode_caps.nBitDepthMinus8 = fmt.bit_depth_chroma_minus8 as _;

        unsafe {
            if !ffi::cuda::cuCtxPushCurrent_v2(self.context.context).ok() {
                return min_surfaces as _;
            }
            if !ffi::cuvid::cuvidGetDecoderCaps(&mut decode_caps).ok() {
                tracing::error!("Failed to get decode caps");
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
        let mut force_recreate = true;
        if !self.decoder.is_null() {
            if self.bit_depth_minus8 != fmt.bit_depth_luma_minus8 {
                tracing::warn!("Reconfigure Not supported for bit depth change");
                force_recreate = true;
            }
            if self.chroma_format != fmt.chroma_format.into() {
                tracing::warn!("Reconfigure Not supported for chroma format change");
                force_recreate = true;
            }
        }
        let res_change =
            !(fmt.coded_width == self.coded_size.0 && fmt.coded_height == self.coded_size.1);
        /*
        let rect_change = !(pVideoFormat->display_area.bottom ==
                                  p_impl->m_videoFormat.display_area.bottom &&
                              pVideoFormat->display_area.top ==
                                  p_impl->m_videoFormat.display_area.top &&
                              pVideoFormat->display_area.left ==
                                  p_impl->m_videoFormat.display_area.left &&
                              pVideoFormat->display_area.right ==
                                  p_impl->m_videoFormat.display_area.right);
        */
        let rect_change = false; // TODO(nemosupremo)

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
        let decode_surfaces =
            (min_surfaces as u64).max(self.requested_decode_surfaces.unwrap_or(1) as u64);

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
        video_decode_create_info.ulNumOutputSurfaces = self
            .requested_output_surfaces
            .map(|n| n as u64)
            .map(|n| if n == 0 { decode_surfaces } else { n })
            .unwrap_or(3);
        video_decode_create_info.ulCreationFlags =
            ffi::cuvid::cudaVideoCreateFlags_enum_cudaVideoCreate_PreferCUVID as _;
        video_decode_create_info.ulNumDecodeSurfaces = decode_surfaces;
        video_decode_create_info.vidLock = self.lock;
        video_decode_create_info.ulWidth = video_fmt.coded_width as _;
        video_decode_create_info.ulHeight = video_fmt.coded_height as _;
        video_decode_create_info.ulMaxWidth = video_fmt.coded_width as _;
        video_decode_create_info.ulMaxHeight = video_fmt.coded_height as _;
        video_decode_create_info.ulTargetWidth =
            (video_fmt.display_area.right - video_fmt.display_area.left) as _;
        video_decode_create_info.ulTargetHeight =
            (video_fmt.display_area.bottom - video_fmt.display_area.top) as _;
        video_decode_create_info.ulIntraDecodeOnly = if self.keyframe_only { 1 } else { 0 };

        self.current_output_surfaces = video_decode_create_info.ulNumOutputSurfaces as _;
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
            if force_recreate {
                {
                    let (lock, cvar) = &*self.frames_in_flight;
                    let mut count = lock.lock().unwrap();
                    while *count > 0 {
                        count = cvar.wait(count).unwrap();
                    }
                }
                ffi::cuvid::cuvidDestroyDecoder(self.decoder);
                self.decoder = std::ptr::null_mut();
            }

            if self.decoder.is_null() {
                if !ffi::cuvid::cuvidCreateDecoder(&mut self.decoder, &mut video_decode_create_info)
                    .ok()
                {
                    tracing::error!("Failed to create decoder");
                    return min_surfaces as _;
                }
            } else {
                if !res_change {
                    if rect_change {
                        // TODO(nemosupremo)
                    }
                } else {
                    let mut video_decode_reconfigure_info: ffi::cuvid::CUVIDRECONFIGUREDECODERINFO =
                        std::mem::zeroed();
                    video_decode_reconfigure_info.ulWidth = video_fmt.coded_width as _;
                    video_decode_reconfigure_info.ulHeight = video_fmt.coded_height as _;
                    video_decode_reconfigure_info.ulTargetWidth = video_fmt.coded_width as _;
                    video_decode_reconfigure_info.ulTargetHeight = video_fmt.coded_height as _;

                    if self.requested_size.0 > 0 && self.requested_size.1 > 0 {
                        video_decode_reconfigure_info.display_area.left =
                            video_fmt.display_area.left as _;
                        video_decode_reconfigure_info.display_area.top =
                            video_fmt.display_area.top as _;
                        video_decode_reconfigure_info.display_area.right =
                            video_fmt.display_area.right as _;
                        video_decode_reconfigure_info.display_area.bottom =
                            video_fmt.display_area.bottom as _;

                        video_decode_reconfigure_info.ulTargetWidth = self.out_size.0 as _;
                        video_decode_reconfigure_info.ulTargetHeight = self.out_size.1 as _;
                    }

                    video_decode_reconfigure_info.ulNumDecodeSurfaces = decode_surfaces as _;

                    if let Err(err) = ffi::cuvid::cuvidReconfigureDecoder(
                        self.decoder,
                        &mut video_decode_reconfigure_info,
                    )
                    .err()
                    {
                        tracing::error!("Failed to reconfigure decoder: {}", err);
                        return min_surfaces as _;
                    }
                }
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
        let pic_idx = {
            let pic_params = unsafe { &*pic_params };
            pic_params.CurrPicIdx as usize
        };
        if pic_idx >= 64 {
            panic!("didn't expect pic_idx to be more than 64")
        }
        let start = std::time::Instant::now();
        let mut warned = false;
        while self.is_frame_in_use(pic_idx) {
            if start.elapsed() > std::time::Duration::from_secs(5) && !warned {
                tracing::warn!(
                    idx = pic_idx,
                    "Waited way too long for frame to become free."
                );
                warned = true;
            }
            std::thread::sleep(std::time::Duration::from_micros(500));
        }
        if start.elapsed() > std::time::Duration::from_secs(5) {
            tracing::warn!(
                idx = pic_idx,
                "Waited {}ms for frame to become free.",
                start.elapsed().as_millis()
            );
        }
        if self.decoder.is_null() {
            tracing::debug!("decoder was dropped while waiting for frame in use.");
            return 0;
        }
        self.set_frame_status(pic_idx, true);

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

        let sender = self.sender.as_ref().unwrap();
        //if sender.is_full() && sender.capacity().unwrap() > 0 {
        // tracing::warn!("picture display cb is full");
        //}
        let res = sender.send(PreparedFrame {
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
    context: Option<&'b super::super::cuda::context::CuContext>,
    frame_timeout: Option<Duration>,
}

impl<'a, 'b> FramesIter<'a, 'b> {
    pub fn next_timeout(&mut self, timeout: Duration) -> Result<Option<GpuFrame>, ()> {
        let frame = self.inner.receiver.recv_timeout(timeout).map_err(|_| ())?;

        Ok(self.map_frame(frame))
    }

    fn map_frame(&self, mut frame: PreparedFrame) -> Option<GpuFrame> {
        let mut dp_src_frame: CUdeviceptr = 0;
        let mut n_src_pitch = 0u32;
        let mut has_concealed_error = None;
        let context = self
            .context
            .map(|c| c.context)
            .unwrap_or(self.inner.context.context);

        unsafe {
            if !ffi::cuda::cuCtxPushCurrent_v2(context).ok() {
                tracing::error!("Failed to push current context.");
                ffi::cuda::cuCtxPopCurrent_v2(std::ptr::null_mut());
                return None;
            }

            // tracing::info!("{}: {}", context.is_some(), frame.index);
            {
                let (lock, cvar) = &*self.inner.frames_in_flight;
                let mut count = lock.lock().unwrap();
                while *count >= self.inner.current_output_surfaces {
                    count = cvar.wait(count).unwrap();
                }
                *count = *count + 1;
                cvar.notify_one();
            }
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
                ffi::cuda::cuCtxPopCurrent_v2(std::ptr::null_mut());
                return None;
            }

            let mut decode_status: ffi::cuvid::CUVIDGETDECODESTATUS = std::mem::zeroed();
            if ffi::cuvid::cuvidGetDecodeStatus(self.inner.decoder, frame.index, &mut decode_status)
                .ok()
            {
                if decode_status.decodeStatus
                    == ffi::cuvid::cuvidDecodeStatus_enum_cuvidDecodeStatus_Error
                {
                    tracing::warn!(concealed = false, "Decoding error occured");
                    has_concealed_error = Some(false);
                } else if decode_status.decodeStatus
                    == ffi::cuvid::cuvidDecodeStatus_enum_cuvidDecodeStatus_Error_Concealed
                {
                    tracing::warn!(concealed = true, "Decoding error occured");
                    has_concealed_error = Some(true);
                }
            }
            if !ffi::cuda::cuCtxPopCurrent_v2(std::ptr::null_mut()).ok() {
                tracing::error!("Failed to pop current context.");
            }
        }

        let frame = GpuFrame {
            width: self.inner.out_size.0,
            height: self.inner.out_size.1,
            ptr: dp_src_frame,
            pitch: n_src_pitch,
            timestamp: frame.timestamp(),
            decoder: self.inner.decoder,
            idx: frame.index,
            has_concealed_error,
            frame_in_use: Arc::clone(&self.inner.frame_in_use),
            frames_in_flight: Arc::clone(&self.inner.frames_in_flight),
            context,
        };

        Some(frame)
    }
}

impl<'a, 'b> Iterator for FramesIter<'a, 'b> {
    type Item = GpuFrame;

    fn next(&mut self) -> Option<Self::Item> {
        let frame = match self.frame_timeout {
            Some(timeout) => self.inner.receiver.recv_timeout(timeout).ok()?,
            None => self.inner.receiver.recv().ok()?,
        };

        self.map_frame(frame)
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
