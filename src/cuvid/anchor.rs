//! A process-lifetime NVENC session, the workaround for NVIDIA bug 6610402.
//!
//! Since driver 520.56.06, destroying the first NVENC session a process opened,
//! while another session is still alive, leaves a libnvcuvid worker thread
//! (started by that first session) holding a pointer into it; the thread later
//! reads it and segfaults. Opening a session first and keeping it until every
//! other session is gone means the first session is never the one that goes
//! away while others are open.
//!
//! The anchor must still be closed before the process exits: with any NVENC
//! session open, libnvcuvid's static destructor joins that worker thread, which
//! only stops once every session is closed, and exit deadlocks.
//!
//! <https://forums.developer.nvidia.com/t/regression-with-use-after-free-linux-drivers-after-520-56-06-inside-nvenc/269974>

use std::num::NonZeroUsize;
use std::sync::atomic::Ordering;
use std::sync::Mutex;

use super::encoder::OPEN_SESSIONS;
use super::{Bitrate, Codec, Encoder};
use crate::cuda::context::CuContext;
use crate::cuda::device::CuDevice;

/// Small enough to cost next to nothing, comfortably above NVENC's minimum.
const ANCHOR_SIZE: u32 = 256;

/// The anchor session of each GPU that has one.
static ANCHORS: Mutex<Vec<(usize, Encoder)>> = Mutex::new(Vec::new());

/// Open the anchor session on `gpu_id`, once; later calls for the same GPU are
/// no-ops until [`close_all`].
///
/// It only helps if it is the first NVENC session in the process, so call it
/// before anything else can open one, GStreamer's NVENC elements included. It
/// runs on the device's primary context, which is retained and never released.
/// It occupies one of the GPU's concurrent NVENC sessions until closed.
pub fn open(gpu_id: usize) -> Result<(), crate::CudaError> {
    let mut anchors = ANCHORS.lock().unwrap();
    if anchors.iter().any(|(id, _)| *id == gpu_id) {
        return Ok(());
    }

    crate::init();
    let device = CuDevice::new(gpu_id as _)?;
    let context: &'static CuContext = CuContext::leak_primary(&device)?;
    let encoder = Encoder::create(
        gpu_id,
        Some(context),
        Codec::H264,
        Bitrate::CQP(51),
        (ANCHOR_SIZE, ANCHOR_SIZE),
        (1, 1),
        NonZeroUsize::MIN,
        false,
    )?;

    anchors.push((gpu_id, encoder));
    Ok(())
}

/// NVENC sessions this crate has open, not counting the anchors. Sessions
/// opened by other libraries (GStreamer's NVENC elements, for one) are not
/// visible here.
pub fn other_open_sessions() -> usize {
    let anchors = ANCHORS.lock().unwrap().len();
    OPEN_SESSIONS.load(Ordering::SeqCst).saturating_sub(anchors)
}

/// Close every anchor session; returns how many there were.
///
/// Call it last, once every other NVENC session is closed (see
/// [`other_open_sessions`]): closing the anchor while another session is open
/// is exactly what triggers the bug.
pub fn close_all() -> usize {
    let anchors = std::mem::take(&mut *ANCHORS.lock().unwrap());
    let closed = anchors.len();
    /* Encoder's drop destroys the session before returning. */
    drop(anchors);
    closed
}
