//! Regression check for NVIDIA bug 6610402: destroying the first NVENC session
//! a process opened, while others are still open, makes libnvcuvid segfault a
//! few seconds later (a worker thread started by that session keeps using it).
//! See `cuvid::anchor` and
//! <https://forums.developer.nvidia.com/t/regression-with-use-after-free-linux-drivers-after-520-56-06-inside-nvenc/269974>.
//!
//!   cargo run --example nvenc_first_session_uaf             # segfaults on affected drivers
//!   cargo run --example nvenc_first_session_uaf -- --anchor # must print "survived" and exit 0
//!
//! Without `--anchor` on a fixed driver it also survives; that is how to tell
//! whether the anchor is still needed.

extern crate nvidia_video_codec;

use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use nvidia_video_codec as nv;
use nv::cuda::context::CuContext;
use nv::cuda::device::CuDevice;
use nv::cuvid::{anchor, Bitrate, Codec, Encoder};

/// Sessions besides the anchor: the first is destroyed, the rest keep the
/// worker thread alive.
const SESSIONS: usize = 3;
/// Long enough to cover the crash however long the driver waits: about 1s in
/// the forum report, up to ~21s measured here with sessions that had encoded.
const WAIT: Duration = Duration::from_secs(25);

fn main() {
    let with_anchor = std::env::args().any(|a| a == "--anchor");
    nv::init();

    if with_anchor {
        anchor::open(0).expect("open NVENC anchor");
        println!("anchor open");
    }

    /* Shared by every session; never dropped. */
    let context: &'static CuContext = Box::leak(Box::new(
        CuContext::new(CuDevice::new(0).expect("CUDA device 0"), 0).expect("CUDA context"),
    ));
    let open = |i: usize| {
        Encoder::create(
            0,
            Some(context),
            Codec::H264,
            Bitrate::CQP(51),
            (256, 256),
            (30, 1),
            NonZeroUsize::MIN,
            false,
        )
        .unwrap_or_else(|err| panic!("open NVENC session {i}: {err}"))
    };
    let mut sessions: Vec<Encoder> = (0..SESSIONS).map(open).collect();
    println!("opened {SESSIONS} sessions");

    /* The trigger: the first-opened session goes while the others are open. */
    drop(sessions.remove(0));
    println!("destroyed the first session; waiting {}s with {} open", WAIT.as_secs(), sessions.len());
    let start = Instant::now();
    while start.elapsed() < WAIT {
        std::thread::sleep(Duration::from_secs(1));
        println!("  {}s", start.elapsed().as_secs());
    }

    /* Reverse order is safe; every session has to be closed before exit,
    or libnvcuvid's static destructor deadlocks. */
    while let Some(session) = sessions.pop() {
        drop(session);
    }
    assert_eq!(anchor::other_open_sessions(), 0);
    if with_anchor {
        anchor::close_all();
    }
    println!("survived");
}
