// TODO do w/out the unions?
// #![feature(untagged_unions)]

#[allow(improper_ctypes, clashing_extern_declarations)]
pub mod cuda;
#[allow(improper_ctypes, clashing_extern_declarations)]
pub mod cuvid;
#[allow(improper_ctypes)]
pub mod npp;

#[cfg(test)]
mod tests {
    use super::cuda::*;
    #[test]
    fn init_and_version() {
        let ret = unsafe { cuInit(0) };
        println!("{:?}", ret);

        let ver = unsafe {
            let mut ver = 0;
            cuDriverGetVersion(&mut ver as *mut i32);
            ver
        };

        println!("Version {}", ver);
    }
}
