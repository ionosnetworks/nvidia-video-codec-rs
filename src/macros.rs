macro_rules! wrap {
    ($val:ident, $res:ident) => {
        if $res == ffi::cuda::cudaError_enum_CUDA_SUCCESS {
            Ok($val)
        } else {
            Err($res)
        }
    };
}
