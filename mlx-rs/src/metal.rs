//! Metal GPU memory management.
//!
//! These functions control the MLX Metal buffer allocator cache, which holds
//! onto GPU memory allocations for reuse. Without a cache limit, the allocator
//! will grow unbounded over many inference calls.
//!
//! # Example
//!
//! ```no_run
//! // Cap the Metal buffer cache at 2 GB
//! quill_mlx::metal::set_cache_limit(2 * 1024 * 1024 * 1024);
//! ```

use crate::error::Exception;
use crate::utils::SUCCESS;
use std::panic::Location;

/// Initialize the error handler, then run an FFI call and convert failures.
#[track_caller]
fn call_ffi(f: impl FnOnce() -> i32) -> Result<(), Exception> {
    crate::error::INIT_ERR_HANDLER
        .with(|init| init.call_once(crate::error::setup_mlx_error_handler));

    match f() {
        SUCCESS => Ok(()),
        _ => {
            let what = crate::error::get_and_clear_last_mlx_error()
                .expect("MLX operation failed but no error was set")
                .what;
            Err(Exception {
                what,
                location: Location::caller(),
            })
        }
    }
}

/// Set the Metal buffer cache limit in bytes.
///
/// Returns the previous cache limit. A limit of 0 means no caching.
#[track_caller]
pub fn set_cache_limit(limit: usize) -> Result<usize, Exception> {
    let mut prev: usize = 0;
    call_ffi(|| unsafe { quill_mlx_sys::mlx_set_cache_limit(&mut prev as *mut _, limit) })?;
    Ok(prev)
}

/// Clear the Metal buffer cache, releasing all cached GPU memory back to the system.
#[track_caller]
pub fn clear_cache() -> Result<(), Exception> {
    call_ffi(|| unsafe { quill_mlx_sys::mlx_clear_cache() })
}

/// Get the amount of actively used GPU memory in bytes.
#[track_caller]
pub fn get_active_memory() -> Result<usize, Exception> {
    let mut res: usize = 0;
    call_ffi(|| unsafe { quill_mlx_sys::mlx_get_active_memory(&mut res as *mut _) })?;
    Ok(res)
}

/// Get the amount of GPU memory held in the buffer cache in bytes.
#[track_caller]
pub fn get_cache_memory() -> Result<usize, Exception> {
    let mut res: usize = 0;
    call_ffi(|| unsafe { quill_mlx_sys::mlx_get_cache_memory(&mut res as *mut _) })?;
    Ok(res)
}

/// Get the peak GPU memory usage in bytes since the last reset.
#[track_caller]
pub fn get_peak_memory() -> Result<usize, Exception> {
    let mut res: usize = 0;
    call_ffi(|| unsafe { quill_mlx_sys::mlx_get_peak_memory(&mut res as *mut _) })?;
    Ok(res)
}

/// Reset the peak memory counter to zero.
#[track_caller]
pub fn reset_peak_memory() -> Result<(), Exception> {
    call_ffi(|| unsafe { quill_mlx_sys::mlx_reset_peak_memory() })
}

/// Set the total GPU memory limit in bytes.
///
/// Returns the previous memory limit.
#[track_caller]
pub fn set_memory_limit(limit: usize) -> Result<usize, Exception> {
    let mut prev: usize = 0;
    call_ffi(|| unsafe { quill_mlx_sys::mlx_set_memory_limit(&mut prev as *mut _, limit) })?;
    Ok(prev)
}

/// Get the current GPU memory limit in bytes.
#[track_caller]
pub fn get_memory_limit() -> Result<usize, Exception> {
    let mut res: usize = 0;
    call_ffi(|| unsafe { quill_mlx_sys::mlx_get_memory_limit(&mut res as *mut _) })?;
    Ok(res)
}

/// Set the wired memory limit in bytes (Metal-specific).
///
/// Returns the previous wired limit.
#[track_caller]
pub fn set_wired_limit(limit: usize) -> Result<usize, Exception> {
    let mut prev: usize = 0;
    call_ffi(|| unsafe { quill_mlx_sys::mlx_set_wired_limit(&mut prev as *mut _, limit) })?;
    Ok(prev)
}

/// Check whether Metal is available on this device.
pub fn is_available() -> bool {
    let mut res = false;
    let status = unsafe { quill_mlx_sys::mlx_metal_is_available(&mut res as *mut _) };
    matches!(status, SUCCESS) && res
}
