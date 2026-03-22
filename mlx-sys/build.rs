extern crate cmake;

use cmake::Config;
use std::process::Command;
use std::{env, fs, path::PathBuf};

/// Determine a stable directory for the MLX metallib that survives `cargo install`
#[cfg(feature = "metal")]
fn stable_metallib_dir() -> PathBuf {
    let version = env!("CARGO_PKG_VERSION");
    let home = env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home)
        .join(".mlx")
        .join("lib")
        .join(format!("v{}", version))
}

fn build_and_link_mlx_c() {
    let mut config = Config::new("src/mlx-c");
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");

    #[cfg(debug_assertions)]
    {
        config.define("CMAKE_BUILD_TYPE", "Debug");
    }

    #[cfg(not(debug_assertions))]
    {
        config.define("CMAKE_BUILD_TYPE", "Release");
    }

    config.define("MLX_BUILD_METAL", "OFF");
    config.define("MLX_BUILD_ACCELERATE", "OFF");

    #[cfg(feature = "metal")]
    {
        config.define("MLX_BUILD_METAL", "ON");

        let metallib_dir = stable_metallib_dir();
        fs::create_dir_all(&metallib_dir).expect("Failed to create stable metallib directory");
        config.define("MLX_METAL_PATH", metallib_dir.to_str().unwrap());
    }

    #[cfg(feature = "accelerate")]
    {
        config.define("MLX_BUILD_ACCELERATE", "ON");
    }

    // build the mlx-c project
    let dst = config.build();

    println!(
        "cargo:rustc-link-search=native={}/build/lib",
        dst.display()
    );
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=static=mlxc");

    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");

    #[cfg(feature = "metal")]
    {
        println!("cargo:rustc-link-lib=framework=Metal");

        let metallib_dir = stable_metallib_dir();
        let target_metallib = metallib_dir.join("mlx.metallib");
        if !target_metallib.exists() {
            eprintln!(
                "cargo:warning=mlx.metallib not found at {}. Runtime Metal operations may fail.",
                target_metallib.display()
            );
        }
    }

    #[cfg(feature = "accelerate")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}

fn main() {
    build_and_link_mlx_c();

    // generate bindings
    let mut builder = bindgen::Builder::default()
        .rust_target("1.73.0".parse().expect("rust-version"))
        .header("src/mlx-c/mlx/c/mlx.h")
        .header("src/mlx-c/mlx/c/linalg.h")
        .header("src/mlx-c/mlx/c/error.h")
        .header("src/mlx-c/mlx/c/transforms_impl.h")
        .clang_arg("-Isrc/mlx-c");

    // Add macOS SDK include path so bindgen can find stdlib.h
    if let Ok(output) = Command::new("xcrun").args(["--show-sdk-path"]).output() {
        if output.status.success() {
            let sdk_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            builder = builder.clang_arg(format!("-isysroot{}", sdk_path));
        }
    }

    let bindings = builder
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
