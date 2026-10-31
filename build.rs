use std::{env, path::PathBuf};
fn main() {
    println!("cargo:rerun-if-env-changed=MISA_QP_NATIVE_ROOT");
    println!("cargo:rerun-if-changed=native/bridge.cpp");
    let features = ["proxqp", "eiquadprog", "qpoases", "osqp", "hpipm"];
    if !features
        .iter()
        .any(|f| env::var_os(format!("CARGO_FEATURE_{}", f.to_uppercase())).is_some())
    {
        return;
    }
    let root = PathBuf::from(
        env::var_os("MISA_QP_NATIVE_ROOT").unwrap_or("/tmp/articara-native-qp".into()),
    );
    assert!(
        root.join("generated").exists(),
        "Run bash native/setup.sh (in the misa-wbc checkout) first; set MISA_QP_NATIVE_ROOT if using a custom location"
    );
    let mut b = cc::Build::new();
    b.cpp(true)
        .std("c++17")
        .opt_level(3)
        .warnings(false)
        .include("/usr/include/eigen3")
        .include(root.join("generated"))
        .file("native/bridge.cpp");
    for f in features {
        if env::var_os(format!("CARGO_FEATURE_{}", f.to_uppercase())).is_none() {
            continue;
        }
        b.define(&format!("MISA_{}", f.to_uppercase()), None);
        match f {
            "proxqp" => {
                b.include(root.join("proxsuite/include"));
            }
            "eiquadprog" => {
                b.include(root.join("eiquadprog/include"))
                    .file(root.join("eiquadprog/src/eiquadprog-fast.cpp"));
            }
            "qpoases" => {
                b.include(root.join("qpoases/include"))
                    .define("__SUPPRESSANYOUTPUT__", None);
                for entry in std::fs::read_dir(root.join("qpoases/src")).unwrap() {
                    let p = entry.unwrap().path();
                    if p.extension().is_some_and(|x| x == "cpp") {
                        b.file(p);
                    }
                }
            }
            "osqp" => {
                b.include(root.join("osqp/include"))
                    .include(root.join("osqp/build/include"));
            }
            "hpipm" => {
                b.include(root.join("install/include"));
            }
            _ => unreachable!(),
        }
    }
    b.compile("misa_qp_native");
    if env::var_os("CARGO_FEATURE_OSQP").is_some() {
        println!(
            "cargo:rustc-link-search=native={}",
            root.join("osqp/build/out").display()
        );
        println!("cargo:rustc-link-lib=static=osqp");
    }
    if env::var_os("CARGO_FEATURE_HPIPM").is_some() {
        println!(
            "cargo:rustc-link-search=native={}",
            root.join("install/lib").display()
        );
        println!("cargo:rustc-link-lib=static=hpipm");
        println!("cargo:rustc-link-lib=static=blasfeo");
    }
}
