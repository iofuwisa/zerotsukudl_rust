#[cfg(target_family = "wasm")]
pub mod wasm;
#[cfg(target_family = "wasm")]
pub use self::wasm::*;

#[cfg (not(target_family = "wasm"))]
pub mod other;
#[cfg (not(target_family = "wasm"))]
pub use self::other::*;