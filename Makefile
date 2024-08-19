start:
# 	wasm-pack build
	cargo build --target wasm32-unknown-unknown
	wasm-bindgen target/wasm32-unknown-unknown/debug/deep_learning.wasm --out-dir ./www/src/script/wasm
	$(MAKE) -C www