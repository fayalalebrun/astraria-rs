# Astraria

**[Try it in your browser](https://fayalalebrun.github.io/astraria-rs/)**

3D orbital mechanics simulator written in Rust. Supports native (desktop) and WebGL2/WASM (browser) targets.

## Building

```bash
# Native
cargo run --release

# Web (requires wasm-pack)
wasm-pack build --target web --no-default-features --features web
```

## Repository

https://github.com/fayalalebrun/astraria-rs
