use thiserror::Error;

#[derive(Error, Debug)]
pub enum MyError {
    #[error("Failed to create shader module")]
    ShaderCreationError,
    #[error("Invalid texture dimensions or empty data")]
    InvalidTextureData,
    #[error("Failed to create texture")]
    TextureCreationError,
    #[error("Failed to create buffer")]
    BufferCreationError,
}
