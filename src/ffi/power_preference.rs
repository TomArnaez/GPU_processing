use wgpu::PowerPreference;

#[repr(C)]
pub enum CPowerPreference {
    None = 0,
    LowPower = 1,
    HighPerformance = 2,
}

// Conversion from Rust enum to C enum
impl From<PowerPreference> for CPowerPreference {
    fn from(pref: PowerPreference) -> Self {
        match pref {
            PowerPreference::None => CPowerPreference::None,
            PowerPreference::LowPower => CPowerPreference::LowPower,
            PowerPreference::HighPerformance => CPowerPreference::HighPerformance,
        }
    }
}

// Conversion from C enum to Rust enum (optional, depending on use case)
impl From<CPowerPreference> for PowerPreference {
    fn from(pref: CPowerPreference) -> Self {
        match pref {
            CPowerPreference::None => PowerPreference::None,
            CPowerPreference::LowPower => PowerPreference::LowPower,
            CPowerPreference::HighPerformance => PowerPreference::HighPerformance,
        }
    }
}