// Copyright (c) 2026 Richard Albright. All rights reserved.

pub mod license;
pub mod continuous_indexing;

pub use license::{validate_license, has_feature, License};
pub use continuous_indexing::ContinuousIndexBuilder;
