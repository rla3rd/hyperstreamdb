use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct License {
    pub key: String,
    pub features: Vec<String>,  // e.g., ["continuous_indexing", "advanced_optimizations"]
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    pub max_tables: Option<usize>,
}

/// Validate a license key
pub fn validate_license(license_key: &str) -> Result<License> {
    // For MVP, we use a simple prefix check
    // In production, this would be a signed JWT or similar encrypted token
    if license_key.starts_with("HSDB-ENT-") {
        Ok(License {
            key: license_key.to_string(),
            features: vec!["continuous_indexing".to_string()],
            expires_at: None,
            max_tables: None,
        })
    } else {
        anyhow::bail!("Invalid license key. Visit https://hyperstreamdb.com/pricing for a valid key.")
    }
}

pub fn has_feature(license: &License, feature: &str) -> bool {
    license.features.contains(&feature.to_string())
}
