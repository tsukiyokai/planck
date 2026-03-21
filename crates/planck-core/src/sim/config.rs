// ==== SimConfig: TOML-driven simulation parameters ====

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct SimConfig {
    #[serde(default)]
    pub collective: CollectiveConfig,
    #[serde(default)]
    pub topology: TopoConfig,
    #[serde(default)]
    pub timing: TimingConfig,
    #[serde(default)]
    pub output: OutputConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CollectiveConfig {
    #[serde(rename = "type", default = "default_coll_type")]
    pub coll_type: String,
    #[serde(default = "default_msg_size", deserialize_with = "deser_size")]
    pub msg_size: usize,
    #[serde(default = "default_chunks")]
    pub pipeline_chunks: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TopoConfig {
    #[serde(default = "default_preset")]
    pub preset: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TimingConfig {
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_bw")]
    pub hccs_bw_gbps: f64,
    #[serde(default = "default_lat")]
    pub hccs_lat_us: f64,
    #[serde(default = "default_notify")]
    pub notify_rounds: u32,
    #[serde(default = "default_hbm")]
    pub hbm_bw_gbps: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OutputConfig {
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default = "default_file")]
    pub file: String,
}

// ==== Defaults (match topo.rs calibrated constants) ====

fn default_coll_type() -> String { "allreduce".into() }
fn default_msg_size() -> usize { 256 << 20 } // 256 MB
fn default_chunks() -> usize { 4 }
fn default_preset() -> String { "hccs_8card".into() }
fn default_model() -> String { "ascend".into() }
fn default_bw() -> f64 { 30.0 } // GB/s
fn default_lat() -> f64 { 1.5 } // us
fn default_notify() -> u32 { 3 } // 3-round handshake
fn default_hbm() -> f64 { 460.0 } // GB/s
fn default_format() -> String { "chrome_trace".into() }
fn default_file() -> String { "trace.json".into() }

impl Default for CollectiveConfig {
    fn default() -> Self {
        Self {
            coll_type: default_coll_type(),
            msg_size: default_msg_size(),
            pipeline_chunks: default_chunks(),
        }
    }
}
impl Default for TopoConfig {
    fn default() -> Self { Self { preset: default_preset() } }
}
impl Default for TimingConfig {
    fn default() -> Self {
        Self {
            model: default_model(),
            hccs_bw_gbps: default_bw(),
            hccs_lat_us: default_lat(),
            notify_rounds: default_notify(),
            hbm_bw_gbps: default_hbm(),
        }
    }
}
impl Default for OutputConfig {
    fn default() -> Self { Self { format: default_format(), file: default_file() } }
}
impl Default for SimConfig {
    fn default() -> Self {
        Self {
            collective: Default::default(),
            topology: Default::default(),
            timing: Default::default(),
            output: Default::default(),
        }
    }
}

// ==== Size parsing: "256MB" -> 268435456 ====

fn deser_size<'de, D: serde::Deserializer<'de>>(de: D) -> Result<usize, D::Error> {
    let s = String::deserialize(de)?;
    parse_size_str(&s).map_err(serde::de::Error::custom)
}

pub fn parse_size_str(s: &str) -> Result<usize, String> {
    let s = s.trim();
    if let Ok(n) = s.parse::<usize>() {
        return Ok(n);
    }

    // Split numeric prefix from suffix (1-2 chars)
    let split = s.len().saturating_sub(2);
    let (num, suffix) = s.split_at(split);
    let n: f64 = num.trim().parse().map_err(|e| format!("{e}"))?;
    match suffix.to_uppercase().as_str() {
        "KB" => Ok((n * 1024.0) as usize),
        "MB" => Ok((n * 1024.0 * 1024.0) as usize),
        "GB" => Ok((n * 1024.0 * 1024.0 * 1024.0) as usize),
        _ => Err(format!("unknown size suffix: {s}")),
    }
}

impl SimConfig {
    pub fn from_toml(s: &str) -> Result<Self, String> {
        toml::from_str(s).map_err(|e| format!("{e}"))
    }
}

// ==== Tests ====

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sim_parse_size_str() {
        assert_eq!(parse_size_str("1024").unwrap(), 1024);
        assert_eq!(parse_size_str("16KB").unwrap(), 16 * 1024);
        assert_eq!(parse_size_str("256MB").unwrap(), 256 * 1024 * 1024);
        assert_eq!(parse_size_str("1GB").unwrap(), 1024 * 1024 * 1024);
    }

    #[test]
    fn sim_default_config() {
        let cfg = SimConfig::default();
        assert_eq!(cfg.collective.msg_size, 256 << 20);
        assert_eq!(cfg.timing.hccs_bw_gbps, 30.0);
        assert_eq!(cfg.timing.notify_rounds, 3);
    }

    #[test]
    fn sim_toml_parse() {
        let toml = r#"
[collective]
type = "allreduce"
msg_size = "16KB"
pipeline_chunks = 2

[timing]
model = "simple"
"#;
        let cfg = SimConfig::from_toml(toml).unwrap();
        assert_eq!(cfg.collective.msg_size, 16 * 1024);
        assert_eq!(cfg.collective.pipeline_chunks, 2);
        assert_eq!(cfg.timing.model, "simple");
    }
}
