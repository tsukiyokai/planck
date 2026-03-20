// ==== Chrome Trace JSON output ====
//
// Generates Perfetto-compatible trace with rank metadata.
// Format: {"traceEvents": [...]} with X (complete) events.

use std::fmt::Write;

pub struct TraceEvent {
    pub name:     String,
    pub ph:       char,       // 'X'=complete
    pub pid:      u16,        // rank
    pub tid:      u8,         // stream
    pub ts:       f64,        // microseconds
    pub dur:      f64,        // microseconds
    pub dst_rank: Option<u16>,
}

pub struct Trace {
    pub events:    Vec<TraceEvent>,
    pub num_ranks: usize,
}

impl Trace {
    pub fn new(num_ranks: usize) -> Self {
        Self { events: Vec::new(), num_ranks }
    }

    pub fn push(&mut self, rank: usize, stream: u8, name: &str, ts: f64, dur: f64, dst: Option<u16>) {
        self.events.push(TraceEvent {
            name: name.to_string(),
            ph: 'X',
            pid: rank as u16,
            tid: stream,
            ts, dur,
            dst_rank: dst,
        });
    }

    pub fn to_json(&self) -> String {
        let mut s = String::with_capacity(self.events.len() * 120 + 256);
        s.push_str("{\"traceEvents\":[\n");

        // Metadata: rank names
        for r in 0..self.num_ranks {
            write!(s, "{{\"ph\":\"M\",\"pid\":{r},\"name\":\"process_name\",\
                        \"args\":{{\"name\":\"Rank {r}\"}}}},\n").ok();
        }

        // Events
        for (i, ev) in self.events.iter().enumerate() {
            write!(s, "{{\"name\":\"{}\",\"cat\":\"planck\",\"ph\":\"{}\",\
                        \"pid\":{},\"tid\":{},\"ts\":{:.3},\"dur\":{:.3}",
                ev.name, ev.ph, ev.pid, ev.tid, ev.ts, ev.dur).ok();
            if let Some(dst) = ev.dst_rank {
                write!(s, ",\"args\":{{\"dst_rank\":{dst}}}").ok();
            }
            s.push('}');
            if i + 1 < self.events.len() { s.push(','); }
            s.push('\n');
        }

        s.push_str("]}\n");
        s
    }

    pub fn total_time(&self) -> f64 {
        self.events.iter().map(|e| e.ts + e.dur).fold(0.0f64, f64::max)
    }
}

// ==== Tests ====

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sim_trace_json_valid() {
        let mut t = Trace::new(2);
        t.push(0, 0, "Put", 0.0, 10.0, Some(1));
        t.push(1, 0, "Wait", 10.0, 2.0, None);
        let json = t.to_json();
        assert!(json.contains("traceEvents"));
        assert!(json.contains("\"name\":\"Put\""));
        assert!(json.contains("\"dst_rank\":1"));
    }

    #[test]
    fn sim_trace_total_time() {
        let mut t = Trace::new(2);
        t.push(0, 0, "Put", 0.0, 10.0, None);
        t.push(1, 0, "Wait", 5.0, 8.0, None); // ends at 13.0
        assert!((t.total_time() - 13.0).abs() < 0.001);
    }
}
