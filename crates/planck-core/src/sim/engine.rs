// ==== DES Engine: discrete event simulation for plan schedule verification ====
//
// EventQueue (BinaryHeap min-heap) + signal/wait tracking + link contention.
// Does NOT simulate data — only time. Answers "is this schedule good?"

use super::{config::SimConfig, link::LinkState, timing::TimingModel, trace::Trace};
use crate::{
    plan::{ExecutionPlan, Opcode},
    topo::Topology,
};
use std::{cmp::Ordering, collections::BinaryHeap};

// ==== Event Types ====

#[derive(Debug, Clone)]
pub struct Event {
    pub time: f64,
    pub rank: u16,
    pub stream: u8,
    pub kind: EventKind,
}

#[derive(Debug, Clone)]
pub enum EventKind {
    OpStart { op_idx: u16 },
    PutEnd { link_idx: usize }, // release link flow
    Unblock { op_idx: u16 },    // signal arrived, resume blocked Wait
}

// Min-heap: BinaryHeap is max-heap, so reverse comparison
impl PartialEq for Event {
    fn eq(&self, o: &Self) -> bool { self.time == o.time }
}
impl Eq for Event {}
impl PartialOrd for Event {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}
impl Ord for Event {
    fn cmp(&self, o: &Self) -> Ordering {
        o.time.partial_cmp(&self.time).unwrap_or(Ordering::Equal)
    }
}

// ==== Simulator ====

pub struct Simulator {
    queue: BinaryHeap<Event>,
    links: Vec<LinkState>,
    plans: Vec<ExecutionPlan>,
    trace: Trace,
    model: Box<dyn TimingModel>,
    _num_ranks: usize,
    // signals[src][dst]: pending signal count from src to dst
    signals: Vec<Vec<i32>>,
    // waiting[rank]: Some(op_idx) if rank is blocked on a Wait
    waiting: Vec<Option<u16>>,
}

impl Simulator {
    pub fn new(
        plans: &[ExecutionPlan],
        topo: &Topology,
        model: Box<dyn TimingModel>,
        _cfg: &SimConfig,
    ) -> Self {
        let n = plans.len();
        let num_streams = plans.first().map_or(1, |p| p.header.num_streams.max(1)) as usize;

        let links: Vec<LinkState> = topo.links.iter().map(|l| LinkState::new(l.clone())).collect();

        let mut sim = Self {
            queue: BinaryHeap::new(),
            links,
            plans: plans.to_vec(),
            trace: Trace::new(n),
            model,
            _num_ranks: n,
            signals: vec![vec![0i32; n]; n],
            waiting: vec![None; n],
        };

        // Seed: first op of each rank/stream at t=0
        for rank in 0..n {
            for s in 0..num_streams {
                if let Some(idx) = plans[rank].ops.iter().position(|o| o.stream_id == s as u8) {
                    sim.queue.push(Event {
                        time: 0.0,
                        rank: rank as u16,
                        stream: s as u8,
                        kind: EventKind::OpStart { op_idx: idx as u16 },
                    });
                }
            }
        }

        sim
    }

    pub fn run(&mut self) {
        while let Some(ev) = self.queue.pop() {
            match ev.kind {
                EventKind::OpStart { op_idx } => {
                    self.handle_op(ev.rank as usize, ev.stream, op_idx, ev.time)
                }
                EventKind::PutEnd { link_idx } => self.links[link_idx].remove_flow(),
                EventKind::Unblock { op_idx } => {
                    self.handle_op(ev.rank as usize, ev.stream, op_idx, ev.time)
                }
            }
        }
    }

    fn handle_op(&mut self, rank: usize, stream: u8, op_idx: u16, now: f64) {
        let op = self.plans[rank].ops[op_idx as usize];
        let oc = Opcode::from_u8(op.opcode).unwrap_or(Opcode::Noop);
        let bufs = &self.plans[rank].buffers;

        match oc {
            Opcode::Noop => {
                self.next(rank, stream, op_idx, now);
            }

            Opcode::Signal => {
                let dst = op.dst_rank as usize;
                self.signals[rank][dst] += 1;
                self.try_unblock(dst, now);
                self.next(rank, stream, op_idx, now);
            }

            Opcode::Wait => {
                let src = op.dst_rank as usize; // "dst_rank" in Wait = signal source
                if self.signals[src][rank] > 0 {
                    self.signals[src][rank] -= 1;
                    let dur = self.notify_time();
                    self.trace.push(rank, stream, "Wait", now, dur, None);
                    self.next(rank, stream, op_idx, now + dur);
                } else {
                    self.waiting[rank] = Some(op_idx);
                }
            }

            Opcode::Put | Opcode::PutWithSignal => {
                let size = bufs[op.src_buf as usize].size as usize;
                let link_idx = self.find_link(rank, op.dst_rank as usize);
                self.links[link_idx].add_flow();
                let dur = self.model.put_time(&self.links[link_idx].link, size);

                self.trace.push(rank, stream, "Put", now, dur, Some(op.dst_rank));
                self.queue.push(Event {
                    time: now + dur,
                    rank: rank as u16,
                    stream,
                    kind: EventKind::PutEnd { link_idx },
                });

                if oc == Opcode::PutWithSignal {
                    let dst = op.dst_rank as usize;
                    self.signals[rank][dst] += 1;
                    self.try_unblock(dst, now + dur);
                }

                self.next(rank, stream, op_idx, now + dur);
            }

            Opcode::LocalReduce | Opcode::LocalCopy => {
                let size = bufs[op.src_buf as usize].size as usize;
                let dur = self.model.reduce_time(size);
                let name = if oc == Opcode::LocalReduce { "Reduce" } else { "Copy" };
                self.trace.push(rank, stream, name, now, dur, None);
                self.next(rank, stream, op_idx, now + dur);
            }

            Opcode::WaitReducePut => {
                // Bug fix: wait_event carries the Wait source rank
                // (impl plan used dst_rank which is the Put destination — wrong)
                let wait_for = op.wait_event as usize;
                if self.signals[wait_for][rank] > 0 {
                    self.signals[wait_for][rank] -= 1;
                    let size = bufs[op.src_buf as usize].size as usize;
                    let link_idx = self.find_link(rank, op.dst_rank as usize);
                    self.links[link_idx].add_flow();
                    let dur = self.model.inline_reduce_put_time(&self.links[link_idx].link, size);

                    self.trace.push(rank, stream, "WaitReducePut", now, dur, Some(op.dst_rank));
                    self.queue.push(Event {
                        time: now + dur,
                        rank: rank as u16,
                        stream,
                        kind: EventKind::PutEnd { link_idx },
                    });

                    // Signal dst_rank after fused put completes
                    let dst = op.dst_rank as usize;
                    self.signals[rank][dst] += 1;
                    self.try_unblock(dst, now + dur);

                    self.next(rank, stream, op_idx, now + dur);
                } else {
                    self.waiting[rank] = Some(op_idx);
                }
            }

            Opcode::WaitReduceCopy => {
                let wait_for = op.dst_rank as usize; // dst_rank = Wait source (preserved in fusion)
                if self.signals[wait_for][rank] > 0 {
                    self.signals[wait_for][rank] -= 1;
                    let size = bufs[op.src_buf as usize].size as usize;
                    // notify + reduce + copy (both HBM-bound)
                    let dur = self.notify_time() + self.model.reduce_time(size) * 2.0;
                    self.trace.push(rank, stream, "WaitReduceCopy", now, dur, None);
                    self.next(rank, stream, op_idx, now + dur);
                } else {
                    self.waiting[rank] = Some(op_idx);
                }
            }
        }
    }

    /// Schedule next op on same stream after current op.
    fn next(&mut self, rank: usize, stream: u8, current: u16, after: f64) {
        let ops = &self.plans[rank].ops;
        for idx in (current as usize + 1)..ops.len() {
            if ops[idx].stream_id == stream {
                self.queue.push(Event {
                    time: after + self.model.kernel_launch_overhead(),
                    rank: rank as u16,
                    stream,
                    kind: EventKind::OpStart { op_idx: idx as u16 },
                });
                return;
            }
        }
    }

    /// Unblock a waiting rank at the given time.
    fn try_unblock(&mut self, target: usize, at: f64) {
        if let Some(wait_op) = self.waiting[target] {
            self.waiting[target] = None;
            let stream = self.plans[target].ops[wait_op as usize].stream_id;
            self.queue.push(Event {
                time: at,
                rank: target as u16,
                stream,
                kind: EventKind::Unblock { op_idx: wait_op },
            });
        }
    }

    /// Shortcut: notify time using first link (all HCCS links have same latency).
    fn notify_time(&self) -> f64 { self.model.notify_time(&self.links[0].link) }

    fn find_link(&self, src: usize, dst: usize) -> usize {
        self.links.iter().position(|l| l.link.src == src && l.link.dst == dst).unwrap_or(0)
    }

    pub fn into_trace(self) -> Trace { self.trace }
}

// ==== Tests ====

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sim_event_ordering() {
        let mut q = BinaryHeap::new();
        q.push(Event { time: 5.0, rank: 0, stream: 0, kind: EventKind::OpStart { op_idx: 0 } });
        q.push(Event { time: 1.0, rank: 0, stream: 0, kind: EventKind::OpStart { op_idx: 1 } });
        q.push(Event { time: 3.0, rank: 0, stream: 0, kind: EventKind::OpStart { op_idx: 2 } });
        assert_eq!(q.pop().unwrap().time, 1.0);
        assert_eq!(q.pop().unwrap().time, 3.0);
        assert_eq!(q.pop().unwrap().time, 5.0);
    }
}
