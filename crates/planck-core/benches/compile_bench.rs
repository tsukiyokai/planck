// ==== Criterion Benchmarks ====
//
// Guards: compile < 1ms, instantiate < 1us

use criterion::{criterion_group, criterion_main, Criterion};
use planck_core::plan::{compile, CompileRequest, Collective, ReduceOp};
use planck_core::template::PlanTemplate;
use planck_core::topo::Topology;

fn bench_compile(c: &mut Criterion) {
    let topo = Topology::hccs_8card();

    let configs: &[(&str, usize, usize)] = &[
        ("compile_256mb_4chunk", 256 << 20, 4),
        ("compile_16kb_1chunk",   16 << 10, 1),
        ("compile_1mb_2chunk",     1 << 20, 2),
    ];

    for &(name, msg_size, chunks) in configs {
        let req = CompileRequest {
            collective:      Collective::AllReduce,
            msg_size,
            reduce_op:       ReduceOp::Sum,
            num_ranks:       8,
            my_rank:         0,
            pipeline_chunks: chunks,
        };
        c.bench_function(name, |b| b.iter(|| compile(&req, &topo)));
    }
}

fn bench_instantiate(c: &mut Criterion) {
    let topo = Topology::hccs_8card();
    let req = CompileRequest {
        collective:      Collective::AllReduce,
        msg_size:        16 << 10,
        reduce_op:       ReduceOp::Sum,
        num_ranks:       8,
        my_rank:         0,
        pipeline_chunks: 1,
    };
    let template = PlanTemplate::from_plan(compile(&req, &topo), 16 << 10);

    c.bench_function("instantiate_16kb", |b| {
        b.iter(|| template.instantiate(16384))
    });
}

criterion_group!(benches, bench_compile, bench_instantiate);
criterion_main!(benches);
