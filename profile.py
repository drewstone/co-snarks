#!/usr/bin/env python3
"""Resource and latency profile: stock co-circom against patched builds, three REP3 parties on loopback.

usage: python3 profile.py --patch masked-root.patch [--patch ...] [--size 2] [--blocks 5] [--delay-ms 0,1] [--work DIR]

It uses reproduce.py (beside it) to fetch the pinned co-snarks tarball and the Fluidex circuit and to build each arm, and
public sources only. The circuit is Fluidex Rescue(size); size 2 is reproduce.py's circuit, size 48 the larger one.

Every connection between two parties goes through a local relay. The relay counts mpc-net frames (an 8-byte big-endian
length, then the payload) and payload bytes per direction, and can hold each frame for a fixed one-way delay before
forwarding it, without holding the frames behind it. Each block draws fresh private inputs, then for each delay runs
every arm once in a shuffled order. Per run and phase (split-input, generate-witness, generate-proof, verify) it records
wall seconds, and the party-0 frames and bytes of each phase.

Peak resident set size comes from a separate pass after the timed blocks: one run per arm at the first delay, with each
process started under a small Python wrapper that forks it, waits for it and records its ru_maxrss. Linux carries the
exec-ing process's high-water RSS into the new program's ru_maxrss, so a process started straight from this script would
report this script's own peak (about 190 MB after hashing the source tarball). Under the wrapper the inherited part is
the wrapper's own size, printed as the floor (the wrapper around `true`). The timed runs do not use the wrapper.

Sequential rounds of a phase are estimated as (median seconds at delay d - median seconds at delay 0) / w: the number
of one-way message waits on the phase's critical path, where w is the relay's effective wait per message at delay d,
measured on the same host before and after the runs by 400 sequential one-way messages (a sleep overshoots d). Circuit setup (compile and a Groth16 setup with one contribution
per phase) is timed once, separately. Throughput is not measured: one proof is in flight at a time.
"""
import argparse, collections, hashlib, json, os, pathlib, random, shutil, socket, statistics, subprocess, sys, tempfile, threading, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import reproduce as R  # noqa: E402

# argv: <rss file> <program> [args...]. Forks the program, waits, writes its ru_maxrss (KiB), exits with its status.
WRAP = ("import os, sys\npid = os.fork()\nif pid == 0:\n    os.execv(sys.argv[2], sys.argv[2:])\n"
        "_, status, usage = os.wait4(pid, 0)\nopen(sys.argv[1], 'w').write(str(usage.ru_maxrss))\n"
        "sys.exit(os.waitstatus_to_exitcode(status) & 255)\n")


def wrapped(argv, rss_file):
    return [sys.executable, "-I", "-S", "-c", WRAP, str(rss_file), *[str(a) for a in argv]]


def rss_floor(work):
    """KiB the wrapper itself contributes: its ru_maxrss report for `true`."""
    out = work / "floor.rss"
    subprocess.run(wrapped([shutil.which("true")], out), check=True)
    return int(out.read_text())

PHASES = ("split", "witness", "prove", "verify")


def setup(work, size):
    """reproduce.setup for Fluidex Rescue(size); returns (circuit directory, seconds)."""
    circuit = work / f"circuit-{size}"
    started = time.perf_counter()
    if not (circuit / "verification_key.json").exists():
        shutil.rmtree(circuit, ignore_errors=True)
        circuit.mkdir()
        R.fetch(f"https://codeload.github.com/fluidex/circuits/tar.gz/{R.FLUIDEX}", R.FLUIDEX_SHA256, work / "fluidex.tar.gz")
        R.sh(["tar", "-xzf", work / "fluidex.tar.gz", "-C", circuit, "--strip-components", "3", f"circuits-{R.FLUIDEX}/src/lib/rescue.circom", f"circuits-{R.FLUIDEX}/src/lib/rescue_constants.circom"])
        (circuit / "circuit.circom").write_text(f'include "rescue.circom";\n\ncomponent main = Rescue({size});\n')
        if not (work / "node/node_modules/.bin/snarkjs").exists():
            (work / "node").mkdir(exist_ok=True)
            R.sh(["npm", "install", "--prefix", work / "node", "--no-audit", "--no-fund", "snarkjs@0.7.6", "circom2@0.2.22"])
        snarkjs = [work / "node/node_modules/.bin/snarkjs"]
        R.sh([work / "node/node_modules/.bin/circom2", "circuit.circom", "--r1cs", "--O2", "-o", "."], cwd=circuit)
        info = R.sh([*snarkjs, "r1cs", "info", "circuit.r1cs"], cwd=circuit).stdout
        constraints = int(next(line for line in info.splitlines() if "# of Constraints" in line).split(":")[-1])
        power = max(8, (constraints + 2 - 1).bit_length() + 1)
        entropy = lambda: hashlib.sha256(os.urandom(32)).hexdigest()
        R.sh([*snarkjs, "powersoftau", "new", "bn128", power, "pot0.ptau"], cwd=circuit)
        R.sh([*snarkjs, "powersoftau", "contribute", "pot0.ptau", "pot1.ptau", "--name=phase1", f"-e={entropy()}"], cwd=circuit)
        R.sh([*snarkjs, "powersoftau", "prepare", "phase2", "pot1.ptau", "pot.ptau"], cwd=circuit)
        R.sh([*snarkjs, "groth16", "setup", "circuit.r1cs", "pot.ptau", "c0.zkey"], cwd=circuit)
        R.sh([*snarkjs, "zkey", "contribute", "c0.zkey", "circuit.zkey", "--name=phase2", f"-e={entropy()}"], cwd=circuit)
        R.sh([*snarkjs, "zkey", "export", "verificationkey", "circuit.zkey", "verification_key.json"], cwd=circuit)
        print(f"setup Rescue({size}): {constraints} constraints, powers of tau 2^{power}, {time.perf_counter() - started:.1f} s", flush=True)
    return circuit


class Relay:
    """One listener per ordered pair (i, j), on a port the system picks (`ports[(i, j)]`), forwarding to party j's own port
    `targets[(i, j)]`; counts frames and bytes per sender>receiver."""

    def __init__(self, targets, delay):
        self.delay, self.frames, self.bytes, self.servers, self.ports, self.lock = delay, collections.Counter(), collections.Counter(), [], {}, threading.Lock()
        for (i, j), target in targets.items():
            server = socket.create_server(("127.0.0.1", 0))
            self.servers.append(server)
            self.ports[(i, j)] = server.getsockname()[1]
            threading.Thread(target=self.accept, args=(server, i, j, target), daemon=True).start()

    def accept(self, server, i, j, target):
        while True:
            try:
                near, _ = server.accept()
            except OSError:
                return
            deadline = time.monotonic() + 60
            while True:
                try:
                    far = socket.create_connection(("127.0.0.1", target))
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        near.close()
                        return
                    time.sleep(0.01)
            for end in (near, far):
                end.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.pump(near, far, f"{i}>{j}")
            self.pump(far, near, f"{j}>{i}")

    def pump(self, source, sink, key):
        queue, ready = collections.deque(), threading.Condition()

        def read():
            header, remaining = b"", 0
            while True:
                try:
                    data = source.recv(1 << 16)
                except OSError:
                    data = b""
                frames, view, at = 0, memoryview(data), 0
                while at < len(view):
                    if remaining:
                        step = min(remaining, len(view) - at)
                        remaining, at = remaining - step, at + step
                    else:
                        step = min(8 - len(header), len(view) - at)
                        header, at = header + bytes(view[at:at + step]), at + step
                        if len(header) == 8:
                            remaining, header, frames = int.from_bytes(header, "big"), b"", frames + 1
                with self.lock:
                    self.frames[key] += frames
                    self.bytes[key] += len(data)
                with ready:
                    queue.append((time.monotonic() + self.delay, data))
                    ready.notify()
                if not data:
                    return

        def write():
            while True:
                with ready:
                    while not queue:
                        ready.wait()
                    due, data = queue.popleft()
                wait = due - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                try:
                    if not data:
                        sink.shutdown(socket.SHUT_WR)
                        return
                    sink.sendall(data)
                except OSError:
                    return

        threading.Thread(target=read, daemon=True).start()
        threading.Thread(target=write, daemon=True).start()

    def close(self):
        for server in self.servers:
            server.close()


def free_ports(count):
    """Ports the system reports free now, for the parties' own listeners."""
    probes = [socket.create_server(("127.0.0.1", 0)) for _ in range(count)]
    ports = [probe.getsockname()[1] for probe in probes]
    for probe in probes:
        probe.close()
    return ports


def calibrate(delay, n=200):
    """Seconds one message adds at `delay`: n sequential echoes (2n one-way waits) through a Relay, minus the same at 0."""
    def trial(d):
        server = socket.create_server(("127.0.0.1", 0))
        target = server.getsockname()[1]

        def echo():
            conn, _ = server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while True:
                data = conn.recv(1 << 16)
                if not data:
                    return
                conn.sendall(data)
        threading.Thread(target=echo, daemon=True).start()
        relay = Relay({(0, 1): target}, d)
        conn = socket.create_connection(("127.0.0.1", relay.ports[(0, 1)]))
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        message = (5).to_bytes(8, "big") + b"probe"

        def echoes(count):
            for _ in range(count):
                conn.sendall(message)
                got = b""
                while len(got) < len(message):
                    got += conn.recv(64)
        echoes(5)
        started = time.perf_counter()
        echoes(n)
        seconds = time.perf_counter() - started
        conn.close(), relay.close(), server.close()
        return seconds
    return (trial(delay) - trial(0.0)) / (2 * n)


def run(binary, stock, circuit, inputs, run_dir, delay, rss=False):
    """One full pipeline through a fresh relay: {phase: {seconds, frames0, bytes0[, rss_kib]}}, total seconds, public output.
    With rss, every process runs under the wrapper and rss_kib is the largest process's peak RSS in that phase."""
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    bind = free_ports(3)
    relay = Relay({(i, j): bind[j] for i in range(3) for j in range(3) if i != j}, delay)
    compiler = '[compiler]\nallow_leaky_loops = false\nlink_library = []\n\n[vm]\nallow_leaky_logs = false\n'
    for party in range(3):
        net = f'\n[network]\nmy_id = {party}\nbind_addr = "127.0.0.1:{bind[party]}"\ntimeout = "600s"\nconnect_timeout = "60s"\n'
        for other in range(3):
            address = bind[other] if other == party else relay.ports[(party, other)]
            net += f'\n[[network.parties]]\nid = {other}\ndns_name = "127.0.0.1:{address}"\n'
        (run_dir / f"party{party}.toml").write_text(compiler + net)
    (run_dir / "split.toml").write_text(compiler)
    (run_dir / "input.json").write_text(json.dumps({"inputs": [str(v) for v in inputs]}))
    env = {**os.environ, "RAYON_NUM_THREADS": "1", "RUST_LOG": "warn"}
    common = ["--protocol", "REP3", "--curve", "BN254"]
    out = {}
    sent = lambda counter: sum(v for k, v in counter.items() if k.startswith("0>"))

    def phase(name, argvs):
        with relay.lock:
            frames_before, bytes_before = sent(relay.frames), sent(relay.bytes)
        started = time.perf_counter()
        errors = [open(run_dir / f"{name}.{n}.err", "w") for n in range(len(argvs))]
        rss_files = [run_dir / f"{name}.{n}.rss" for n in range(len(argvs))]
        argvs = [wrapped(argv, rss_files[n]) if rss else [str(a) for a in argv] for n, argv in enumerate(argvs)]
        procs = [subprocess.Popen(argv, env=env, stdout=subprocess.DEVNULL, stderr=errors[n]) for n, argv in enumerate(argvs)]
        timer = threading.Timer(1800, lambda: [p.kill() for p in procs])
        timer.start()
        for proc in procs:
            _, status, _ = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
        timer.cancel()
        seconds = time.perf_counter() - started
        for handle in errors:
            handle.close()
        for n, proc in enumerate(procs):
            if proc.returncode != 0:
                sys.exit(f"{name} process {n} failed: {(run_dir / f'{name}.{n}.err').read_text()[-2000:]}")
        with relay.lock:
            frames0, bytes0 = sent(relay.frames) - frames_before, sent(relay.bytes) - bytes_before
        out[name] = {"seconds": round(seconds, 4), "frames0": frames0, "bytes0": bytes0}
        if rss:
            out[name]["rss_kib"] = max(int(f.read_text()) for f in rss_files)

    try:
        started = time.perf_counter()
        phase("split", [[stock, "split-input", "--circuit", circuit / "circuit.circom", "--input", run_dir / "input.json", *common, "--out-dir", run_dir, "--config", run_dir / "split.toml"]])
        phase("witness", [[binary, "generate-witness", "-O2", "--input", run_dir / f"input.json.{p}.shared", "--circuit", circuit / "circuit.circom", *common,
                           "--config", run_dir / f"party{p}.toml", "--out", run_dir / f"witness.{p}.shared"] for p in range(3)])
        phase("prove", [[binary, "generate-proof", "groth16", "--witness", run_dir / f"witness.{p}.shared", "--zkey", circuit / "circuit.zkey", *common,
                         "--config", run_dir / f"party{p}.toml", "--out", run_dir / f"proof.{p}.json"] + (["--public-input", run_dir / "public.json"] if p == 0 else [])
                        for p in range(3)])
        phase("verify", [[binary, "verify", "groth16", "--proof", run_dir / "proof.0.json", "--vk", circuit / "verification_key.json", "--public-input", run_dir / "public.json", "--curve", "BN254"]])
        total = time.perf_counter() - started
    finally:
        relay.close()
    R.sh([stock, "verify", "groth16", "--proof", run_dir / "proof.0.json", "--vk", circuit / "verification_key.json", "--public-input", run_dir / "public.json", "--curve", "BN254"], env=env)
    return out, total, json.loads((run_dir / "public.json").read_text())


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--patch", action="append", required=True)
    args.add_argument("--size", type=int, default=2)
    args.add_argument("--blocks", type=int, default=5)
    args.add_argument("--delay-ms", default="0,1")
    args.add_argument("--work")
    args.add_argument("--seed", default="1")
    opts = args.parse_args()
    patches = {pathlib.Path(p).resolve().name: pathlib.Path(p).resolve() for p in opts.patch}
    delays = [float(d) / 1000 for d in opts.delay_ms.split(",")]
    work = pathlib.Path(opts.work).resolve() if opts.work else pathlib.Path(tempfile.mkdtemp(prefix="co-circom-profile-"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        for name, path in patches.items():
            print(f"patch {name} sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}; co-snarks {R.PIN}; Rescue({opts.size}); blocks {opts.blocks}; delays ms {opts.delay_ms}", flush=True)
        R.fetch(f"https://codeload.github.com/TaceoLabs/co-snarks/tar.gz/{R.PIN}", R.PIN_SHA256, work / "co-snarks.tar.gz")
        stock = R.build(work, "stock", None)
        binaries = {"stock": stock, **{name: R.build(work, "patched", path) for name, path in patches.items()}}
        circuit = setup(work, opts.size)
        waits = {d: [calibrate(d)] for d in delays if d}
        rng = random.Random(f"profile:{opts.seed}:{opts.size}")
        warm = [rng.randrange(R.P) for _ in range(opts.size)]
        for binary in binaries.values():
            run(binary, stock, circuit, warm, work / "run", delays[0])
        rows = collections.defaultdict(list)
        for block in range(opts.blocks):
            inputs = [rng.randrange(R.P) for _ in range(opts.size)]
            for delay in delays:
                order = list(binaries)
                rng.shuffle(order)
                publics = set()
                for arm in order:
                    phases, total, public = run(binaries[arm], stock, circuit, inputs, work / "run", delay)
                    publics.add(json.dumps(public))
                    rows[(arm, delay)].append({"total": total, **phases})
                    print(f"RUN block {block} delay_ms {delay * 1000:g} {arm} total {total:.3f} " + json.dumps(phases) + f" load1 {os.getloadavg()[0]:.2f}", flush=True)
                if len(publics) != 1:
                    sys.exit(f"block {block}: the arms disagree on the public output")
        floor = rss_floor(work)
        peak = {}
        for arm in binaries:
            phases, _, _ = run(binaries[arm], stock, circuit, inputs, work / "run", delays[0], rss=True)
            peak[arm] = {p: phases[p]["rss_kib"] for p in PHASES}
            print(f"RSS {arm} delay_ms {delays[0] * 1000:g} " + json.dumps(peak[arm]) + f" floor_kib {floor}", flush=True)
        for d in waits:
            waits[d].append(calibrate(d))
            print(f"CALIBRATE delay_ms {d * 1000:g} effective wait ms before {waits[d][0] * 1000:.4f} after {waits[d][1] * 1000:.4f}", flush=True)
        for arm in binaries:
            for delay in delays:
                got = rows[(arm, delay)]
                med = {p: statistics.median(r[p]["seconds"] for r in got) for p in PHASES}
                summary = {"arm": arm, "size": opts.size, "delay_ms": delay * 1000, "runs": len(got), "total_s": round(statistics.median(r["total"] for r in got), 4),
                           "phase_s": {p: round(v, 4) for p, v in med.items()}, "peak_rss_kib": peak[arm], "rss_floor_kib": floor,
                           "frames0": {p: sorted({r[p]["frames0"] for r in got}) for p in ("witness", "prove")},
                           "bytes0": {p: sorted({r[p]["bytes0"] for r in got}) for p in ("witness", "prove")}}
                if delay and (arm, 0.0) in rows:
                    base = {p: statistics.median(r[p]["seconds"] for r in rows[(arm, 0.0)]) for p in ("witness", "prove")}
                    wait = statistics.mean(waits[delay])
                    summary["rounds_estimate"] = {p: round((med[p] - base[p]) / wait) for p in ("witness", "prove")}
                print("PROFILE " + json.dumps(summary), flush=True)
    finally:
        if not opts.work:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
