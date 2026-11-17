#!/usr/bin/env python3
"""Standalone reproduction: stock co-circom against co-circom with a patch, three REP3 parties on loopback.

usage: python3 reproduce.py [--patch candidate.patch ...] [--blocks 16] [--work DIR] [--seed 1]

Everything comes from public sources: the pinned co-snarks tarball, the Fluidex Rescue(2) circuit from
fluidex/circuits, and snarkjs 0.7.6 and circom2 0.2.22 from npm. It needs curl, tar, patch, cargo, node and npm, and
python3 (standard library only).

1. Builds co-circom from TaceoLabs/co-snarks at PIN: stock, and stock with each patch applied on its own.
2. Compiles Rescue(2) (fluidex/circuits test/testcases/rescue.ts TestRescueHash2) with circom2 and makes a fresh Groth16
   setup with one contribution in each phase (snarkjs).
3. Runs blocks. Each block draws two random field elements as private inputs and runs the full pipeline (split-input,
   generate-witness on three parties, generate-proof on three parties, verify) once per arm in a shuffled order: two
   stock copies and one run per patch. Each pipeline's wall time is measured. Every proof must verify under the stock
   binary, and every arm must produce the same public output.
4. Prints, per patch, the geometric-mean reduction of patched / sqrt(stock1 * stock2) with a 95% percentile-bootstrap
   interval, and the same summary of stock2 / stock1 (the A/A noise floor). Pass --patch more than once to time
   several patches against the same stock copies, for example each half of a patch and both together.

Parties talk over plain TCP on 127.0.0.1 with one Rayon thread each. The work directory is removed at the end unless
--work is given.
"""
import argparse, hashlib, json, math, os, pathlib, random, shutil, statistics, subprocess, sys, tempfile, time, urllib.request

PIN = "b6e99e07fef4911f0423edc7ed06065fba555f73"
PIN_SHA256 = "f82b0112175520b2156013c1a82def67625aa43dff6736c71e98ce8f46cb93b5"
FLUIDEX = "d7e285a0f242829b9444ad9237123e4b66c8d4b1"
FLUIDEX_SHA256 = "3223b5b8b6946df319f3d72752d87f250f46dfbd7b2be4dfa36cf89895691bec"
P = 21888242871839275222246405745257275088548364400416034343698204186575808495617
HERE = pathlib.Path(__file__).resolve().parent


def sh(argv, cwd=None, env=None, quiet=True):
    done = subprocess.run([str(a) for a in argv], cwd=cwd, env=env, capture_output=quiet, text=True)
    if done.returncode != 0:
        sys.exit(f"failed ({done.returncode}): {' '.join(map(str, argv))}\n{(done.stderr or '')[-3000:]}")
    return done


def fetch(url, sha, path):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=300) as response, open(path, "wb") as out:
            shutil.copyfileobj(response, out)
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != sha:
        sys.exit(f"{url}: sha256 {got}, expected {sha}")


def build(work, name, patch):
    tree = work / (f"{name}-{hashlib.sha256(patch.read_bytes()).hexdigest()[:12]}" if patch else name)
    if not (tree / "target/release/co-circom").exists():
        shutil.rmtree(tree, ignore_errors=True)
        tree.mkdir()
        sh(["tar", "-xzf", work / "co-snarks.tar.gz", "-C", tree, "--strip-components", "1"])
        if patch:
            sh(["patch", "-p1", "--forward", "-i", patch], cwd=tree)
        print(f"building {name} co-circom (cargo build --release -p co-circom)", flush=True)
        sh(["cargo", "build", "--release", "-p", "co-circom", "--bin", "co-circom", "--target-dir", tree / "target"], cwd=tree)
    return tree / "target/release/co-circom"


def setup(work):
    circuit = work / "circuit"
    if (circuit / "verification_key.json").exists():
        return circuit
    shutil.rmtree(circuit, ignore_errors=True)
    circuit.mkdir()
    fetch(f"https://codeload.github.com/fluidex/circuits/tar.gz/{FLUIDEX}", FLUIDEX_SHA256, work / "fluidex.tar.gz")
    sh(["tar", "-xzf", work / "fluidex.tar.gz", "-C", circuit, "--strip-components", "3", f"circuits-{FLUIDEX}/src/lib/rescue.circom", f"circuits-{FLUIDEX}/src/lib/rescue_constants.circom"])
    (circuit / "circuit.circom").write_text('include "rescue.circom";\n\ncomponent main = Rescue(2);\n')
    (work / "node").mkdir(exist_ok=True)
    sh(["npm", "install", "--prefix", work / "node", "--no-audit", "--no-fund", "snarkjs@0.7.6", "circom2@0.2.22"])
    snarkjs = [work / "node/node_modules/.bin/snarkjs"]
    sh([work / "node/node_modules/.bin/circom2", "circuit.circom", "--r1cs", "--O2", "-o", "."], cwd=circuit)
    info = sh([*snarkjs, "r1cs", "info", "circuit.r1cs"], cwd=circuit).stdout
    constraints = int(next(line for line in info.splitlines() if "# of Constraints" in line).split(":")[-1])
    power = max(8, math.ceil(math.log2(constraints + 2)) + 1)
    entropy = lambda: hashlib.sha256(os.urandom(32)).hexdigest()
    sh([*snarkjs, "powersoftau", "new", "bn128", power, "pot0.ptau"], cwd=circuit)
    sh([*snarkjs, "powersoftau", "contribute", "pot0.ptau", "pot1.ptau", "--name=phase1", f"-e={entropy()}"], cwd=circuit)
    sh([*snarkjs, "powersoftau", "prepare", "phase2", "pot1.ptau", "pot.ptau"], cwd=circuit)
    sh([*snarkjs, "groth16", "setup", "circuit.r1cs", "pot.ptau", "c0.zkey"], cwd=circuit)
    sh([*snarkjs, "zkey", "contribute", "c0.zkey", "circuit.zkey", "--name=phase2", f"-e={entropy()}"], cwd=circuit)
    sh([*snarkjs, "zkey", "export", "verificationkey", "circuit.zkey", "verification_key.json"], cwd=circuit)
    print(f"setup: {constraints} constraints, powers of tau 2^{power}", flush=True)
    return circuit


PORT = [20000]


def pipeline(binary, stock, circuit, inputs, run_dir):
    """One full pipeline; returns (seconds, public output)."""
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    PORT[0] = 20000 + (PORT[0] - 20000 + 3) % 9000  # fresh ports each run, so no socket lingers in TIME_WAIT
    ports = [PORT[0] + i for i in range(3)]
    compiler = '[compiler]\nallow_leaky_loops = false\nlink_library = []\n\n[vm]\nallow_leaky_logs = false\n'
    for party in range(3):
        net = f'\n[network]\nmy_id = {party}\nbind_addr = "127.0.0.1:{ports[party]}"\ntimeout = "120s"\nconnect_timeout = "60s"\n'
        net += "".join(f'\n[[network.parties]]\nid = {other}\ndns_name = "127.0.0.1:{ports[other]}"\n' for other in range(3))
        (run_dir / f"party{party}.toml").write_text(compiler + net)
    (run_dir / "split.toml").write_text(compiler)
    (run_dir / "input.json").write_text(json.dumps({"inputs": [str(v) for v in inputs]}))
    env = {**os.environ, "RAYON_NUM_THREADS": "1", "RUST_LOG": "warn"}
    common = ["--protocol", "REP3", "--curve", "BN254"]
    started = time.perf_counter()
    sh([binary, "split-input", "--circuit", circuit / "circuit.circom", "--input", run_dir / "input.json", *common, "--out-dir", run_dir, "--config", run_dir / "split.toml"], env=env)

    def parties(argv_of):
        procs = [subprocess.Popen([str(a) for a in argv_of(p)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True) for p in range(3)]
        for p, proc in enumerate(procs):
            _, err = proc.communicate(timeout=600)
            if proc.returncode != 0:
                sys.exit(f"party {p} failed: {err[-2000:]}")

    parties(lambda p: [binary, "generate-witness", "-O2", "--input", run_dir / f"input.json.{p}.shared", "--circuit", circuit / "circuit.circom", *common,
                       "--config", run_dir / f"party{p}.toml", "--out", run_dir / f"witness.{p}.shared"])
    parties(lambda p: [binary, "generate-proof", "groth16", "--witness", run_dir / f"witness.{p}.shared", "--zkey", circuit / "circuit.zkey", *common,
                       "--config", run_dir / f"party{p}.toml", "--out", run_dir / f"proof.{p}.json"] + (["--public-input", run_dir / "public.json"] if p == 0 else []))
    sh([binary, "verify", "groth16", "--proof", run_dir / "proof.0.json", "--vk", circuit / "verification_key.json", "--public-input", run_dir / "public.json", "--curve", "BN254"], env=env)
    seconds = time.perf_counter() - started
    # Independent of the arm's own verify: the stock binary must accept the proof too.
    sh([stock, "verify", "groth16", "--proof", run_dir / "proof.0.json", "--vk", circuit / "verification_key.json", "--public-input", run_dir / "public.json", "--curve", "BN254"], env=env)
    return seconds, json.loads((run_dir / "public.json").read_text())


def summary(logs, seed):
    rng = random.Random(seed)
    n, mean = len(logs), sum(logs) / len(logs)
    means = sorted(sum(rng.choice(logs) for _ in logs) / n for _ in range(5000))
    pct = lambda x: 100 * (1 - math.exp(x))
    return pct(mean), pct(means[int(0.975 * 5000)]), pct(means[int(0.025 * 5000)]), sum(1 for x in logs if x < 0)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--patch", action="append", help="a patch to time against stock; repeat for more (default: candidate.patch)")
    args.add_argument("--blocks", type=int, default=16)
    args.add_argument("--work")
    args.add_argument("--seed", default="1")
    opts = args.parse_args()
    patches = {}
    for given in opts.patch or [str(HERE / "candidate.patch")]:
        path = pathlib.Path(given).resolve()
        patches[path.name] = path
    work = pathlib.Path(opts.work).resolve() if opts.work else pathlib.Path(tempfile.mkdtemp(prefix="co-circom-repro-"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        for name, path in patches.items():
            print(f"patch {name} sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}; co-snarks {PIN}; blocks {opts.blocks}; seed {opts.seed}", flush=True)
        fetch(f"https://codeload.github.com/TaceoLabs/co-snarks/tar.gz/{PIN}", PIN_SHA256, work / "co-snarks.tar.gz")
        stock = build(work, "stock", None)
        binaries = {name: build(work, "patched", path) for name, path in patches.items()}
        circuit = setup(work)
        rng = random.Random(f"repro:{opts.seed}")
        warm = [rng.randrange(P), rng.randrange(P)]
        for binary in (stock, *binaries.values()):  # one untimed run each, so none pays for a cold page cache
            pipeline(binary, stock, circuit, warm, work / "run")
        ratios, floors, seconds = {name: [] for name in patches}, [], {"stock": [], **{name: [] for name in patches}}
        print("block | order | seconds | load1", flush=True)
        for block in range(opts.blocks):
            inputs = [rng.randrange(P), rng.randrange(P)]
            order = ["stock", "stock", *patches]
            rng.shuffle(order)
            got = [pipeline(binaries.get(arm, stock), stock, circuit, inputs, work / "run") for arm in order]
            if len({json.dumps(public) for _, public in got}) != 1:
                sys.exit(f"block {block}: the arms disagree on the public output")
            base = [s for arm, (s, _) in zip(order, got) if arm == "stock"]
            for arm, (s, _) in zip(order, got):
                seconds[arm].append(s)
            for name in patches:
                ratios[name].append(math.log(got[order.index(name)][0] / math.sqrt(base[0] * base[1])))
            floors.append(math.log(base[1] / base[0]))
            print(f"{block:>5} | {' '.join(order)} | {' '.join(f'{s:.3f}' for s, _ in got)} | {os.getloadavg()[0]:.2f}", flush=True)
        print(f"\nfluidex Rescue(2), {opts.blocks} blocks, three REP3 parties on loopback:")
        for name in patches:
            point, low, high, favorable = summary(ratios[name], f"boot:{name}" if len(patches) > 1 else "boot")
            print(f"  {name} vs stock: {point:.2f}% lower full-pipeline time, 95% interval [{low:.2f}%, {high:.2f}%], faster in {favorable}/{opts.blocks} blocks")
        floor, flow, fhigh, _ = summary(floors, "aa")
        print(f"  stock vs stock (A/A noise floor): {floor:.2f}% [{flow:.2f}%, {fhigh:.2f}%]")
        print("  median seconds: " + ", ".join(f"{arm} {statistics.median(values):.3f}" for arm, values in seconds.items()))
    finally:
        if not opts.work:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
