import argparse, math, os, time
import threading, multiprocessing as mp

# ──────────────────────────────────────────────────────────────────────────────
def cpu_bound(n: int = 20_000_00) -> float:
    """
    純 Python 計算，完全不釋放 GIL。
    迴圈跑大量 sqrt 來吃 CPU。
    """
    acc = 0.0
    for i in range(n):
        acc += math.sqrt(i) % 1
    return acc


def run_with_threads(workers: int) -> None:
    ts = []
    for _ in range(workers):
        t = threading.Thread(target=cpu_bound)
        t.start()
        ts.append(t)

    for t in ts:
        t.join()


def worker_entry(_: int) -> None:
    cpu_bound()


def run_with_processes(workers: int) -> None:
    ps = []
    for _ in range(workers):
        p = mp.Process(target=worker_entry, args=(0,))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()


# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=("thread", "process"),
                        required=True,
                        help="thread = threading.Thread; process = multiprocessing.Process")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()

    print(f"PID={os.getpid()} MODE={args.mode} WORKERS={args.workers}")
    t0 = time.perf_counter()

    if args.mode == "thread":
        run_with_threads(args.workers)
    else:
        run_with_processes(args.workers)

    print(f"Done in {(time.perf_counter() - t0):.2f} s")


if __name__ == "__main__":
    main()