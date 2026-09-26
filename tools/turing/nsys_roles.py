"""Where the wall time of a pipelined nsys profile goes: the kernel timeline split into moments when only the
encoder runs, only the decoder, both, or nothing, with each role's kernel time in those moments. A stream is
the encoder's when its kernels average over 100 us (as nsys_interfere.py). The decoder-only and idle moments
are where the decoder's latency sets the wall time; the both-moments where the two share the GPU.
usage: nsys_roles.py <profile.sqlite>"""
import collections, sqlite3, sys

db = sqlite3.connect(sys.argv[1])
rows = db.execute("SELECT start, end, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
per_stream = collections.defaultdict(list)
for s, e, st in rows:
    per_stream[st].append(e - s)
enc_streams = {st for st, d in per_stream.items() if sum(d) / len(d) > 100_000}
events = []                                            # (time, +1/-1, role)
for s, e, st in rows:
    role = "enc" if st in enc_streams else "dec"
    events += [(s, 1, role), (e, -1, role)]
events.sort()
active = {"enc": 0, "dec": 0}
spent = collections.Counter()
t0, last = events[0][0], events[0][0]
for t, d, role in events:
    state = ("both" if active["enc"] and active["dec"] else "enc only" if active["enc"]
             else "dec only" if active["dec"] else "idle")
    spent[state] += t - last
    active[role] += d
    last = t
span = last - t0
print(f"span {span / 1e9:.2f} s, encoder streams {sorted(enc_streams)}, decoder streams "
      f"{sorted(set(per_stream) - enc_streams)}")
for state in ("dec only", "enc only", "both", "idle"):
    print(f"  {state:9s} {spent[state] / 1e9:7.3f} s  {100 * spent[state] / span:5.1f}%")
for role, streams in (("encoder", enc_streams), ("decoder", set(per_stream) - enc_streams)):
    print(f"  {role} kernel time {sum(sum(per_stream[s]) for s in streams) / 1e9:.3f} s")
