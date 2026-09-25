"""A unit's row group (uuid, audio) from HF, through an optional cache folder (RUN_CACHE): a cached unit is read
from disk, a fetched one is saved there first, so benchmark runs repeat on the same bytes without the network."""
import os, time
import pyarrow.parquet as pq
from units import DS


def read_unit(fs, handles, u, uid, log, cache=None):
    """The unit's table, or None after 6 failed fetches. handles keeps one open ParquetFile per shard."""
    path = os.path.join(cache, uid + ".parquet") if cache else None
    if path and os.path.exists(path):
        return pq.read_table(path)
    for attempt in range(6):
        try:
            if u[0] not in handles:
                handles[u[0]] = pq.ParquetFile(fs.open(f"datasets/{DS}/{u[0]}", "rb"))
            t = handles[u[0]].read_row_group(u[1], columns=["uuid", "audio"])
            break
        except Exception as e:
            handles.pop(u[0], None); log(f"fetch retry {uid}: {e}"); time.sleep(10 * (attempt + 1))
    else:
        return None
    if path:
        pq.write_table(t, path + ".tmp")
        os.replace(path + ".tmp", path)
    return t
