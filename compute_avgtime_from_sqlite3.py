import sqlite3
import re

# Code adapted from https://github.com/ezyang/nvprof2json

def munge_time(t):
    """Take a timestamp from nsys (ns) and convert it into us (the default for chrome://tracing)."""
    # For strict correctness, divide by 1000, but this reduces accuracy.
    return t / 1000.

# For reference of the schema, see
# https://docs.nvidia.com/nsight-systems/UserGuide/index.html#exporter-sqlite-schema

def parse_cupti_kernel_events(conn: sqlite3.Connection, strings: dict, traceEvents: list, kernel_name: str):
    """
    Copied from the docs:
    CUPTI_ACTIVITY_KIND_KERNEL
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER   NOT NULL,                    -- Event end timestamp (ns).
    deviceId                    INTEGER   NOT NULL,                    -- Device ID.
    contextId                   INTEGER   NOT NULL,                    -- Context ID.
    streamId                    INTEGER   NOT NULL,                    -- Stream ID.
    correlationId               INTEGER,                               -- REFERENCES CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)
    globalPid                   INTEGER,                               -- Serialized GlobalId.
    demangledName               INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Kernel function name w/ templates
    shortName                   INTEGER   NOT NULL,                    -- REFERENCES StringIds(id) -- Base kernel function name
    mangledName                 INTEGER,                               -- REFERENCES StringIds(id) -- Raw C++ mangled kernel function name
    launchType                  INTEGER,                               -- REFERENCES ENUM_CUDA_KRENEL_LAUNCH_TYPE(id)
    cacheConfig                 INTEGER,                               -- REFERENCES ENUM_CUDA_FUNC_CACHE_CONFIG(id)
    registersPerThread          INTEGER   NOT NULL,                    -- Number of registers required for each thread executing the kernel.
    gridX                       INTEGER   NOT NULL,                    -- X-dimension grid size.
    gridY                       INTEGER   NOT NULL,                    -- Y-dimension grid size.
    gridZ                       INTEGER   NOT NULL,                    -- Z-dimension grid size.
    blockX                      INTEGER   NOT NULL,                    -- X-dimension block size.
    blockY                      INTEGER   NOT NULL,                    -- Y-dimension block size.
    blockZ                      INTEGER   NOT NULL,                    -- Z-dimension block size.
    staticSharedMemory          INTEGER   NOT NULL,                    -- Static shared memory allocated for the kernel (B).
    dynamicSharedMemory         INTEGER   NOT NULL,                    -- Dynamic shared memory reserved for the kernel (B).
    localMemoryPerThread        INTEGER   NOT NULL,                    -- Amount of local memory reserved for each thread (B).
    localMemoryTotal            INTEGER   NOT NULL,                    -- Total amount of local memory reserved for the kernel (B).
    gridId                      INTEGER   NOT NULL,                    -- Unique grid ID of the kernel assigned at runtime.
    sharedMemoryExecuted        INTEGER,                               -- Shared memory size set by the driver.
    graphNodeId                 INTEGER,                               -- REFERENCES CUDA_GRAPH_EVENTS(graphNodeId)
    sharedMemoryLimitConfig     INTEGER                                -- REFERENCES ENUM_CUDA_SHARED_MEM_LIMIT_CONFIG(id)
    """
    for row in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        if strings[row["shortName"]] == kernel_name and row["deviceId"] == 0:
            event = {
                    "name": strings[row["shortName"]],
                    "ph": "X", # Complete Event (Begin + End event)
                    "cat": "cuda",
                    "ts": munge_time(row["start"]),
                    "dur": munge_time(row["end"] - row["start"]),
                    "tid": "Stream {}".format(row["streamId"]),
                    "pid": "Device {}".format(row["deviceId"]),
                    "args": {
                        # TODO: More
                        },
                    }
            traceEvents.append(event)

def compute_avg_time(kernel_dict):
    avg_time = 0.0
    times = len(kernel_dict)
    print(times)
    for i in range(times):
        avg_time += float(kernel_dict[i]["dur"])
    return avg_time / times


def nsys2json():
    conn = sqlite3.connect("./xxx.sqlite")
    conn.row_factory = sqlite3.Row

    strings = {}
    for r in conn.execute("SELECT id, value FROM StringIds"):
        strings[r["id"]] = r["value"]

    traceEvents = []
    parse_cupti_kernel_events(conn, strings, traceEvents, "mmha")

    # make the timelines appear in pid and tid order
    traceEvents.sort(key=lambda x: (x["pid"], x["tid"]))

    avg_time = compute_avg_time(traceEvents)
    print(avg_time)

nsys2json()