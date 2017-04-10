import os
import re
import sys
import tempfile
import tensorflow as tf

debug_messages = False

def vlog(level):
  os.environ['TF_CPP_MIN_VLOG_LEVEL'] = str(level)

# this helper is here in case we later want to capture huge stderr that doesn't fit in RAM
class TemporaryFileHelper:
  """Provides a way to fetch contents of temporary file.""" 
  def __init__(self, temporary_file):
    self.temporary_file = temporary_file
  def getvalue(self):
    return open(self.temporary_file.name).read() 


STDOUT=1
STDERR=2
class capture_stderr:
  """Utility to capture output, use as follows
     with util.capture_stderr() as stderr:
        sess = tf.Session()
    print("Captured:", stderr.getvalue()).
    """

  def __init__(self, fd=STDERR):
    self.fd = fd
    self.prevfd = None

  def __enter__(self):
    t = tempfile.NamedTemporaryFile()
    self.prevfd = os.dup(self.fd)
    os.dup2(t.fileno(), self.fd)
    return TemporaryFileHelper(t)

  def __exit__(self, exc_type, exc_value, traceback):
    os.dup2(self.prevfd, self.fd)


################################################################################
# LOG_MEMORY_PARSING
################################################################################
# Until https://github.com/tensorflow/tensorflow/issues/6716 is resolved, the
# reliable way to get access to tensor deallocation information is to parse
# __LOG_MEMORY__ from VLOG print statements. This is sensitive to print order
# run unbuffered to prevent interleaving:
#   python -u script.py

# Regex'es to parse __LOG_MEMORY__ statements
# Each regex is preceded by an example of line it's meant to pass

# I 5143420588.000000 file tensorflow/core/framework/log_memory.cc:41] __LOG_MEMORY__ MemoryLogTensorAllocation { step_id: -6 kernel_name: "Unknown (from Proto)" tensor { dtype: DT_INT32 shape { dim { size: 3 } } allocation_description { requested_bytes: 12 allocated_bytes: 12 allocator_name: "cpu" allocation_id: 3 has_single_reference: true ptr: 29496256 } } }
tensor_allocation_regex = re.compile("""MemoryLogTensorAllocation.*?step_id: (?P<step_id>[-0123456789]+).*kernel_name: \"(?P<kernel_name>[^"]+)\".*?allocated_bytes: (?P<allocated_bytes>\d+).*allocator_name: \"(?P<allocator_name>[^"]+)\".*allocation_id: (?P<allocation_id>\d+).*""")

# I 6795349363.000000 file tensorflow/core/framework/log_memory.cc:41] __LOG_MEMORY__ MemoryLogRawAllocation { step_id: -3 operation: "TF_AllocateTensor" num_bytes: 1000000 ptr: 80910752 allocation_id: 99 allocator_name: "cpu" }
raw_allocation_regex = re.compile("""MemoryLogRawAllocation.*?step_id: (?P<step_id>[-0123456789]+).*operation: \"(?P<kernel_name>[^"]+)\".*?num_bytes: (?P<allocated_bytes>\d+).*allocation_id: (?P<allocation_id>\d+).*allocator_name: "(?P<allocator_name>[^"]+)".*""")

# I 5143420588.000000 file tensorflow/core/framework/log_memory.cc:41] __LOG_MEMORY__ MemoryLogTensorOutput { step_id: 1 kernel_name: "Const" tensor { dtype: DT_INT32 shape { dim { size: 3 } } allocation_description { requested_bytes: 12 allocated_bytes: 12 allocator_name: "cpu" allocation_id: 3 ptr: 29496256 } } }
# 2017-01-26 10:13:30: I tensorflow/core/framework/log_memory.cc:35] __LOG_MEMORY__ MemoryLogTensorOutput { step_id: 2 kernel_name: "a0" tensor { dtype: DT_FLOAT shape { dim { size: 250000 } } allocation_description { requested_bytes: 1000000 allocated_bytes: 1000192 allocator_name: "gpu_bfc" allocation_id: 3 ptr: 30076651520 } } }
#tensor_output_regex = re.compile("""MemoryLogTensorOutput.* step_id: (?P<step_id>[-0123456789]+) kernel_name: \"(?P<kernel_name>[^"]+).*allocated_bytes: (?P<allocated_bytes>\d+).*allocation_id: (?P<allocation_id>\d+).*""")   
tensor_output_regex = re.compile("""MemoryLogTensorOutput.* step_id: (?P<step_id>[-0123456789]+) kernel_name: \"(?P<kernel_name>[^"]+).*allocated_bytes: (?P<allocated_bytes>\d+).*allocator_name: \"(?P<allocator_name>[^"]+)\".*allocation_id: (?P<allocation_id>\d+).*""")

# some Shape lines are missing bytes info so have separate regex for them
# I 5162643141.000000 file tensorflow/core/framework/log_memory.cc:41] __LOG_MEMORY__ MemoryLogTensorOutput { step_id: 5 kernel_name: "gradients/Shape" tensor { dtype: DT_INT32 shape { dim { } } } }
tensor_output_regex_no_bytes = re.compile("""MemoryLogTensorOutput.* step_id: (?P<step_id>[-0123456789]+) kernel_name: \"(?P<kernel_name>[^"]+).*""")


# 5143420588.000000 file tensorflow/core/framework/log_memory.cc:41] __LOG_MEMORY__ MemoryLogTensorDeallocation { allocation_id: 2 allocator_name: "cpu" }
tensor_deallocation_regex = re.compile("""allocation_id: (?P<allocation_id>\d+).*allocator_name: \"(?P<allocator_name>[^"]+)\".*""")

# I 6796000229.000000 file tensorflow/core/framework/log_memory.cc:41] __LOG_MEMORY__ MemoryLogRawDeallocation { step_id: -3 operation: "TensorFlow C Api" allocation_id: 177 allocator_name: "cpu" }
raw_deallocation_regex = re.compile("""allocation_id: (?P<allocation_id>\d+).*allocator_name: \"(?P<allocator_name>[^"]+)\".*""")

# I 5143420588.000000 file tensorflow/core/framework/log_memory.cc:41] __LOG_MEMORY__ MemoryLogStep { step_id: 1 handle: "->Print:0//0/;0" }
tensor_logstep_regex = re.compile("""MemoryLogStep.*?step_id: (?P<step_id>[-0123456789]+).*""")


def _parse_logline(l):
    if 'MemoryLogTensorOutput' in l:
        m = tensor_output_regex.search(l)
        if not m:
            m = tensor_output_regex_no_bytes.search(l)

        assert m, l
        d = m.groupdict()
        d["type"] = "MemoryLogTensorOutput"
            
    elif 'MemoryLogTensorAllocation' in l:
        m = tensor_allocation_regex.search(l)

        # Broadcast args give weird allocation messages without size, ignore
        # I tensorflow/core/framework/log_memory.cc:35] __LOG_MEMORY__ MemoryLogTensorAllocation { step_id: 2 kernel_name: "gradients/node_5_grad/BroadcastGradientArgs" tensor { dtype: DT_INT32 shape { dim { } } } }
        if not m:
            return {"type": "MemoryLogTensorAllocation", "line": l,
                    "allocation_id": "-1"}

        assert m, l
        d = m.groupdict()
        d["type"] = "MemoryLogTensorAllocation"
        if debug_messages:
            print("Got allocation for %s, %s"%(d["allocation_id"], d["kernel_name"]))
    elif 'MemoryLogTensorDeallocation' in l:
        m = tensor_deallocation_regex.search(l)
        assert m, l
        d = m.groupdict()
        d["type"] = "MemoryLogTensorDeallocation"
        if debug_messages:
            print("Got deallocation for %s"%(d["allocation_id"]))
    elif 'MemoryLogStep' in l:
        m = tensor_logstep_regex.search(l)
        assert m, l
        d = m.groupdict()
        d["type"] = "MemoryLogStep"
    elif 'MemoryLogRawAllocation' in l:
        m = raw_allocation_regex.search(l)
        assert m, l
        d = m.groupdict()
        d["type"] = "MemoryLogRawAllocation"
    elif 'MemoryLogRawDeallocation' in l:
        m = raw_deallocation_regex.search(l)
        assert m, l
        d = m.groupdict()
        d["type"] = "MemoryLogRawDeallocation"
    else:
        assert False, "Unknown log line: "+l
        
    if not "allocation_id" in d:
        d["allocation_id"] = "-1"

    d["line"] = l
    return d

def memory_timeline(log):
    if hasattr(log, 'getvalue'):
        log = log.getvalue()
    
    def unique_alloc_id(line):
        if line["allocation_id"] == "-1":
            return "-1"
        return line["allocation_id"]+"-"+line["allocator_name"]
    
    def get_alloc_names(line):
        alloc_id = unique_alloc_id(line)
        for entry in reversed(allocation_map.get(alloc_id, [])):
            kernel_name = entry.get("kernel_name", "unknown")
            if not "unknown" in kernel_name:
                return kernel_name+"("+unique_alloc_id(line)+")"
        # couldn't find an allocation message with name of kernel
        return "("+alloc_id+")"

    def get_alloc_bytes(line):
        for entry in allocation_map.get(unique_alloc_id(line), []):
            if "allocated_bytes" in entry:
                return entry["allocated_bytes"]
        return "0"

    def get_alloc_type(line):
        for entry in allocation_map.get(unique_alloc_id(line), []):
            if "allocator_name" in entry:
                return entry["allocator_name"]
        return "0"

    parsed_lines = []
    for l in log.split("\n"):
        if 'LOG_MEMORY' in l: # and not 'step_id: -6' in l:
            parsed_lines.append(_parse_logline(l))

    allocation_map = {} # map of <allocation_id>-<allocator_name>->parsed_logline of allocation
    for line in parsed_lines:
        if (line["type"] == "MemoryLogTensorAllocation" or line["type"] == "MemoryLogRawAllocation" or
            line["type"] == "MemoryLogTensorOutput"):
            allocation_map.setdefault(unique_alloc_id(line), []).append(line)
    if debug_messages:
        print(allocation_map)
    result = []
    for i, line in enumerate(parsed_lines):
        # skip lines without allocation_id, ie lines like
        # I tensorflow/core/framework/log_memory.cc:35] __LOG_MEMORY__ MemoryLogStep { step_id: 2 handle: "->/gradients/a1_grad/TanhGrad/0/;1" }

        if int(line["allocation_id"]) == -1:
            continue
        alloc_names = get_alloc_names(line)
        # if line doesn't specify bytes, look in history if there was corresponding TensorOutput or TensorAllocation msg
        if int(line.get('allocated_bytes', -1)) < 0:
            alloc_bytes = get_alloc_bytes(line)
        else:
            alloc_bytes = line.get('allocated_bytes', -1)
        alloc_type = get_alloc_type(line)
        if line["type"] == "MemoryLogTensorOutput":
            continue
        if line["type"] == "MemoryLogTensorDeallocation" or line["type"]=="MemoryLogRawDeallocation":
            alloc_bytes = "-" + alloc_bytes
        result.append((i, alloc_names, alloc_bytes, alloc_type))
    return result

def peak_memory(log, gpu_only=False):
    """Peak memory used across all devices."""
    peak_memory = -123456789 # to catch bugs
    total_memory = 0
    for record in memory_timeline(log):
        i, kernel_name, allocated_bytes, allocator_type = record
        allocated_bytes = int(allocated_bytes)
        if gpu_only:
            if not allocator_type.startswith("gpu"):
                continue
        total_memory += allocated_bytes
        peak_memory = max(total_memory, peak_memory)
    return peak_memory
    
def print_memory_timeline(log, gpu_only=False, ignore_less_than_bytes=0):
      
    total_memory = 0
    for record in memory_timeline(log):
        i, kernel_name, allocated_bytes, allocator_type = record
        allocated_bytes = int(allocated_bytes)
        if gpu_only:
            if not allocator_type.startswith("gpu"):
                continue
        if abs(allocated_bytes)<ignore_less_than_bytes:
            continue  # ignore small allocations
        total_memory += allocated_bytes
        print("%9d %42s %11d %11d %s"%(i, kernel_name, allocated_bytes, total_memory, allocator_type))

import matplotlib.pyplot as plt
def plot_memory_timeline(log, gpu_only=False, ignore_less_than_bytes=1000):
      
    total_memory = 0
    timestamps = []
    data = []
    current_time = 0
    for record in memory_timeline(log):
        timestamp, kernel_name, allocated_bytes, allocator_type = record
        allocated_bytes = int(allocated_bytes)
        
        if abs(allocated_bytes)<ignore_less_than_bytes:
            continue  # ignore small allocations
        if gpu_only:
            if not record[3].startswith("gpu"):
                continue
        timestamps.append(current_time-.00000001)
        data.append(total_memory)
        total_memory += int(record[2])
        timestamps.append(current_time)
        data.append(total_memory)
        current_time+=1
    plt.plot(timestamps, data)

################################################################################
# smart initialize
################################################################################

def smart_initialize(variables=None, sess=None):
  """Initializes all uninitialized variables in correct order. Initializers
  are only run for uninitialized variables, so it's safe to run this multiple
  times.
  Args:
      sess: session to use. Use default session if None.
  """

  from tensorflow.contrib import graph_editor as ge
  def make_initializer(var): 
    def f():
      return tf.assign(var, var.initial_value).op
    return f
  
  def make_noop(): return tf.no_op()

  def make_safe_initializer(var):
    """Returns initializer op that only runs for uninitialized ops."""
    return tf.cond(tf.is_variable_initialized(var), make_noop,
                   make_initializer(var), name="safe_init_"+var.op.name).op

  if not sess:
    sess = tf.get_default_session()
  g = tf.get_default_graph()

  if not variables:
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      
  safe_initializers = {}
  for v in variables:
    safe_initializers[v.op.name] = make_safe_initializer(v)
      
  # initializers access variable vaue through read-only value cached in
  # <varname>/read, so add control dependency to trigger safe_initializer
  # on read access
  for v in variables:
    var_name = v.op.name
    var_cache = g.get_operation_by_name(var_name+"/read")
    ge.reroute.add_control_inputs(var_cache, [safe_initializers[var_name]])

  sess.run(tf.group(*safe_initializers.values()))
    
  # remove initializer dependencies to avoid slowing down future variable reads
  for v in variables:
    var_name = v.op.name
    var_cache = g.get_operation_by_name(var_name+"/read")
    ge.reroute.remove_control_inputs(var_cache, [safe_initializers[var_name]])
