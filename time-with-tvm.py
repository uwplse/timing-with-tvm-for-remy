import tvm
from tvm import relay
import sys
import numpy as np


w_file = sys.argv[1]
h_file = sys.argv[2]

w_data = np.load(w_file).astype('float32')
h_data = np.load(h_file).astype('float32')

m = w_data.shape[0]
n = h_data.shape[1]
r = w_data.shape[1]

w = relay.var('w', shape=(m,r), dtype='float32')
h = relay.var('h', shape=(r,n), dtype='float32')
program = relay.nn.dense(w, h)
program = relay.sum(program, axis=None)
program = relay.Function([w,h], program)
module = relay.Module.from_expr(program)

_, tvm_module, _ = relay.build(module, 'llvm')

timer = tvm_module.time_evaluator(tvm_module.entry_name, tvm.cpu(0))
# TODO why is it expecting this output size?
output_tvm = tvm.nd.array(np.empty((m,r)).astype('float32'))
res = timer(tvm.nd.array(w_data), tvm.nd.array(h_data), output_tvm)
print(res)


