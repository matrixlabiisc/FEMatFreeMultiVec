import numpy as np
import pandas as pd
import re
import os
# import heatmap


def flopsCalculate(filename, nData, timeString, useGetTime):

    filepath = "./" + filename

    with open(filepath, 'r') as f:
        _data = f.readlines()

    _data = list(map(str.strip, _data))

    pattern = '"ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time","Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"'

    flops_list = ["sm__sass_thread_inst_executed_op_dadd_pred_on.sum", "sm__sass_thread_inst_executed_op_dfma_pred_on.sum", "sm__sass_thread_inst_executed_op_dmul_pred_on.sum", "sm__sass_thread_inst_executed_op_fadd_pred_on.sum", "sm__sass_thread_inst_executed_op_ffma_pred_on.sum", "sm__sass_thread_inst_executed_op_fmul_pred_on.sum", "sm__sass_thread_inst_executed_op_hadd_pred_on.sum", "sm__sass_thread_inst_executed_op_hfma_pred_on.sum", "sm__sass_thread_inst_executed_op_hmul_pred_on.sum"]

    cols = re.findall(r'"([^"]*)"', pattern)
    idx = _data.index(pattern)
    data = _data[idx+1:]

    for i in range(len(data)):
        data[i] = re.findall(r'"([^"]*)"', data[i])

    df = pd.DataFrame(data, columns = cols)

    int_cols = ['ID', 'Process ID', 'Context', 'Stream']

    for col in int_cols:
        df[col] = df[col].astype('int')

    df["Metric Value"] = df["Metric Value"].apply(lambda x: np.double(x.replace(',', '')))

    sm_cycles_elapsed = np.array(df[df["Metric Name"] == "sm__cycles_elapsed.avg"]["Metric Value"])
    sm_cycles_per_second = np.array(df[df["Metric Name"] == "sm__cycles_elapsed.avg.per_second"]["Metric Value"])
    _dram = np.array(df[df["Metric Name"] == "dram__bytes.sum"]["Metric Value"])
    _tensorflops = np.array(df[df["Metric Name"] == "sm__inst_executed_pipe_tensor.sum"]["Metric Value"])

    size = len(sm_cycles_elapsed)
    nKernels = size // nData

    time = np.zeros(nData)
    stdDev = np.zeros(nData)
    flops = np.zeros(nData)
    tensorflops = np.zeros(nData)
    dram = np.zeros(nData)
    _flops = []

    if (useGetTime):
        # Time getTime
        if (filename.endswith("rank0.txt")):
            n = 0
            for line in _data:
                if timeString[0] in line:
                    time[n] = np.double(line.split(":")[-1])

                if timeString[1] in line:
                    stdDev[n] = np.double(line.split(":")[-1])
                    n += 1
    else:
        # Time Profiler
        _time = sm_cycles_elapsed / sm_cycles_per_second
        for i in range(size):
            time[i // nKernels] += _time[i]

    # CC FLOPS
    for i in flops_list:
        _flops.append(np.array(df[df["Metric Name"] == i]["Metric Value"]))

    for i in range(len(flops_list)):
        for j in range(size):
            flops[j // nKernels] += (2 if i in [1, 4, 7] else 1) * _flops[i][j]

    # Tensor FLOPS
    for i in range(size):
        tensorflops[i // nKernels] += _tensorflops[i]

    # DRAM
    for i in range(size):
        dram[i // nKernels] += _dram[i]

    return [time, stdDev, flops, tensorflops, dram]


######################################################
###################### Main ##########################
######################################################

# Start of File Name
fileStr = "flopsHX_Uniform_FP64Comm_CM_poly8_vec512"

timeString = ("GPU Mean Time MatrixFree", "GPU StdDev Time MatrixFree")
# timeString = ("GPU Mean Time CellMatrix", "GPU StdDev Time CellMatrix")
# timeString = ("GPU Mean Time Dealii", "GPU StdDev Time Dealii")

# No of vectors/mpi to be profiled
nData = 1           # 1

# Time Flag
useGetTime = True

savePath = "./test/"

# { fileName : [times(ms if getTime else s), stdDev(ms), flops, tensorflops, dram(bytes), flopsPerSecond(TFLOPS/sec), dramBandwidth(GB/sec)] }
fileDict = {}
metric_list = ["times", "stdDev", "flops", "dram"]

for f in os.scandir():
    if f.is_file() and f.name.startswith(fileStr):
        f_out = "_".join(f.name.split("_")[:-1])

        time = np.zeros(nData)
        stdDev = np.zeros(nData)
        flops = np.zeros(nData)
        tensorflops = np.zeros(nData)
        dram = np.zeros(nData)

        fileDict.setdefault(f_out, [time, stdDev, flops, tensorflops, dram])

        metrics = flopsCalculate(f.name, nData, timeString, useGetTime)

        for i in range(2, 5):
            fileDict[f_out][i] += metrics[i]

        for i in range(nData):
            fileDict[f_out][0][i] = (1 if useGetTime else 1e3) * max(fileDict[f_out][0][i], metrics[0][i])   # ms
            fileDict[f_out][1][i] = (1 if useGetTime else 1e3) * max(fileDict[f_out][1][i], metrics[1][i])   # ms

        print(f"{f_out} : {fileDict[f_out][2]}")



# Add flopsPerSecond and dramBandwidth to the fileDict
# for key in fileDict.keys():
#     fileDict[key].append(fileDict[key][2] / fileDict[key][0] * 1e3 / 1e12)      # TFLOPS/sec
#     fileDict[key].append(fileDict[key][4] / fileDict[key][0] * 1e3 / 1e9)       # GB/sec


# for key in fileDict.keys():
#     np.savetxt(savePath + key + "_flops", fileDict[key][-2])
#     np.savetxt(savePath + key + "_times", fileDict[key][0])
#     np.savetxt(savePath + key + "_times_stdDev", fileDict[key][1])
