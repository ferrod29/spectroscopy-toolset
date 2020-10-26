import numpy as np
import json
path = 'CompiledScan_xxx.scan'
​
with open(path) as f:
    compiledScan = json.load(f)
​
compiledScan[0] = np.asarray(compiledScan[0],dtype=np.float64) # wavelengths
compiledScan[1] = np.asarray(compiledScan[1],dtype=np.int32) # time delays
compiledScan[2] = np.asarray(compiledScan[2],dtype=np.float64) # delta OD
compiledScan[3] = np.asarray(compiledScan[3],dtype=np.float64) # bkg
​
#Create arrays for the data
wavelengths = np.asarray(compiledScan[0])
timeDelays = np.asarray(compiledScan[1])
​
#Calculate dOD
dOD = compiledScan[2]
dOD = dOD.T
Collapse




