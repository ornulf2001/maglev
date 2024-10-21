import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV data
data = pd.read_csv('output.csv')
#meanBx=data["meanBx"][10] #These doesnt change so i pick a value somewhere at the start, here 10
#meanBy=data["meanBy"][10]
#meanBz=data["meanBz"][10]
bx=data["bx"]
by=data["by"]
bz=data["bz"]



### SNR for a horizon of 400:
N=400
varX, varY, varZ = 0,0,0
rmsX,rmsY,rmsZ = 0,0,0
meanBx =  np.mean(bx[:N])
meanBy =  np.mean(by[:N])
meanBz =  np.mean(bz[:N])



for i in range(N):
    varX += (bx[i]-meanBx)**2
    varY += (by[i]-meanBy)**2
    varZ += (bz[i]-meanBz)**2
varX = varX/(N-1)
varY = varY/(N-1)
varZ = varZ/(N-1)

def snr(list,N):
    sum_square=0
    for i in range(N):
        sum_square+=list[i]**2
    snr=(1/N * sum_square)**0.5
    return snr

P_signalX=snr(bx,N)**2
P_signalY=snr(by,N)**2
P_signalZ=snr(bz,N)**2
P_noiseX = varX
P_noiseY = varY
P_noiseZ = varZ

snrX=P_signalX/P_noiseX
snrY=P_signalY/P_noiseY
snrZ=P_signalZ/P_noiseZ



print(snrX,snrY,snrZ)








# Plot each column from the CSV file
def plotting():
    plt.figure()

    # Assuming the columns are named 'bx', 'by', 'bz', 'ux', 'uy'
    for column in data.columns:
        plt.plot(data[column], label=column)

    # Add labels and legend
    plt.xlabel('Sample Index')
    plt.ylabel('Sensor Values')
    plt.title('Sensor Data Over Time')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

#plotting()