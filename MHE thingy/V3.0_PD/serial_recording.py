import serial
import time

# Serial
ser = serial.Serial('COM3', 115200, timeout=1)  
time.sleep(2)  

# Open the file
with open("output.csv", "w") as f:
    f.write("bx,by,bz\n")  # Write header
    while True:
        try:
            # Read a line from the serial port
            line = ser.readline().decode('utf-8').strip()
            if line:
                f.write(line + "\n")
                print(line) 
        except KeyboardInterrupt:
            print("Data logging stopped.")
            break
ser.close()
