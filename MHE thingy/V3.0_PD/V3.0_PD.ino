#include <Tle493d_w2b6.h>
#include <Wire.h>

// IIR filter constants
#define ALPHA 0.06  // 0.6, 0.1
#define DALPHA 0.03 //0.03, 0.02

// LP filter
const float alpha = 0.1; // Smoothing factor (adjust as needed)

// Sample timer
const double f_s = 5000; // Hz
double prev_time = 1;
double current_time = 2;
const int T = round(1e6 / f_s);

int version_maglev=27;

  // // Sensors
  // Tle493d_a2b6 Sensor = Tle493d_a2b6(Tle493d::FASTMODE, Tle493d::TLE493D_A0);
  // // Four motor drivers - IN1 and IN2 are PWM inputs. Polarity of the output is set based on which one of them is active.
  // //// Motor driver 1 
  // #define MD1_IN1 3
  // #define MD1_IN2 2

  // //// Motor driver 2
  // #define MD2_IN1 5
  // #define MD2_IN2 4

  // //// Motor driver 3
  // #define MD3_IN1 7
  // #define MD3_IN2 6

  // //// Motor driver 4
  // #define MD4_IN1 9
  // #define MD4_IN2 8

  // Sensors
Tle493d_w2b6 Sensor = Tle493d_w2b6(Tle493d::FASTMODE, Tle493d::TLE493D_A0);
  // Four motor drivers - IN1 and IN2 are PWM inputs. Polarity of the output is set based on which one of them is active.
// Motor driver pins for four motors
#define MD1_IN1 2
#define MD1_IN2 3

#define MD2_IN1 4
#define MD2_IN2 5

#define MD3_IN1 8
#define MD3_IN2 9

#define MD4_IN1 6
#define MD4_IN2 7

// Counters
int initCounter = 0;
int loopCounter = 0;

// Measurements
double rawBx = 0, rawBy = 0, rawBz = 0;
double bx = 0, by = 0, bz = 0;
double bx_prev = 0, by_prev = 0, bz_prev = 0;
double dbx = 0, dby = 0, dbz = 0;
double dbx_prev = 0, dby_prev = 0, dbz_prev = 0;

// Mean of measurements
float meanBx = 0, meanBy = 0, meanBz = 0;

// Bias
float biasX = 0.121;
float biasY = 0.095;

// PID
double Kp = 800, Kd = 4; // 450,2 value for the double magnets, maybe not the best but it's stabilise, (800,2.6)
double ux = 0, uy = 0;
double ex = 0, ey = 0;
double dex = 0, dey = 0;

//NEW -- usage: PCA9548A(2) to select SCL2/SDA2 on bus (Wire to use i2C_0, Wire2 to use i2C_2)
void PCA9548A(TwoWire& bus, uint8_t track){
  bus.beginTransmission(0x70);  // PCA9548A address is 0x70
  bus.write(1 << track);        // send byte to select track
  bus.endTransmission();
}

void setup() {
  Serial.begin(115200);
  // while(!Serial);
  Serial.println("Initialization...");
  Wire.begin();
  PCA9548A(Wire, 0);
  // Initialize sensor
  Sensor.begin();

  // Initialize motor drivers
  analogWriteResolution(8);

  //// Set pin mode
  pinMode(MD1_IN1, OUTPUT);
  pinMode(MD1_IN2, OUTPUT);

  pinMode(MD2_IN1, OUTPUT);
  pinMode(MD2_IN2, OUTPUT);
  
  pinMode(MD3_IN1, OUTPUT);
  pinMode(MD3_IN2, OUTPUT);
  
  pinMode(MD4_IN1, OUTPUT);
  pinMode(MD4_IN2, OUTPUT);

  // Set PWM frequency to avoid audible tones
  analogWriteFrequency(MD1_IN1, 32258);
  analogWriteFrequency(MD1_IN2, 32258);

  analogWriteFrequency(MD2_IN1, 32258);
  analogWriteFrequency(MD2_IN2, 32258);

  analogWriteFrequency(MD3_IN1, 32258);
  analogWriteFrequency(MD3_IN2, 32258);

  analogWriteFrequency(MD4_IN1, 32258);
  analogWriteFrequency(MD4_IN2, 32258);

  //// Start with motor drivers off
  digitalWrite(MD1_IN1, LOW);
  digitalWrite(MD1_IN2, LOW);
  
  digitalWrite(MD2_IN1, LOW);
  digitalWrite(MD2_IN2, LOW);
  
  digitalWrite(MD3_IN1, LOW);
  digitalWrite(MD3_IN2, LOW);

  digitalWrite(MD4_IN1, LOW);
  digitalWrite(MD4_IN2, LOW);

  // We adjust the I2C clock speed to reach maximum sample time (5.2kHz). This HAS to be done after initializing the sensors!
  Wire.begin();
  Wire.setClock(1000000); // Set I2C frequency to 400kHz

  // Compute sensor mean
  while(initCounter < 1000) {
    delay(1);
    PCA9548A(Wire, 0);
    Sensor.updateData();
    meanBx += Sensor.getX()/1000;
    meanBy += Sensor.getY()/1000;
    meanBz += Sensor.getZ()/1000;
    
    initCounter++;
  }
}


int map_from_u_to_pwm(int u){
  double a = (165.0 - 135.0)/255.0;
  double b = 135.0;

  if(u > 0){
    return round(a*u + b); 
  }
  else{
    return round(a*u - b);
  }
}

void change_input(int ux, int uy){

  int ux_pwm = map_from_u_to_pwm(ux);
  int uy_pwm = map_from_u_to_pwm(uy);

  if(ux > 0){
    analogWrite(MD3_IN1, ux_pwm);
    analogWrite(MD3_IN2, 0);

    analogWrite(MD2_IN1, 0);
    analogWrite(MD2_IN2, ux_pwm);
  }
  else{
    analogWrite(MD3_IN1, 0);
    analogWrite(MD3_IN2, -ux_pwm);

    analogWrite(MD2_IN1, -ux_pwm);
    analogWrite(MD2_IN2, 0);
  }

  if(uy > 0){
    analogWrite(MD1_IN1, uy_pwm);
    analogWrite(MD1_IN2, 0);
    
    analogWrite(MD4_IN1, 0);
    analogWrite(MD4_IN2, uy_pwm);
  }
  else{
    analogWrite(MD1_IN1, 0);
    analogWrite(MD1_IN2, -uy_pwm);
    
    analogWrite(MD4_IN1, -uy_pwm);
    analogWrite(MD4_IN2, 0);
  }
}


void loop() {
  if(micros() - prev_time >= T){
    current_time = micros();
      // Get measurement  
      PCA9548A(Wire, 0);    
      Sensor.updateData();
      rawBx = Sensor.getX() - meanBx - 0.87/255*ux + biasX; // 0.08393; // 0.25 value for the double magnet, just guess the value but don't know what is the bias is Compensate for bias by permanent and electromagnets
      rawBy = Sensor.getY() - meanBy - 0.82/255*uy + biasY; //0.0872; //0.125, 0.08/0.09, (x0.08393, y0.0872)
      rawBz = Sensor.getZ() - meanBz;

      bx = ALPHA * rawBx + (1.0 - ALPHA) * bx_prev;
      by = ALPHA * rawBy + (1.0 - ALPHA) * by_prev;
      bz = ALPHA * rawBz + (1.0 - ALPHA) * bz_prev;

      dbx = DALPHA * ((bx - bx_prev)/((current_time - prev_time + 10)/1000000.0)) + (1.0 - DALPHA) * dbx_prev;
      dby = DALPHA * ((by - by_prev)/((current_time - prev_time + 10)/1000000.0)) + (1.0 - DALPHA) * dby_prev;

      bx_prev = bx;
      by_prev = by;
      bz_prev = bz;
      
      dbx_prev = dbx;
      dby_prev = dby;

      if(abs(bz) > 3){
        ex = -bx;
        ey = -by;

        dex = -dbx;
        dey = -dby;

        ux = constrain(Kp*ex + Kd*dex, -255, 255);
        uy = constrain(Kp*ey + Kd*dey, -255, 255);
        change_input(ux, uy);
      }
      else{
          ux = 0;
          uy = 0;
          change_input(ux, uy);
      }

      if( loopCounter % 100 == 0){
        //Serial.print(meanBx);
        //Serial.print(",");
        //Serial.print(meanBy);
        //Serial.print(",");
        //Serial.print(meanBz);
        //Serial.print(",");
        Serial.print(bx);
        Serial.print(',');
        Serial.print(by);
        Serial.print(',');
        Serial.println(bz);
        //Serial.print(',');
        //Serial.print(ux);
        //Serial.print(',');
        //Serial.println(uy);
      }
    loopCounter++;

    prev_time = current_time;
  }
}
