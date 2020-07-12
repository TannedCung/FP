#include <Stepper.h>
#include <Wire.h>
#include <Adafruit_MLX90614.h>


int in1=8;    // pin 
int in2=9;
int in3=10;
int in4=11;
float offset = 1.4
Adafruit_MLX90614 mlx = Adafruit_MLX90614();
float raw_temp, temp;
int proximityPin = 2;
// by default pin A5, A4 in arduino connect to SCL and SDA in MLX90614

Stepper myStepper(200,in1,in2,in3,in4);
String command;

void setup()
{
  Wire.setClock(10000);
  pinMode(in1,OUTPUT);
  pinMode(in2,OUTPUT);
  pinMode(in3,OUTPUT);
  pinMode(in4,OUTPUT);
  pinMode(proximityPin, INPUT);
  mlx.begin();

  //Open Serial port:
  Serial.begin(9600);

}
void mask_out()
{ 
  myStepper.setSpeed(300);
  myStepper.step(2600);
  delay(5000);
  myStepper.setSpeed(300);
  myStepper.step(-2600);
  command = "in";
  // Serial.println("done");
   }

void get_temp()
{
  if (digitalRead(proximityPin)==0)
  {
    raw_temp = mlx.readObjectTempC()+offset;
    Serial.println(raw_temp);
    delay(500);
  }
}

void loop()
{
  get_temp();
  if(Serial.available())
  {
    command= Serial.readString();
  }
  
  if (command == "out")
  {
    mask_out();
    Serial.println("on");
    
  }

}
