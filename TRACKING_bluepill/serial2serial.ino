class Nano {
  uint8_t data[4];
  public:
  void update() {
    if (Serial2.available() == 4) {
      for (int i(0); i < 4; ++i)
        data[i] = Serial2.read();
      Serial1.write(data, 4);
      Serial1.flush();
    }
  }
};

class STM {
  uint8_t data[4];
  public:
  void update() {
    if (Serial1.available() == 4) {
      for (int i(0); i < 4; ++i)
        data[i] = Serial1.read();
      Serial2.write(data, 4); 
      Serial2.flush();
    }
  }
};

STM stm32;
Nano jetson;

void setup() {
  pinMode(13, OUTPUT);
  Serial.begin(115200);
  Serial1.begin(115200);
  Serial2.begin(115200);
}

void loop() {
  stm32.update();
  jetson.update();
}
