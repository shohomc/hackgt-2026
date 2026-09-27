Servo myServo;

const int SERVO_PIN = 10;
const int CENTER = 90;
const int VIBRATION_AMOUNT = 8;
const int VIBRATION_DELAY = 15;

void setupHaptic() {
  myServo.attach(SERVO_PIN);
  myServo.write(CENTER);
  delay(500);
}

void oscillate(unsigned long duration) {
  unsigned long startTime = millis();
  while (millis() - startTime < duration) {
    myServo.write(CENTER + VIBRATION_AMOUNT);
    delay(VIBRATION_DELAY);
    myServo.write(CENTER - VIBRATION_AMOUNT);
    delay(VIBRATION_DELAY);
  }
  myServo.write(CENTER);
}

void hapticSoft() {
  oscillate(500);
  delay(500);
  oscillate(500);
}

void hapticHard() {
  oscillate(1500);
}

void hapticBrake() {
  oscillate(16000);
}
