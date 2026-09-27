#include <Servo.h>

Servo myServo;

const int SERVO_PIN = 10;

// Center position and vibration range
const int CENTER = 90;
const int VIBRATION_AMOUNT = 8;

// How quickly the servo switches directions
const int VIBRATION_DELAY = 15;


// --------------------------------------------------
// Rapidly oscillate the servo
// --------------------------------------------------
void oscillate(unsigned long duration) {
  unsigned long startTime = millis();

  while (millis() - startTime < duration) {
    myServo.write(CENTER + VIBRATION_AMOUNT);
    delay(VIBRATION_DELAY);

    myServo.write(CENTER - VIBRATION_AMOUNT);
    delay(VIBRATION_DELAY);
  }

  // Return to center when finished
  myServo.write(CENTER);
}


// --------------------------------------------------
// Left: 8 seconds of vibration
// --------------------------------------------------
void left() {
  oscillate(8000);
}


// --------------------------------------------------
// Right: 4 seconds vibration,
//        1 second pause,
//        4 seconds vibration
// --------------------------------------------------
void right() {
  oscillate(4000);

  delay(1000);

  oscillate(4000);
}


// --------------------------------------------------
// Brake: 16 seconds of vibration
// --------------------------------------------------
void brake() {
  oscillate(16000);
}


// --------------------------------------------------
// Setup
// --------------------------------------------------
void setup() {
  Serial.begin(9600);
  myServo.attach(SERVO_PIN);
  myServo.write(CENTER);
  delay(500);
}


// --------------------------------------------------
// Loop
// --------------------------------------------------
void loop() {

  if (!Serial.available()) {
    return;
  }

  String command = Serial.readStringUntil('\n');
  command.trim();

  if (command == "left" || command == "soft left" || command == "hard left") {
    left();
  }
  else if (command == "right" || command == "soft right" || command == "hard right") {
    right();
  }
  else if (command == "brake" || command == "stop") {
    brake();
  }

}