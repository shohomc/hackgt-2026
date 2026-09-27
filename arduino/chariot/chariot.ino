#include <Servo.h>

// Wheel motor (TB6612) and vibration servo on one board.
// The Pi sends one line over USB serial at 9600 baud:
//   soft left, hard left, soft right, hard right, stop, straight
//
// Plug the Arduino USB port into the Pi. Motor pins are 5, 7, 8, 9.
// Servo signal is pin 10. Center the wheel by hand before powering on.

void setup() {
  Serial.begin(9600);
  setupMotor();
  setupHaptic();
  Serial.println("ready");
}

void loop() {
  String command = "";
  while (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      command = line;
    }
  }
  if (command.length() == 0) {
    return;
  }

  if (command == "soft left") {
    slightLeft();
    hapticSoft();
  } else if (command == "hard left") {
    hardLeft();
    hapticHard();
  } else if (command == "soft right") {
    slightRight();
    hapticSoft();
  } else if (command == "hard right") {
    hardRight();
    hapticHard();
  } else if (command == "stop") {
    brake();
    hapticBrake();
  } else if (command == "straight") {
    stopMotor();
  }
}
